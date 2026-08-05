import abc
import copy
import hashlib
import logging
import os
from pathlib import Path

import torch

from slime.utils.data import Dataset, _parse_generalized_path
from slime.utils.misc import load_function
from slime.utils.processing_utils import load_processor, load_tokenizer
from slime.utils.types import Sample

logger = logging.getLogger(__name__)

_SEQUENCE_CONFIGURATION_DEFAULTS = {
    "prompt_data": None,
    "input_key": "input",
    "label_key": None,
    "metadata_key": "metadata",
    "tool_key": "tools",
    "multimodal_keys": None,
    "apply_chat_template": False,
    "apply_chat_template_kwargs": {},
    "rollout_max_prompt_len": None,
    "rollout_shuffle": False,
    "rollout_seed": 42,
    "n_samples_per_prompt": 1,
    "buffer_filter_path": None,
}
_TOKENIZER_IDENTITY_FILES = (
    "added_tokens.json",
    "chat_template.jinja",
    "config.json",
    "preprocessor_config.json",
    "processor_config.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
)


def _sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def tokenizer_identity(checkpoint: object) -> dict[str, object]:
    value = str(checkpoint) if checkpoint is not None else None
    files = {}
    if value is not None:
        checkpoint_path = Path(value).expanduser()
        if checkpoint_path.is_dir():
            for filename in _TOKENIZER_IDENTITY_FILES:
                path = checkpoint_path / filename
                if path.is_file():
                    files[filename] = _sha256_file(str(path))
    return {"source": value, "files": files}


class DataSource(abc.ABC):
    @abc.abstractmethod
    def get_samples(self, num_samples: int) -> list[list[Sample]]:
        """
        Return num_samples samples
        """

    @abc.abstractmethod
    def add_samples(self, samples: list[list[Sample]]):
        """
        Add samples to the data source
        """

    @abc.abstractmethod
    def save(self, rollout_id):
        """
        Save the state of the data source
        """

    @abc.abstractmethod
    def load(self, rollout_id=None):
        """
        Load the state of the data source
        """

    @abc.abstractmethod
    def __len__(self) -> int:
        """
        Length of the data source. May change when samples are added/fetched.
        """


# TODO may further refactor data-loading part later
class RolloutDataSource(DataSource):
    def __init__(self, args):
        self.args = args

        self.epoch_id = 0
        self.sample_group_index = 0
        self.sample_index = 0
        self.sample_offset = 0
        # TODO remove this
        self.metadata = {}

        if args.rollout_global_dataset and args.prompt_data is not None:
            tokenizer = load_tokenizer(args.hf_checkpoint, trust_remote_code=True)
            processor = load_processor(args.hf_checkpoint, trust_remote_code=True)

            # TODO move (during the refactor)
            if (d := args.dump_details) is not None:
                tokenizer.save_pretrained(Path(d) / "tokenizer")
                if processor:
                    processor.save_pretrained(Path(d) / "processor")

            self.dataset = Dataset(
                args.prompt_data,
                tokenizer=tokenizer,
                processor=processor,
                max_length=args.rollout_max_prompt_len,
                prompt_key=args.input_key,
                multimodal_keys=args.multimodal_keys,
                label_key=args.label_key,
                metadata_key=args.metadata_key,
                tool_key=args.tool_key,
                apply_chat_template=args.apply_chat_template,
                apply_chat_template_kwargs=args.apply_chat_template_kwargs,
                seed=args.rollout_seed,
            )
            if self.args.rollout_shuffle:
                self.dataset.shuffle(self.epoch_id)
        else:
            self.dataset = None

        self._sequence_configuration = {
            name: getattr(args, name, default)
            for name, default in _SEQUENCE_CONFIGURATION_DEFAULTS.items()
        }
        self._sequence_configuration["tokenizer"] = tokenizer_identity(
            getattr(args, "hf_checkpoint", None)
        )

    def get_samples(self, num_samples):
        # TODO further improve code
        if self.dataset is not None:
            if self.sample_offset + num_samples <= len(self.dataset):
                prompt_samples = self.dataset.samples[self.sample_offset : self.sample_offset + num_samples]
                self.sample_offset += num_samples
            else:
                prompt_samples = self.dataset.samples[self.sample_offset :]
                num_samples -= len(prompt_samples)
                self.epoch_id += 1
                if self.args.rollout_shuffle:
                    self.dataset.shuffle(self.epoch_id)
                prompt_samples += self.dataset.samples[:num_samples]
                self.sample_offset = num_samples
        else:
            prompt_samples = [Sample() for _ in range(num_samples)]

        samples = []
        for prompt_sample in prompt_samples:
            group = []
            for _ in range(self.args.n_samples_per_prompt):
                sample = copy.deepcopy(prompt_sample)
                sample.group_index = self.sample_group_index
                sample.index = self.sample_index
                self.sample_index += 1
                group.append(sample)
            self.sample_group_index += 1
            samples.append(group)
        return samples

    def add_samples(self, samples: list[list[Sample]]):
        raise RuntimeError(f"Cannot add samples to {self.__class__.__name__}. This is a read-only data source.")

    def save(self, rollout_id):
        if not self.args.rollout_global_dataset:
            return

        state_dict = {
            "state_version": 2,
            "source_sha256": self._source_sha256(),
            "sample_count": len(self.dataset) if self.dataset is not None else None,
            "sequence_configuration": self._sequence_configuration,
            "sample_offset": self.sample_offset,
            "epoch_id": self.epoch_id,
            "sample_group_index": self.sample_group_index,
            "sample_index": self.sample_index,
            "metadata": self.metadata,
        }
        if hasattr(self, "buffer"):
            self._validate_buffer_samples(
                self.buffer,
                sample_index_limit=self.sample_index,
                group_index_limit=self.sample_group_index,
                check_against_buffer=False,
            )
            state_dict["buffer"] = self.buffer
        path = os.path.join(self.args.save, f"rollout/global_dataset_state_dict_{rollout_id}.pt")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        temporary_path = f"{path}.tmp"
        try:
            torch.save(state_dict, temporary_path)
            with open(temporary_path, "rb") as checkpoint:
                os.fsync(checkpoint.fileno())
            os.replace(temporary_path, path)
            directory_fd = os.open(
                os.path.dirname(path),
                os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
            )
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        except BaseException:
            if os.path.exists(temporary_path):
                os.unlink(temporary_path)
            raise

    def load(self, rollout_id=None):
        if not self.args.rollout_global_dataset:
            return

        if self.args.load is None:
            return

        path = os.path.join(self.args.load, f"rollout/global_dataset_state_dict_{rollout_id}.pt")
        if not os.path.exists(path):
            logger.info(f"Checkpoint {path} does not exist.")
            checkpoint_tracker = os.path.join(
                self.args.load,
                "latest_checkpointed_iteration.txt",
            )
            if os.path.isfile(checkpoint_tracker):
                raise FileNotFoundError(
                    "rollout data-source state is missing for checkpoint resume: "
                    f"{path}"
                )
            return

        logger.info(f"load metadata from {path}")
        logger.info(f"load metadata: {self.metadata}")
        state_dict = torch.load(path, weights_only=False)
        if not isinstance(state_dict, dict) or state_dict.get("state_version") != 2:
            raise ValueError(f"unsupported rollout data-source state: {path}")
        if state_dict.get("source_sha256") != self._source_sha256():
            raise ValueError("rollout training data changed since checkpoint save")
        current_sample_count = len(self.dataset) if self.dataset is not None else None
        if state_dict.get("sample_count") != current_sample_count:
            raise ValueError("rollout training sample count changed since checkpoint save")
        if state_dict.get("sequence_configuration") != self._sequence_configuration:
            raise ValueError("rollout data-source sequence configuration changed since checkpoint save")
        integer_fields = (
            "sample_offset",
            "epoch_id",
            "sample_group_index",
            "sample_index",
        )
        for field in integer_fields:
            value = state_dict.get(field)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"invalid {field} in rollout data-source state: {value!r}")
        if self.dataset is not None and state_dict["sample_offset"] > len(self.dataset):
            raise ValueError("rollout data-source offset exceeds the current sample count")
        if not isinstance(state_dict.get("metadata"), dict):
            raise TypeError("invalid rollout data-source metadata in checkpoint")
        if hasattr(self, "buffer"):
            if "buffer" not in state_dict:
                raise ValueError("rollout data-source checkpoint has no retry buffer")
            buffer = state_dict["buffer"]
            self._validate_buffer_samples(
                buffer,
                sample_index_limit=state_dict["sample_index"],
                group_index_limit=state_dict["sample_group_index"],
            )

        self.sample_offset = state_dict["sample_offset"]
        self.epoch_id = state_dict["epoch_id"]
        self.sample_group_index = state_dict["sample_group_index"]
        self.sample_index = state_dict["sample_index"]
        self.metadata = state_dict["metadata"]
        if hasattr(self, "buffer"):
            self.buffer = buffer

        if self.args.rollout_global_dataset and self.args.rollout_shuffle and self.dataset is not None:
            self.dataset.shuffle(self.epoch_id)

    def _source_sha256(self) -> str | None:
        if self.dataset is None or self.args.prompt_data is None:
            return None
        source_path, _ = _parse_generalized_path(self.args.prompt_data)
        return _sha256_file(source_path)

    def __len__(self) -> int:
        if self.dataset is None:
            return 0
        return len(self.dataset)


class RolloutDataSourceWithBuffer(RolloutDataSource):
    def __init__(self, args):
        super().__init__(args)
        self.buffer = []
        if self.args.buffer_filter_path is None:
            self.buffer_filter = pop_first
        else:
            self.buffer_filter = load_function(self.args.buffer_filter_path)

    def get_samples(self, num_samples: int) -> list[list[Sample]]:
        """
        Return num_samples samples
        """

        samples = self._get_samples_from_buffer(num_samples)
        num_samples -= len(samples)

        if num_samples == 0:
            return samples

        samples += super().get_samples(num_samples=num_samples)
        return samples

    def _get_samples_from_buffer(self, num_samples: int) -> list[list[Sample]]:
        if len(self.buffer) == 0 or num_samples == 0:
            return []

        samples = self.buffer_filter(self.args, None, self.buffer, num_samples)
        return samples

    def add_samples(self, samples: list[list[Sample]]):
        """
        Add a sample group to buffer.
        """
        self._validate_buffer_samples(samples, allow_missing_indices=True)
        self.buffer.extend(samples)

    def prepend_samples(self, samples: list[list[Sample]]) -> None:
        """Restore unconsumed groups ahead of the remaining retry buffer."""

        self._validate_buffer_samples(samples)
        self.buffer[0:0] = samples

    def _validate_buffer_samples(
        self,
        samples: list[list[Sample]],
        *,
        sample_index_limit: int | None = None,
        group_index_limit: int | None = None,
        allow_missing_indices: bool = False,
        check_against_buffer: bool = True,
    ) -> None:
        if not isinstance(samples, list):
            raise TypeError(f"samples must be a list, got {type(samples)}")
        existing_buffer = self.buffer if check_against_buffer else []
        seen_sample_indices = {
            sample.index
            for group in existing_buffer
            for sample in group
            if sample.index is not None
        }
        seen_group_indices = {
            group[0].group_index
            for group in existing_buffer
            if group and group[0].group_index is not None
        }
        for group in samples:
            if not isinstance(group, list):
                raise TypeError(f"the elements of samples must be list, got {type(group)}")
            if len(group) != self.args.n_samples_per_prompt:
                raise ValueError(f"the length of the elements of samples must be equal to n_samples_per_prompt, got {len(group)} != {self.args.n_samples_per_prompt}")
            for sample in group:
                if not isinstance(sample, Sample):
                    raise TypeError(
                        "rollout data-source buffer groups must contain Sample objects"
                    )
                for name, value, limit in (
                    ("sample", sample.index, sample_index_limit),
                    ("group", sample.group_index, group_index_limit),
                ):
                    if allow_missing_indices and value is None:
                        continue
                    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                        raise ValueError(
                            f"rollout data-source buffer has invalid {name} index: {value!r}"
                        )
                    if limit is not None and value >= limit:
                        raise ValueError(
                            f"rollout data-source buffer {name} index exceeds checkpoint counter"
                        )
                if sample.index is not None and sample.index in seen_sample_indices:
                    raise ValueError("rollout data-source buffer has duplicate sample indices")
                if sample.index is not None:
                    seen_sample_indices.add(sample.index)
            group_indices = {sample.group_index for sample in group}
            if len(group_indices) != 1:
                raise ValueError("rollout data-source buffer group has inconsistent group indices")
            group_index = next(iter(group_indices))
            if group_index is not None and group_index in seen_group_indices:
                raise ValueError("rollout data-source buffer has duplicate group indices")
            if group_index is not None:
                seen_group_indices.add(group_index)

    # TODO remove
    def update_metadata(self, metadata: dict):
        self.metadata.update(metadata)

    # TODO remove
    def get_metadata(self):
        return self.metadata

    def get_buffer_length(self):
        return len(self.buffer)


def pop_first(args, rollout_id, buffer: list[list[Sample]], num_samples: int) -> list[list[Sample]]:
    num_to_pop = min(len(buffer), num_samples)
    samples = buffer[:num_to_pop]
    del buffer[:num_to_pop]
    return samples
