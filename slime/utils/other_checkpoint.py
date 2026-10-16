"""The shared-buffer other-checkpoint operation, independent of native imports."""

from contextlib import contextmanager


@contextmanager
def temporary_args(args, updates):
    """Restore values and attribute absence even when native loading fails."""
    missing = object()
    previous = {name: getattr(args, name, missing) for name in updates}
    try:
        for name, value in updates.items():
            setattr(args, name, value)
        yield
    finally:
        for name, value in previous.items():
            if value is missing:
                delattr(args, name)
            else:
                setattr(args, name, value)


def load_other_checkpoint(*, args, model, model_tag, path, load, backup, overrides=None):
    """Load shared model buffers and publish a named backup only on success.

    The caller owns tensor rollback: a native loader can mutate part of the
    model before raising. The operation always restores its temporary arguments.
    """
    settings = {"load": path, "no_load_optim": True, "no_load_rng": True, "finetune": True}
    settings.update(overrides or {})
    with temporary_args(args, settings):
        load(model, None, None, checkpointing_context={}, skip_load_to_model_and_opt=False)
    backup(model_tag)
