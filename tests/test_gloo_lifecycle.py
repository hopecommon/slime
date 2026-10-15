"""Exercise the canonical CPU process-group owner with real Gloo resources."""
import pytest
import torch.distributed as dist

from slime.utils import distributed_utils


@pytest.mark.parametrize("world_alias", [False, True])
def test_terminal_gloo_cleanup_clears_subgroup_and_temporary_world(tmp_path, monkeypatch, world_alias):
    dist.init_process_group(backend="gloo", init_method=(tmp_path / "rendezvous").as_uri(), rank=0, world_size=1)
    try:
        if world_alias:
            distributed_utils.set_gloo_group(dist.group.WORLD)
        else:
            raw_new_group = dist.new_group
            monkeypatch.setattr(dist, "old_new_group", raw_new_group, raising=False)
            def replaced_new_group(**kwargs):
                raise AssertionError("canonical Gloo group entered model reload registry")
            monkeypatch.setattr(dist, "new_group", replaced_new_group)
            distributed_utils.init_gloo_group()
        dist.barrier(group=distributed_utils.get_gloo_group())
        distributed_utils.destroy_gloo_group()
        distributed_utils.destroy_gloo_group()
        with pytest.raises(RuntimeError, match="not been initialized"):
            distributed_utils.get_gloo_group()
        assert dist.is_initialized() is not world_alias
    finally:
        distributed_utils.set_gloo_group(None)
        if dist.is_initialized():
            dist.destroy_process_group()
