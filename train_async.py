import logging

import ray

from slime.ray.placement_group import create_placement_groups, create_rollout_manager, create_training_models
from slime.utils.arguments import parse_args
from slime.utils.logging_utils import configure_logger, finish_tracking, init_tracking
from slime.utils.misc import should_run_periodic_action

logger = logging.getLogger(__name__)
ROLLOUT_MANAGER_DISPOSE_TIMEOUT_SECONDS = 600.0


def _dispose_rollout_manager(rollout_manager):
    try:
        return ray.get(
            rollout_manager.dispose.remote(),
            timeout=ROLLOUT_MANAGER_DISPOSE_TIMEOUT_SECONDS,
        )
    except BaseException:
        try:
            ray.kill(rollout_manager, no_restart=True)
        except BaseException:
            logger.exception("failed to kill rollout manager after dispose failure")
        raise


def _shutdown_training_models(actor_model, critic_model):
    errors = []
    for model in (critic_model, actor_model):
        if model is None:
            continue
        try:
            model.shutdown()
        except BaseException as exc:
            errors.append(exc)
    if errors:
        raise RuntimeError(
            "; ".join(f"{type(error).__name__}: {error}" for error in errors)
        ) from errors[0]


def train(args):
    assert not args.colocate, "Colocation is not supported for async training."
    configure_logger()
    rollout_manager = None
    actor_model = None
    critic_model = None
    rollout_data_next_future = None
    tracking_initialized = False
    active_error = None

    try:
        pgs = create_placement_groups(args)
        init_tracking(args)
        tracking_initialized = True

        rollout_manager, num_rollout_per_epoch = create_rollout_manager(args, pgs["rollout"])
        actor_model, critic_model = create_training_models(args, pgs, rollout_manager)

        actor_model.update_weights()
        if args.check_weight_update_equal:
            ray.get(rollout_manager.check_weights.remote(action="compare"))

        if args.start_rollout_id > args.num_rollout:
            raise ValueError("start_rollout_id cannot exceed num_rollout")
        if args.num_rollout == 0 and args.eval_interval is not None:
            ray.get(rollout_manager.eval.remote(rollout_id=0))
        if args.start_rollout_id < args.num_rollout:
            rollout_data_next_future = rollout_manager.generate.remote(args.start_rollout_id)
        for rollout_id in range(args.start_rollout_id, args.num_rollout):
            if rollout_data_next_future is None:
                raise RuntimeError("missing rollout future for async training step")
            rollout_data_curr_ref = ray.get(rollout_data_next_future)
            rollout_data_next_future = None

            save_due = should_run_periodic_action(
                rollout_id,
                args.save_interval,
                num_rollout_per_epoch,
                args.num_rollout,
            )
            update_due = (rollout_id + 1) % args.update_weights_interval == 0
            eval_due = should_run_periodic_action(
                rollout_id,
                args.eval_interval,
                num_rollout_per_epoch,
            )
            boundary_due = save_due or update_due or eval_due

            if rollout_id + 1 < args.num_rollout and not boundary_due:
                rollout_data_next_future = rollout_manager.generate.remote(rollout_id + 1)

            if args.use_critic:
                actor_trains_this_step = rollout_id >= args.num_critic_only_steps
                value_refs = critic_model.async_train(rollout_id, rollout_data_curr_ref)
                if actor_trains_this_step:
                    ray.get(actor_model.async_train(rollout_id, rollout_data_curr_ref, external_data=value_refs))
                else:
                    ray.get(value_refs)
            else:
                ray.get(actor_model.async_train(rollout_id, rollout_data_curr_ref))

            if boundary_due:
                ray.get(rollout_manager.quiesce.remote())

            if save_due:
                if (not args.use_critic) or rollout_id >= args.num_critic_only_steps:
                    actor_model.save_model(
                        rollout_id,
                        force_sync=rollout_id == args.num_rollout - 1,
                    )
                if args.use_critic:
                    critic_model.save_model(
                        rollout_id,
                        force_sync=rollout_id == args.num_rollout - 1,
                    )
                if args.rollout_global_dataset:
                    ray.get(rollout_manager.save.remote(rollout_id))

            if update_due:
                actor_model.update_weights()

            if eval_due:
                ray.get(rollout_manager.eval.remote(rollout_id))

            if rollout_id + 1 < args.num_rollout and rollout_data_next_future is None:
                rollout_data_next_future = rollout_manager.generate.remote(rollout_id + 1)
    except BaseException as exc:
        active_error = exc
        raise
    finally:
        cleanup_errors = []
        if rollout_data_next_future is not None:
            try:
                ray.cancel(rollout_data_next_future)
            except BaseException as exc:
                cleanup_errors.append(exc)
                logger.exception("failed to cancel prefetched rollout")
        if rollout_manager is not None:
            try:
                _dispose_rollout_manager(rollout_manager)
            except BaseException as exc:
                cleanup_errors.append(exc)
                logger.exception("failed to dispose rollout manager")
        try:
            _shutdown_training_models(actor_model, critic_model)
        except BaseException as exc:
            cleanup_errors.append(exc)
            logger.exception("failed to shut down training models")
        if tracking_initialized:
            try:
                finish_tracking(args)
            except BaseException as exc:
                cleanup_errors.append(exc)
                logger.exception("failed to finish training tracking")
        if cleanup_errors and active_error is None:
            if len(cleanup_errors) == 1:
                raise cleanup_errors[0]
            raise RuntimeError(
                "; ".join(f"{type(error).__name__}: {error}" for error in cleanup_errors)
            ) from cleanup_errors[0]


if __name__ == "__main__":
    args = parse_args()
    train(args)
