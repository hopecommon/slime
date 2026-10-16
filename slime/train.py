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


def train(args, *, actor_cls=None):
    configure_logger()
    release_train = args.release_train
    rollout_manager = None
    actor_model = None
    critic_model = None
    tracking_initialized = False
    active_error = None

    try:
        pgs = create_placement_groups(args)
        init_tracking(args)
        tracking_initialized = True

        rollout_manager, num_rollout_per_epoch = create_rollout_manager(args, pgs["rollout"])
        factory_options = {} if actor_cls is None else {"actor_cls": actor_cls}
        actor_model, critic_model = create_training_models(args, pgs, rollout_manager, **factory_options)

        if args.offload_rollout and not release_train:
            ray.get(rollout_manager.onload_weights.remote())
        actor_model.update_weights()
        if args.check_weight_update_equal:
            ray.get(rollout_manager.check_weights.remote(action="compare"))
        if args.offload_rollout:
            ray.get(rollout_manager.onload_kv.remote())

        if args.num_rollout == 0 and args.eval_interval is not None:
            ray.get(rollout_manager.eval.remote(rollout_id=0))

        def offload_train(actor_trains_this_step):
            if not args.offload_train:
                if not args.use_critic or actor_trains_this_step:
                    actor_model.clear_memory()
                else:
                    critic_model.clear_memory()

        def save(rollout_id):
            actor_trains_this_step = (not args.use_critic) or rollout_id >= args.num_critic_only_steps
            if actor_trains_this_step:
                actor_model.save_model(
                    rollout_id,
                    force_sync=release_train or rollout_id == args.num_rollout - 1,
                )
            if args.use_critic:
                critic_model.save_model(
                    rollout_id,
                    force_sync=release_train or rollout_id == args.num_rollout - 1,
                )
            if args.rollout_global_dataset:
                ray.get(rollout_manager.save.remote(rollout_id))

        for rollout_id in range(args.start_rollout_id, args.num_rollout):
            if args.eval_interval is not None and rollout_id == 0 and not args.skip_eval_before_train:
                ray.get(rollout_manager.eval.remote(rollout_id))

            rollout_data_ref = ray.get(rollout_manager.generate.remote(rollout_id))

            if args.offload_rollout:
                ray.get(rollout_manager.quiesce.remote())
                ray.get(rollout_manager.offload.remote())

            if release_train:
                actor_model.create()

            actor_trains_this_step = (not args.use_critic) or rollout_id >= args.num_critic_only_steps
            if args.use_critic:
                value_refs = critic_model.async_train(rollout_id, rollout_data_ref)
                if actor_trains_this_step:
                    ray.get(actor_model.async_train(rollout_id, rollout_data_ref, external_data=value_refs))
                else:
                    ray.get(value_refs)
            else:
                ray.get(actor_model.async_train(rollout_id, rollout_data_ref))

            ray.get(rollout_manager.quiesce.remote())
            if release_train or should_run_periodic_action(
                rollout_id,
                args.save_interval,
                num_rollout_per_epoch,
                args.num_rollout,
            ):
                save(rollout_id)

            offload_train(actor_trains_this_step)
            if args.offload_rollout and not release_train:
                ray.get(rollout_manager.onload_weights.remote())
            actor_model.update_weights()

            if args.offload_rollout:
                ray.get(rollout_manager.onload_kv.remote())

            if should_run_periodic_action(
                rollout_id,
                args.eval_interval,
                num_rollout_per_epoch,
            ):
                ray.get(rollout_manager.eval.remote(rollout_id))
    except BaseException as exc:
        active_error = exc
        raise
    finally:
        cleanup_errors = []
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


def main():
    train(parse_args())


if __name__ == "__main__":
    main()
