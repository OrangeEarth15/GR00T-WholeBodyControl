"""Sync-free ``env.step`` for SONIC on Isaac Lab 3.0 + Newton (``manager_env.config.sync_free_step``).

``ManagerBasedRLEnv.step`` resets environments through ``reset_buf.nonzero()``. That host sync waits for the
physics of the step, so the reset, command and observation work after it (about 40 ms of kernel launches per
control step) only starts once the GPU is idle. Here every reset-path update covers all environments under the
reset mask, with Isaac Lab 3.0's ``*_mask`` writes, so ``env.step`` has no host sync left and the CPU queues the
post-processing while the GPU runs the physics. Export ``CUDA_SCALE_LAUNCH_QUEUES=4x`` before CUDA starts: the
default launch queues fill up with the step's ~1300 small kernels and stall the CPU while the physics runs.
``newton.graph_decimation=True`` adds about 0.5% on top (one H100 PCIe, 1 substep, 2 mm gap, 4096 envs: training
iterations 3.37 s -> 2.84 s with all three).

``enable`` checks that the task only uses what is implemented here, which is what sonic_release uses: no
curriculum, recorder or reset-mode event terms, no stateful observation, reward or termination terms,
``push_by_setting_velocity`` as the only interval event, joint actions, and the tracking command with a time_out
termination.

With ``graph=True`` (``manager_env.config.post_processing_graph``) the post-processing is recorded once as a
CUDA graph and replayed in later steps (see ``PostProcessingGraph``), so the CPU launches it with one call.
"""

from __future__ import annotations

import gc
import os
import types

from isaaclab.envs.mdp import push_by_setting_velocity
from isaaclab.envs.mdp.actions.joint_actions import JointAction
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import math as math_utils
from loguru import logger
import torch
import warp as wp

from gear_sonic.envs.manager_env.mdp.commands import TrackingCommand
from gear_sonic.envs.manager_env.mdp.terminations import tracking_time_out


def enable(env, graph: bool = False):
    """Replace ``env.step`` with the sync-free step after checking that the task is supported."""
    unsupported = []
    if (
        env.curriculum_manager.active_terms
        or env.recorder_manager.active_terms
        or env.video_recorders
    ):
        unsupported.append("curriculum, recorder or video terms")
    events = env.event_manager
    if set(events._mode_term_cfgs) - {"prestartup", "startup", "interval"}:  # noqa: SLF001
        unsupported.append(f"event modes {sorted(events._mode_term_cfgs)}")  # noqa: SLF001
    for term_cfg in events._mode_term_cfgs.get("interval", []):  # noqa: SLF001
        if term_cfg.func is not push_by_setting_velocity or term_cfg.is_global_time:
            unsupported.append(f"interval event {term_cfg.func}")
    obs = env.observation_manager
    if (
        any(obs._group_obs_class_term_cfgs.values())  # noqa: SLF001
        or any(obs._group_obs_term_delay_buffer.values())  # noqa: SLF001
        or obs._group_obs_class_instances  # noqa: SLF001
    ):
        unsupported.append("stateful observation terms, delays, noise models or modifiers")
    if (
        env.reward_manager._class_term_cfgs or env.termination_manager._class_term_cfgs
    ):  # noqa: SLF001
        unsupported.append("class-based reward or termination terms")
    if not all(
        isinstance(term, JointAction) for term in env.action_manager._terms.values()
    ):  # noqa: SLF001
        unsupported.append("non-joint action terms")
    commands = env.command_manager._terms.values()  # noqa: SLF001
    if not all(isinstance(term, TrackingCommand) for term in commands):
        unsupported.append("command terms other than TrackingCommand")
    termination_funcs = [
        env.termination_manager.get_term_cfg(name).func
        for name in env.termination_manager.active_terms
    ]
    if tracking_time_out not in termination_funcs:
        unsupported.append("tracking without a time_out termination")
    if unsupported:
        raise NotImplementedError(f"sync_free_step does not support: {'; '.join(unsupported)}")
    for term in commands:
        term.check_masked_reset_support()
    if not os.environ.get("CUDA_SCALE_LAUNCH_QUEUES"):
        logger.warning(
            "sync_free_step: CUDA_SCALE_LAUNCH_QUEUES is not set; the default launch queues fill up "
            "while the physics runs and stall the CPU. Export CUDA_SCALE_LAUNCH_QUEUES=4x before starting."
        )
    env.step = types.MethodType(step, env)
    env._sync_free_log = None  # noqa: SLF001
    env._sync_free_graph = PostProcessingGraph(env) if graph else None  # noqa: SLF001


def step(self, action: torch.Tensor):
    """``ManagerBasedRLEnv.step`` with the reset path replaced by ``reset`` (no rendering or recording)."""
    self.action_manager.process_action(action.to(self.device))
    # physics-owned decimation covers all substeps in one call
    steps_per_call = self.cfg.decimation if self._physics_handles_decimation else 1
    for _ in range(self.cfg.decimation // steps_per_call):
        self._sim_step_counter += steps_per_call
        self.action_manager.apply_action()
        self.scene.write_data_to_sim()
        self.sim.step(render=False)
        self.scene.update(dt=self.physics_dt * steps_per_call)
    self.common_step_counter += 1

    graph = self._sync_free_graph
    obs, reward, terminated, time_outs, log = post_process(self) if graph is None else graph.run()
    if graph is not None:
        # replays rewrite these buffers in place
        obs, reward, terminated, time_outs = _clone((obs, reward, terminated, time_outs))
    # the trainer keeps every step's log
    self.extras["log"] = dict(zip(self._sync_free_log_keys, log.clone().unbind()))
    return obs, reward, terminated, time_outs, self.extras


def post_process(self):
    """The part of ``step`` after the physics, up to the observations."""
    self.episode_length_buf += 1
    self.reset_buf = self.termination_manager.compute()
    self.reset_terminated = self.termination_manager.terminated
    self.reset_time_outs = self.termination_manager.time_outs
    self.reward_buf = self.reward_manager.compute(dt=self.step_dt)

    log = reset(self, self.reset_buf)
    for term in self.command_manager._terms.values():  # noqa: SLF001
        term.compute_masked(self.step_dt)
    apply_interval_events(self, self.step_dt)
    self.obs_buf = self.observation_manager.compute(update_history=True)
    return self.obs_buf, self.reward_buf, self.reset_terminated, self.reset_time_outs, log


def reset(env, mask: torch.Tensor) -> torch.Tensor:
    """``ManagerBasedRLEnv._reset_idx`` for the envs where ``mask`` is set, in the same order.

    Returns the episode log as one tensor (names in ``env._sync_free_log_keys``).
    """
    for articulation in env.scene.articulations.values():
        articulation.reset(env_mask=mask)
    for sensor in env.scene.sensors.values():
        sensor.reset(env_mask=mask)

    for buffers in env.observation_manager._group_obs_term_history_buffer.values():  # noqa: SLF001
        for buffer in buffers.values():
            _reset_history(buffer, mask)

    actions = env.action_manager
    actions._prev_action.masked_fill_(mask[:, None], 0.0)  # noqa: SLF001
    actions._action.masked_fill_(mask[:, None], 0.0)  # noqa: SLF001
    for term in actions._terms.values():  # noqa: SLF001
        term._raw_actions.masked_fill_(mask[:, None], 0.0)  # noqa: SLF001

    count = mask.sum()
    names, values = [], []
    rewards = env.reward_manager
    episode_sums = rewards._episode_sum_buf  # noqa: SLF001
    episodic_sum_avg = (episode_sums * mask[:, None]).sum(dim=0) / count.clamp(min=1)
    names += ["Episode_Reward/" + key for key in rewards._term_names]  # noqa: SLF001
    values.append(episodic_sum_avg / env.max_episode_length_s)
    episode_sums.masked_fill_(mask[:, None], 0.0)

    for name, term in env.command_manager._terms.items():  # noqa: SLF001
        for metric_name, value in term.reset_masked(mask).items():
            names.append(f"Metrics/{name}/{metric_name}")
            values.append(value)

    _reset_interval_timers(env.event_manager, mask)

    terminations = env.termination_manager
    names += ["Episode_Termination/" + key for key in terminations._term_names]  # noqa: SLF001
    values.append(terminations._last_episode_dones.float().mean(dim=0))  # noqa: SLF001

    env.episode_length_buf.masked_fill_(mask, 0)
    # Isaac Lab only rewrites extras["log"] in steps that reset an env; keep the last values otherwise
    log = torch.cat([value.reshape(-1).float() for value in values])
    if env._sync_free_log is None or env._sync_free_log_keys != names:  # noqa: SLF001
        # first steps: the command metrics only exist after the first command update
        env._sync_free_log, env._sync_free_log_keys = log, names  # noqa: SLF001
    else:
        torch.where(count > 0, log, env._sync_free_log, out=env._sync_free_log)  # noqa: SLF001
    return env._sync_free_log  # noqa: SLF001


def _reset_history(buffer, mask: torch.Tensor):
    """``CircularBuffer.reset`` for the batch entries in ``mask``; ``append`` backfills them without syncing."""
    buffer._num_pushes.masked_fill_(mask, 0)  # noqa: SLF001
    buffer._need_reset = True  # noqa: SLF001
    if buffer._buffer is not None:  # noqa: SLF001
        batch_dim = 1 if buffer._stack_dim_internal is None else 0  # noqa: SLF001
        shape = [1] * buffer._buffer.dim()  # noqa: SLF001
        shape[batch_dim] = -1
        buffer._buffer.masked_fill_(mask.view(shape), 0.0)  # noqa: SLF001


def _reset_interval_timers(events, mask: torch.Tensor):
    """The interval-timer part of ``EventManager.reset`` for the envs in ``mask``."""
    for index, term_cfg in enumerate(events._mode_term_cfgs.get("interval", [])):  # noqa: SLF001
        if term_cfg.resample_interval_on_reset:
            time_left = events._interval_term_time_left[index]  # noqa: SLF001
            lower, upper = term_cfg.interval_range_s
            sampled = torch.rand_like(time_left) * (upper - lower) + lower
            torch.where(mask, sampled, time_left, out=time_left)


def apply_interval_events(env, dt: float):
    """``EventManager.apply(mode="interval")`` for per-env ``push_by_setting_velocity`` terms."""
    events = env.event_manager
    for index, term_cfg in enumerate(events._mode_term_cfgs.get("interval", [])):  # noqa: SLF001
        time_left = events._interval_term_time_left[index]  # noqa: SLF001
        time_left -= dt
        due = time_left < 1e-6
        lower, upper = term_cfg.interval_range_s
        sampled = torch.rand_like(time_left) * (upper - lower) + lower
        torch.where(due, sampled, time_left, out=time_left)
        _push_by_setting_velocity(env, due, **term_cfg.params)


def _push_by_setting_velocity(
    env,
    mask: torch.Tensor,
    velocity_range: dict[str, tuple[float, float]],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """``isaaclab.envs.mdp.push_by_setting_velocity`` for the envs in ``mask``."""
    asset = env.scene[asset_cfg.name]
    vel_w = asset.data.root_vel_w.torch.clone()
    vel_w += math_utils.sample_uniform_from_ranges(
        velocity_range,
        ("x", "y", "z", "roll", "pitch", "yaw"),
        vel_w.shape[:-1],
        device=asset.device,
    )
    asset.write_root_velocity_to_sim_mask(root_velocity=vel_w, env_mask=mask)


class PostProcessingGraph:
    """``post_process`` recorded once as a CUDA graph and replayed in later steps.

    A replay reruns the recorded kernels on the recorded buffers without running the Python code. What keeps it
    equal to calling ``post_process``:

    - Warp work (Isaac Lab's ``*_mask`` writes and lazy data, Newton's FK) is recorded on the same stream.
    - Tensor attributes that the step reassigns (metrics, relative body frames, ``reset_buf``, ...) are copied
      back into the buffers they held before the capture, at the end of the graph, where the next replay reads
      them.
    - Before each replay, attributes that code outside the step reassigned (adaptive sampling statistics, the
      adaptive cursor snapshot, a reloaded motion batch of the same size) are copied into those buffers as well.
      A new shape or dtype, or other Python flags on the command terms and motion libraries (evaluation mode,
      no adaptive snapshot), records another graph.
    """

    def __init__(self, env, warmup_steps: int = 2):
        self.env = env
        # eager steps first: they allocate buffers and load kernels
        self.warmup_steps = warmup_steps
        self.graphs = {}  # flags -> (graph, outputs, pinned slots)

    def run(self):
        if self.warmup_steps > 0:
            self.warmup_steps -= 1
            return post_process(self.env)
        flags = _flags(self.env)
        entry = self.graphs.get(flags)
        if entry is None or not _pin(entry[2]):
            entry = self.graphs[flags] = self._capture()
        entry[0].replay()
        return entry[1]

    def _capture(self):
        objects = _tracked_objects(self.env)
        before = {(id(obj), attr, key): tensor for obj, attr, key, tensor in _tensor_slots(objects)}
        graph = torch.cuda.CUDAGraph()
        stream = torch.cuda.Stream(device=self.env.device)
        gc.disable()  # a Warp array collected mid-capture would be freed on the capture stream
        try:
            with torch.cuda.graph(graph, stream=stream, capture_error_mode="thread_local"):
                with wp.ScopedStream(wp.stream_from_torch(stream), sync_enter=False):
                    outputs = post_process(self.env)
                pinned, copied = [], 0
                for obj, attr, key, tensor in _tensor_slots(objects):
                    old = before.get((id(obj), attr, key))
                    if old is not None and old is not tensor:
                        if old.shape != tensor.shape or old.dtype != tensor.dtype:
                            raise RuntimeError(
                                f"post_process changes the shape of {type(obj).__name__}.{attr}"
                            )
                        if old.data_ptr() != tensor.data_ptr():
                            old.copy_(tensor)
                            copied += 1
                        _put(obj, attr, key, old)
                        tensor = old
                    pinned.append((obj, attr, key, tensor))
        finally:
            gc.enable()
        logger.info(
            f"sync_free: recorded the post-processing graph ({copied} reassigned buffers copied back)"
        )
        return graph, outputs, pinned


def _clone(value):
    if isinstance(value, dict):
        return {name: _clone(item) for name, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_clone(item) for item in value)
    return value.clone()


_ATTR = object()  # slot key of a plain attribute (dict keys can be None)


def _tracked_objects(env) -> list:
    objects = [
        env,
        env.observation_manager,
        env.action_manager,
        env.reward_manager,
        env.termination_manager,
        env.event_manager,
        env.command_manager,
    ]
    objects += env.action_manager._terms.values()  # noqa: SLF001
    for term in env.command_manager._terms.values():  # noqa: SLF001
        objects += [term, term.motion_lib]
    for buffers in env.observation_manager._group_obs_term_history_buffer.values():  # noqa: SLF001
        objects += buffers.values()
    return objects


def _tensor_slots(objects) -> list:
    """(object, attribute, key, tensor) for tensor attributes, including tensors in dict and list attributes."""
    slots = []
    for obj in objects:
        for attr, value in vars(obj).items():
            if isinstance(value, torch.Tensor):
                slots.append((obj, attr, _ATTR, value))
            elif isinstance(value, dict | list):
                items = value.items() if isinstance(value, dict) else enumerate(value)
                slots += [(obj, attr, k, v) for k, v in items if isinstance(v, torch.Tensor)]
    return slots


def _put(obj, attr, key, tensor):
    if key is _ATTR:
        vars(obj)[attr] = tensor
    else:
        vars(obj)[attr][key] = tensor


def _pin(slots) -> bool:
    """Copy reassigned attributes into the buffers a graph reads; False if one changed shape or is gone."""
    for obj, attr, key, tensor in slots:
        try:
            current = vars(obj)[attr] if key is _ATTR else vars(obj)[attr][key]
        except (KeyError, IndexError):
            return False
        if current is tensor:
            continue
        if (
            not isinstance(current, torch.Tensor)
            or current.shape != tensor.shape
            or current.dtype != tensor.dtype
        ):
            return False
        tensor.copy_(current)
        _put(obj, attr, key, tensor)
    return True


def _flags(env) -> tuple:
    """Python flags of the command terms and motion libraries, which select the recorded branches."""
    flags = []
    for term in env.command_manager._terms.values():  # noqa: SLF001
        for obj in (term, term.motion_lib):
            flags += [
                (name, value)
                for name, value in vars(obj).items()
                if isinstance(value, bool | int | float | str | None)
            ]
    return tuple(flags)
