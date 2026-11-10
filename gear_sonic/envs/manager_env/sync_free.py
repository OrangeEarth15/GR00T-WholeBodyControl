"""Sync-free ``env.step`` for SONIC on Isaac Lab 3.0 + Newton (``manager_env.config.sync_free_step``).

``ManagerBasedRLEnv.step`` resets environments through ``reset_buf.nonzero()``. That host sync waits for the
physics of the step, so the reset, command and observation work after it (about 40 ms of kernel launches per
control step) only starts once the GPU is idle. Here every reset-path update covers all environments under the
reset mask, with Isaac Lab 3.0's ``*_mask`` writes, so ``env.step`` has no host sync left and the CPU queues the
post-processing while the GPU runs the physics. Use it with ``newton.graph_decimation=True``: without it,
``sim.step`` blocks the CPU on every physics step anyway.

``enable`` checks that the task only uses what is implemented here, which is what sonic_release uses: no
curriculum, recorder or reset-mode event terms, no stateful observation, reward or termination terms,
``push_by_setting_velocity`` as the only interval event, joint actions, and the tracking command with a time_out
termination.
"""

from __future__ import annotations

import types

from isaaclab.envs.mdp import push_by_setting_velocity
from isaaclab.envs.mdp.actions.joint_actions import JointAction
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import math as math_utils
import torch

from gear_sonic.envs.manager_env.mdp.commands import TrackingCommand
from gear_sonic.envs.manager_env.mdp.terminations import tracking_time_out


def enable(env):
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
    env.step = types.MethodType(step, env)


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

    self.episode_length_buf += 1
    self.common_step_counter += 1
    self.reset_buf = self.termination_manager.compute()
    self.reset_terminated = self.termination_manager.terminated
    self.reset_time_outs = self.termination_manager.time_outs
    self.reward_buf = self.reward_manager.compute(dt=self.step_dt)

    reset(self, self.reset_buf)
    for term in self.command_manager._terms.values():  # noqa: SLF001
        term.compute_masked(self.step_dt)
    apply_interval_events(self, self.step_dt)
    self.obs_buf = self.observation_manager.compute(update_history=True)
    return self.obs_buf, self.reward_buf, self.reset_terminated, self.reset_time_outs, self.extras


def reset(env, mask: torch.Tensor):
    """``ManagerBasedRLEnv._reset_idx`` for the envs where ``mask`` is set, in the same order."""
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
    log = {}
    rewards = env.reward_manager
    episode_sums = rewards._episode_sum_buf  # noqa: SLF001
    episodic_sum_avg = (episode_sums * mask[:, None]).sum(dim=0) / count.clamp(min=1)
    episodic_sum_avg = episodic_sum_avg / env.max_episode_length_s
    for i, key in enumerate(rewards._term_names):  # noqa: SLF001
        log["Episode_Reward/" + key] = episodic_sum_avg[i]
    episode_sums.masked_fill_(mask[:, None], 0.0)

    for name, term in env.command_manager._terms.items():  # noqa: SLF001
        for metric_name, value in term.reset_masked(mask).items():
            log[f"Metrics/{name}/{metric_name}"] = value

    _reset_interval_timers(env.event_manager, mask)

    terminations = env.termination_manager
    done_stats = terminations._last_episode_dones.float().mean(dim=0)  # noqa: SLF001
    for i, key in enumerate(terminations._term_names):  # noqa: SLF001
        log["Episode_Termination/" + key] = done_stats[i]

    env.episode_length_buf.masked_fill_(mask, 0)
    # Isaac Lab only rewrites extras["log"] in steps that reset an env; keep the last values otherwise
    previous = env.extras.get("log", {})
    any_reset = count > 0
    env.extras["log"] = {
        key: torch.where(any_reset, value, previous[key]) if key in previous else value
        for key, value in log.items()
    }


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
