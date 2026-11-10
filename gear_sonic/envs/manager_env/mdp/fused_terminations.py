"""Fused Warp evaluation of the tracking termination terms (Direct-Warp).

sonic_release ends episodes on five tracking terms: anchor height, anchor orientation, end-effector height,
foot position and the end of the clip. Evaluated one by one in torch they launch about 60 kernels per
control step. FusedTerminationManager evaluates them in one Warp kernel, one thread per env, reading the
reference motion at the current frame from the motion library and the robot state from Isaac Lab's Warp
buffers, and keeps the bookkeeping of isaaclab.managers.TerminationManager.compute. If a configured term or
parameter is not supported it keeps the stock per-term path.

Enable with ++manager_env.config.fused_terminations=True. With ++manager_env.config.fused_terminations_check=True
the stock terms are also evaluated every step and mismatches are counted (printed every 24 steps).
"""

from __future__ import annotations

from isaaclab.managers import TerminationManager
import torch
import warp as wp

from gear_sonic.envs.manager_env import mdp
from gear_sonic.envs.manager_env.mdp.commands import _get_body_indexes
from gear_sonic.envs.manager_env.mdp.fused_rewards import _int_array, _quat_error, _wxyz

# Term kinds, in the order the kernel evaluates them.
ANCHOR_HEIGHT, ANCHOR_ORI, BODY_HEIGHT, BODY_POS, TIME_OUT = range(5)
NUM_KINDS = 5

_KIND_OF_FUNC = {
    mdp.exceeded_anchor_height: ANCHOR_HEIGHT,
    mdp.exceeded_anchor_ori: ANCHOR_ORI,
    mdp.exceeded_body_height: BODY_HEIGHT,
    mdp.exceeded_body_pos: BODY_POS,
    mdp.tracking_time_out: TIME_OUT,
}


@wp.func
def _height_threshold(
    params: wp.array2d(dtype=float), kind: int, running_ref_root_height: float
) -> float:
    """exceeded_*_height: the down threshold while the reference root is low, when adaptive."""
    if params[kind, 1] > 0.0 and running_ref_root_height < params[kind, 3]:
        return params[kind, 2]
    return params[kind, 0]


@wp.kernel
def _tracking_terminations_kernel(
    # command state and the motion library (reference quaternions (w, x, y, z))
    motion_ids: wp.array(dtype=wp.int64),
    starts: wp.array(dtype=wp.int64),
    time_steps: wp.array(dtype=wp.int64),
    length_starts: wp.array(dtype=wp.int64),
    num_frames: wp.array(dtype=wp.int64),
    ref_body_pos: wp.array3d(dtype=float),
    ref_body_quat: wp.array3d(dtype=float),
    env_origins: wp.array2d(dtype=float),
    body_pos_relative: wp.array3d(dtype=float),
    running_ref_root_height: wp.array(dtype=float),
    # robot state, Isaac Lab Warp buffers
    body_pos: wp.array2d(dtype=wp.vec3f),
    body_quat: wp.array2d(dtype=wp.quatf),
    robot_anchor: int,
    motion_anchor: int,
    tracked_bodies: wp.array(dtype=int),
    height_bodies: wp.array(dtype=int),
    pos_bodies: wp.array(dtype=int),
    columns: wp.array(dtype=int),
    time_out: wp.array(dtype=wp.bool),
    # per kind: threshold, adaptive flag, down threshold, root height threshold
    params: wp.array2d(dtype=float),
    term_dones: wp.array2d(dtype=wp.bool),
    last_episode_dones: wp.array2d(dtype=wp.bool),
    truncated: wp.array(dtype=wp.bool),
    terminated: wp.array(dtype=wp.bool),
):
    e = wp.tid()
    frame = length_starts[motion_ids[e]] + starts[e] + time_steps[e]
    root_height = running_ref_root_height[e]

    col = columns[ANCHOR_HEIGHT]
    if col >= 0:
        ref_z = ref_body_pos[frame, motion_anchor, 2] + env_origins[e, 2]
        diff = wp.abs(ref_z - body_pos[e, robot_anchor][2])
        term_dones[e, col] = diff > _height_threshold(params, ANCHOR_HEIGHT, root_height)

    col = columns[ANCHOR_ORI]
    if col >= 0:
        ref_q = wp.vec4(
            ref_body_quat[frame, motion_anchor, 0],
            ref_body_quat[frame, motion_anchor, 1],
            ref_body_quat[frame, motion_anchor, 2],
            ref_body_quat[frame, motion_anchor, 3],
        )
        err = _quat_error(ref_q, _wxyz(body_quat[e, robot_anchor]))
        term_dones[e, col] = err * err > params[ANCHOR_ORI, 0]

    col = columns[BODY_HEIGHT]
    if col >= 0:
        thresh = _height_threshold(params, BODY_HEIGHT, root_height)
        hit = bool(False)
        for k in range(height_bodies.shape[0]):
            b = height_bodies[k]
            err = wp.abs(body_pos_relative[e, b, 2] - body_pos[e, tracked_bodies[b]][2])
            hit = hit or err > thresh
        term_dones[e, col] = hit

    col = columns[BODY_POS]
    if col >= 0:
        hit = bool(False)
        for k in range(pos_bodies.shape[0]):
            b = pos_bodies[k]
            p = body_pos[e, tracked_bodies[b]]
            d = wp.vec3(
                body_pos_relative[e, b, 0] - p[0],
                body_pos_relative[e, b, 1] - p[1],
                body_pos_relative[e, b, 2] - p[2],
            )
            hit = hit or wp.length(d) > params[BODY_POS, 0]
        term_dones[e, col] = hit

    col = columns[TIME_OUT]
    if col >= 0:
        term_dones[e, col] = time_steps[e] + starts[e] + wp.int64(1) >= num_frames[motion_ids[e]]

    # TerminationManager.compute bookkeeping
    any_truncated = bool(False)
    any_terminated = bool(False)
    for i in range(term_dones.shape[1]):
        if term_dones[e, i]:
            if time_out[i]:
                any_truncated = True
            else:
                any_terminated = True
    truncated[e] = any_truncated
    terminated[e] = any_terminated
    if any_truncated or any_terminated:
        for i in range(term_dones.shape[1]):
            last_episode_dones[e, i] = term_dones[e, i]


class FusedTerminationManager(TerminationManager):
    """TerminationManager whose compute() evaluates the tracking terms in one Warp kernel."""

    def __init__(self, cfg: object, env, check: bool = False):
        super().__init__(cfg, env)
        self._check = check
        self._check_steps = 0
        self._check_mismatches = torch.zeros(
            len(self._term_names), device=self.device, dtype=torch.long
        )
        self._unsupported = self._prepare_fused()
        if self._unsupported:
            print(
                f"[FusedTerminationManager] stock per-term path: {self._unsupported}"
            )  # noqa: T201
        else:
            print(
                f"[FusedTerminationManager] {len(self._term_names)} terms in one Warp kernel"
            )  # noqa: T201

    def _prepare_fused(self) -> str | None:
        """Map the configured terms onto kernel slots; returns why not, or None when all are supported."""
        dev = self.device
        columns = [-1] * NUM_KINDS
        params = [[0.0, 0.0, 0.0, 0.0] for _ in range(NUM_KINDS)]
        bodies = {BODY_HEIGHT: None, BODY_POS: None}
        command_names = set()
        for col, (name, term_cfg) in enumerate(zip(self._term_names, self._term_cfgs)):
            kind = _KIND_OF_FUNC.get(term_cfg.func)
            if kind is None or columns[kind] != -1:
                return f"term '{name}' ({getattr(term_cfg.func, '__name__', term_cfg.func)})"
            if term_cfg.time_out != (kind == TIME_OUT):
                return f"term '{name}' with time_out={term_cfg.time_out}"
            p = dict(term_cfg.params)
            command_names.add(p.pop("command_name"))
            p.pop("asset_cfg", None)
            if kind in (ANCHOR_HEIGHT, BODY_HEIGHT):
                params[kind] = [
                    float(p.pop("threshold")),
                    float(bool(p.pop("threshold_adaptive", False))),
                    float(p.pop("down_threshold", 0.5)),
                    float(p.pop("root_height_threshold", 1.0 if kind == ANCHOR_HEIGHT else 0.5)),
                ]
            elif kind in (ANCHOR_ORI, BODY_POS):
                params[kind][0] = float(p.pop("threshold"))
            if kind in bodies:
                bodies[kind] = p.pop("body_names", None)
            if p:
                return f"term '{name}' parameters {sorted(p)}"
            columns[kind] = col
        if len(command_names) != 1:
            return f"commands {command_names}"
        command = self._env.command_manager.get_term(command_names.pop())
        if command.use_ref_motion_root_quat_w_as_anchor or getattr(command, "_offline", False):
            return "the command reads the anchor from the reference motion"
        self._command = command
        self._robot = command.robot
        self._columns = _int_array(columns, dev)
        self._params = wp.array(params, dtype=float, device=dev)
        self._tracked_bodies = _int_array(command.body_indexes.tolist(), dev)
        self._height_bodies = _int_array(
            _get_body_indexes(command, bodies[BODY_HEIGHT]).tolist(), dev
        )
        self._pos_bodies = _int_array(_get_body_indexes(command, bodies[BODY_POS]).tolist(), dev)
        self._time_out = wp.array(
            [bool(c.time_out) for c in self._term_cfgs], dtype=wp.bool, device=dev
        )
        self._env_origins = wp.from_torch(self._env.scene.env_origins.contiguous())
        self._term_dones_wp = wp.from_torch(self._term_dones)
        self._last_episode_dones_wp = wp.from_torch(self._last_episode_dones)
        self._truncated_wp = wp.from_torch(self._truncated_buf)
        self._terminated_wp = wp.from_torch(self._terminated_buf)
        return None

    def compute(self) -> torch.Tensor:
        if self._unsupported:
            return super().compute()
        command, lib = self._command, self._command.motion_lib

        def ref(t, dtype=None):
            return wp.from_torch(t.contiguous() if dtype is None else t.to(dtype).contiguous())

        if self._check:
            stock = torch.stack([c.func(self._env, **c.params) for c in self._term_cfgs], dim=1).to(
                torch.bool
            )
        data = self._robot.data
        wp.launch(
            _tracking_terminations_kernel,
            dim=self.num_envs,
            inputs=[
                ref(command.motion_ids, torch.int64),
                ref(command.motion_start_time_steps, torch.int64),
                ref(command.time_steps, torch.int64),
                ref(lib.length_starts, torch.int64),
                ref(lib._motion_num_frames, torch.int64),  # noqa: SLF001
                ref(lib.body_pos_w),
                ref(lib.body_quat_w),
                self._env_origins,
                ref(command.body_pos_relative_w),
                ref(command.running_ref_root_height),
                data.body_pos_w.warp,
                data.body_quat_w.warp,
                command.robot_anchor_body_index,
                command.motion_anchor_body_index,
                self._tracked_bodies,
                self._height_bodies,
                self._pos_bodies,
                self._columns,
                self._time_out,
                self._params,
            ],
            outputs=[
                self._term_dones_wp,
                self._last_episode_dones_wp,
                self._truncated_wp,
                self._terminated_wp,
            ],
            device=self.device,
        )
        if self._check:
            self._check_mismatches += (stock != self._term_dones).sum(dim=0)
            self._check_steps += 1
            if self._check_steps % 24 == 0:
                counts = ", ".join(
                    f"{n}={v}" for n, v in zip(self._term_names, self._check_mismatches.tolist())
                )
                print(
                    f"[FusedTerminationManager] mismatches over {self._check_steps} steps: {counts}"
                )  # noqa: T201
        return self._truncated_buf | self._terminated_buf
