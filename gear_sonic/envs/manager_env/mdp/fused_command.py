"""Warp kernels for the tracking command's per-step frame update (Direct-Warp).

Every control step the tracking command re-expresses the reference body poses around the robot's anchor
(``_update_relative_body_frames``; the sync-free step calls it twice, after the reset and after the cursor
advance) and computes ten tracking-error metrics (``_update_metrics``). In torch that is about a hundred
small kernels, most of them gathers of the same reference frame. ``enable(command)`` replaces both with one
Warp kernel each, one thread per env, reading the reference motion at the current frame from the motion
library and the robot state from Isaac Lab's Warp buffers. The relative frames and metrics are written into
the tensors the attributes already hold.

Enable with ++manager_env.config.fused_command=True. With ++manager_env.config.fused_command_check=True the
torch versions also run and the largest differences are printed every 24 steps.
"""

from __future__ import annotations

import types

import torch
import warp as wp

from gear_sonic.envs.manager_env.mdp.fused_rewards import (
    _int_array,
    _quat_apply,
    _quat_error,
    _quat_inv,
    _quat_mul,
    _wxyz,
)

# _update_metrics' keys, in the order it assigns them
METRICS = (
    "error_anchor_pos",
    "error_anchor_rot",
    "error_anchor_lin_vel",
    "error_anchor_ang_vel",
    "error_body_pos",
    "error_body_rot",
    "error_body_lin_vel",
    "error_body_ang_vel",
    "error_joint_pos",
    "error_joint_vel",
)


@wp.func
def _ref3(a: wp.array3d(dtype=float), f: wp.int64, b: int) -> wp.vec3:
    return wp.vec3(a[f, b, 0], a[f, b, 1], a[f, b, 2])


@wp.func
def _ref4(a: wp.array3d(dtype=float), f: wp.int64, b: int) -> wp.vec4:
    return wp.vec4(a[f, b, 0], a[f, b, 1], a[f, b, 2], a[f, b, 3])


@wp.func
def _frame(
    motion_ids: wp.array(dtype=wp.int64),
    starts: wp.array(dtype=wp.int64),
    time_steps: wp.array(dtype=wp.int64),
    length_starts: wp.array(dtype=wp.int64),
    e: int,
) -> wp.int64:
    return length_starts[motion_ids[e]] + starts[e] + time_steps[e]


@wp.kernel
def _relative_body_frames_kernel(
    motion_ids: wp.array(dtype=wp.int64),
    starts: wp.array(dtype=wp.int64),
    time_steps: wp.array(dtype=wp.int64),
    length_starts: wp.array(dtype=wp.int64),
    ref_body_pos: wp.array3d(dtype=float),
    ref_body_quat: wp.array3d(dtype=float),
    env_origins: wp.array2d(dtype=float),
    body_pos: wp.array2d(dtype=wp.vec3f),
    body_quat: wp.array2d(dtype=wp.quatf),
    robot_anchor: int,
    motion_anchor: int,
    out_pos: wp.array3d(dtype=float),
    out_quat: wp.array3d(dtype=float),
):
    e = wp.tid()
    f = _frame(motion_ids, starts, time_steps, length_starts, e)
    origin = wp.vec3(env_origins[e, 0], env_origins[e, 1], env_origins[e, 2])
    anchor_pos = _ref3(ref_body_pos, f, motion_anchor) + origin
    robot_pos = body_pos[e, robot_anchor]
    # torch_transform.get_heading_q of robot_anchor * inv(ref_anchor): x and y zeroed, normalized
    q = _quat_mul(
        _wxyz(body_quat[e, robot_anchor]), _quat_inv(_ref4(ref_body_quat, f, motion_anchor))
    )
    heading = wp.vec4(q[0], 0.0, 0.0, q[3])
    heading = heading / wp.max(wp.length(heading), 1.0e-9)
    delta = wp.vec3(robot_pos[0], robot_pos[1], anchor_pos[2])
    for b in range(out_pos.shape[1]):
        p = delta + _quat_apply(heading, (_ref3(ref_body_pos, f, b) + origin) - anchor_pos)
        r = _quat_mul(heading, _ref4(ref_body_quat, f, b))
        for i in range(3):
            out_pos[e, b, i] = p[i]
        for i in range(4):
            out_quat[e, b, i] = r[i]


@wp.kernel
def _metrics_kernel(
    motion_ids: wp.array(dtype=wp.int64),
    starts: wp.array(dtype=wp.int64),
    time_steps: wp.array(dtype=wp.int64),
    length_starts: wp.array(dtype=wp.int64),
    ref_body_pos: wp.array3d(dtype=float),
    ref_body_quat: wp.array3d(dtype=float),
    ref_body_lin_vel: wp.array3d(dtype=float),
    ref_body_ang_vel: wp.array3d(dtype=float),
    ref_dof_pos: wp.array2d(dtype=float),
    ref_dof_vel: wp.array2d(dtype=float),
    env_origins: wp.array2d(dtype=float),
    body_pos_relative: wp.array3d(dtype=float),
    body_quat_relative: wp.array3d(dtype=float),
    body_pos: wp.array2d(dtype=wp.vec3f),
    body_quat: wp.array2d(dtype=wp.quatf),
    body_lin_vel: wp.array2d(dtype=wp.vec3f),
    body_ang_vel: wp.array2d(dtype=wp.vec3f),
    joint_pos: wp.array2d(dtype=float),
    joint_vel: wp.array2d(dtype=float),
    robot_anchor: int,
    motion_anchor: int,
    tracked_bodies: wp.array(dtype=int),
    joints: wp.array(dtype=int),
    metrics: wp.array2d(dtype=float),
):
    e = wp.tid()
    f = _frame(motion_ids, starts, time_steps, length_starts, e)
    origin = wp.vec3(env_origins[e, 0], env_origins[e, 1], env_origins[e, 2])
    metrics[e, 0] = wp.length(
        _ref3(ref_body_pos, f, motion_anchor) + origin - body_pos[e, robot_anchor]
    )
    metrics[e, 1] = _quat_error(
        _ref4(ref_body_quat, f, motion_anchor), _wxyz(body_quat[e, robot_anchor])
    )
    metrics[e, 2] = wp.length(
        _ref3(ref_body_lin_vel, f, motion_anchor) - body_lin_vel[e, robot_anchor]
    )
    metrics[e, 3] = wp.length(
        _ref3(ref_body_ang_vel, f, motion_anchor) - body_ang_vel[e, robot_anchor]
    )
    pos, rot, lin, ang = float(0.0), float(0.0), float(0.0), float(0.0)
    nb = tracked_bodies.shape[0]
    for b in range(nb):
        k = tracked_bodies[b]
        rel = wp.vec3(
            body_pos_relative[e, b, 0], body_pos_relative[e, b, 1], body_pos_relative[e, b, 2]
        )
        pos += wp.length(rel - body_pos[e, k])
        rel_q = wp.vec4(
            body_quat_relative[e, b, 0],
            body_quat_relative[e, b, 1],
            body_quat_relative[e, b, 2],
            body_quat_relative[e, b, 3],
        )
        rot += _quat_error(rel_q, _wxyz(body_quat[e, k]))
        lin += wp.length(_ref3(ref_body_lin_vel, f, b) - body_lin_vel[e, k])
        ang += wp.length(_ref3(ref_body_ang_vel, f, b) - body_ang_vel[e, k])
    metrics[e, 4] = pos / float(nb)
    metrics[e, 5] = rot / float(nb)
    metrics[e, 6] = lin / float(nb)
    metrics[e, 7] = ang / float(nb)
    jp, jv = float(0.0), float(0.0)
    nj = joints.shape[0]
    for j in range(nj):
        jp += wp.abs(ref_dof_pos[f, j] - joint_pos[e, joints[j]])
        jv += wp.abs(ref_dof_vel[f, j] - joint_vel[e, joints[j]])
    metrics[e, 8] = jp / float(nj)
    metrics[e, 9] = jv / float(nj)


def enable(command, check: bool = False):
    """Run ``command``'s relative body frames and metrics as Warp kernels."""
    if command.use_ref_motion_root_quat_w_as_anchor or getattr(command, "_offline", False):
        raise NotImplementedError("fused_command reads the robot anchor from the simulator")
    dev = command.device
    num_joints = command.motion_lib.dof_pos.shape[-1]
    joints = command.body_joint_indices if command.has_dof_mismatch else range(num_joints)
    command._fused = types.SimpleNamespace(  # noqa: SLF001
        tracked=_int_array(command.body_indexes.tolist(), dev),
        joints=_int_array([int(j) for j in joints], dev),
        origins=wp.from_torch(command._env.scene.env_origins.contiguous()),  # noqa: SLF001
        metrics=torch.zeros((command.num_envs, len(METRICS)), device=dev),
        columns={name: i for i, name in enumerate(METRICS)},
        check=check,
        steps=0,
        worst=torch.zeros(len(METRICS) + 2, device=dev),
        stock_frames=command._update_relative_body_frames,  # noqa: SLF001
        stock_metrics=command._update_metrics,  # noqa: SLF001
    )
    for i, name in enumerate(METRICS):
        command.metrics[name] = command._fused.metrics[:, i]  # noqa: SLF001
    command._update_relative_body_frames = types.MethodType(  # noqa: SLF001
        _update_relative_body_frames, command
    )
    command._update_metrics = types.MethodType(_update_metrics, command)  # noqa: SLF001


def _ref(t: torch.Tensor, dtype=None):
    return wp.from_torch(t.contiguous() if dtype is None else t.to(dtype).contiguous())


def _cursor(command):
    lib = command.motion_lib
    return [
        _ref(command.motion_ids, torch.int64),
        _ref(command.motion_start_time_steps, torch.int64),
        _ref(command.time_steps, torch.int64),
        _ref(lib.length_starts, torch.int64),
    ]


def _stream(command):
    # the current torch stream, also when sync_free records the step as a CUDA graph
    return wp.stream_from_torch(torch.cuda.current_stream(command.device))


def _update_relative_body_frames(self):
    fused = self._fused  # noqa: SLF001
    if fused.check:
        fused.stock_frames()
        stock = self.body_pos_relative_w.clone(), self.body_quat_relative_w.clone()
    lib, data = self.motion_lib, self.robot.data
    wp.launch(
        _relative_body_frames_kernel,
        dim=self.num_envs,
        inputs=_cursor(self)
        + [
            _ref(lib.body_pos_w),
            _ref(lib.body_quat_w),
            fused.origins,
            data.body_pos_w.warp,
            data.body_quat_w.warp,
            self.robot_anchor_body_index,
            self.motion_anchor_body_index,
        ],
        outputs=[wp.from_torch(self.body_pos_relative_w), wp.from_torch(self.body_quat_relative_w)],
        device=self.device,
        stream=_stream(self),
    )
    if fused.check:
        n = len(METRICS)
        fused.worst[n] = torch.maximum(
            fused.worst[n], (self.body_pos_relative_w - stock[0]).abs().max()
        )
        fused.worst[n + 1] = torch.maximum(
            fused.worst[n + 1], (self.body_quat_relative_w - stock[1]).abs().max()
        )


def _update_metrics(self):
    fused = self._fused  # noqa: SLF001
    if fused.check:
        metrics = dict(self.metrics)
        fused.stock_metrics()
        stock = torch.stack([self.metrics[name] for name in METRICS], dim=1)
        self.metrics.update(metrics)
    lib, data = self.motion_lib, self.robot.data
    wp.launch(
        _metrics_kernel,
        dim=self.num_envs,
        inputs=_cursor(self)
        + [
            _ref(lib.body_pos_w),
            _ref(lib.body_quat_w),
            _ref(lib.body_lin_vel_w),
            _ref(lib.body_ang_vel_w),
            _ref(lib.dof_pos),
            _ref(lib.dof_vel),
            fused.origins,
            _ref(self.body_pos_relative_w),
            _ref(self.body_quat_relative_w),
            data.body_pos_w.warp,
            data.body_quat_w.warp,
            data.body_lin_vel_w.warp,
            data.body_ang_vel_w.warp,
            data.joint_pos.warp,
            data.joint_vel.warp,
            self.robot_anchor_body_index,
            self.motion_anchor_body_index,
            fused.tracked,
            fused.joints,
        ],
        outputs=[wp.from_torch(fused.metrics)],
        device=self.device,
        stream=_stream(self),
    )
    if fused.check:
        n = len(METRICS)
        fused.worst[:n] = torch.maximum(fused.worst[:n], (fused.metrics - stock).abs().amax(dim=0))
        fused.steps += 1
        if fused.steps % 24 == 0:
            names = list(METRICS) + ["body_pos_relative_w", "body_quat_relative_w"]
            worst = ", ".join(f"{n}={v:.2e}" for n, v in zip(names, fused.worst.tolist()))
            print(f"[fused_command] max abs diff over {fused.steps} steps: {worst}")  # noqa: T201
