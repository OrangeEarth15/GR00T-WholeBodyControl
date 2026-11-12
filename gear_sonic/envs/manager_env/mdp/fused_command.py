"""Warp kernels for the tracking command's per-step frame update.

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

from gear_sonic.envs.manager_env.mdp.fused_warp import (
    Cursor,
    cursor,
    env3,
    frame,
    int_array,
    lib3,
    lib4,
    qapply,
    qinv,
    qmul,
    quat_error,
    ref,
    wxyz,
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


@wp.kernel
def _relative_body_frames_kernel(
    c: Cursor,
    lib_body_pos: wp.array3d(dtype=float),
    lib_body_quat: wp.array3d(dtype=float),
    env_origins: wp.array2d(dtype=float),
    body_pos: wp.array2d(dtype=wp.vec3f),
    body_quat: wp.array2d(dtype=wp.quatf),
    robot_anchor: int,
    motion_anchor: int,
    out_pos: wp.array3d(dtype=float),
    out_quat: wp.array3d(dtype=float),
):
    e = wp.tid()
    f = frame(c, e)
    origin = env3(env_origins, e)
    anchor_pos = lib3(lib_body_pos, f, motion_anchor) + origin
    robot_pos = body_pos[e, robot_anchor]
    # torch_transform.get_heading_q of robot_anchor * inv(ref_anchor): x and y zeroed, normalized
    q = qmul(wxyz(body_quat[e, robot_anchor]), qinv(lib4(lib_body_quat, f, motion_anchor)))
    heading = wp.vec4(q[0], 0.0, 0.0, q[3])
    heading = heading / wp.max(wp.length(heading), 1.0e-9)
    delta = wp.vec3(robot_pos[0], robot_pos[1], anchor_pos[2])
    for b in range(out_pos.shape[1]):
        p = delta + qapply(heading, (lib3(lib_body_pos, f, b) + origin) - anchor_pos)
        r = qmul(heading, lib4(lib_body_quat, f, b))
        for i in range(3):
            out_pos[e, b, i] = p[i]
        for i in range(4):
            out_quat[e, b, i] = r[i]


@wp.kernel
def _metrics_kernel(
    c: Cursor,
    lib_body_pos: wp.array3d(dtype=float),
    lib_body_quat: wp.array3d(dtype=float),
    lib_body_lin_vel: wp.array3d(dtype=float),
    lib_body_ang_vel: wp.array3d(dtype=float),
    lib_dof_pos: wp.array2d(dtype=float),
    lib_dof_vel: wp.array2d(dtype=float),
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
    tracked_bodies: wp.array(dtype=wp.int32),
    joints: wp.array(dtype=wp.int32),
    metrics: wp.array2d(dtype=float),
):
    e = wp.tid()
    f = frame(c, e)
    ee = wp.int64(e)
    metrics[e, 0] = wp.length(
        lib3(lib_body_pos, f, motion_anchor) + env3(env_origins, e) - body_pos[e, robot_anchor]
    )
    metrics[e, 1] = quat_error(lib4(lib_body_quat, f, motion_anchor), wxyz(body_quat[e, robot_anchor]))
    metrics[e, 2] = wp.length(lib3(lib_body_lin_vel, f, motion_anchor) - body_lin_vel[e, robot_anchor])
    metrics[e, 3] = wp.length(lib3(lib_body_ang_vel, f, motion_anchor) - body_ang_vel[e, robot_anchor])
    pos, rot, lin, ang = float(0.0), float(0.0), float(0.0), float(0.0)
    nb = tracked_bodies.shape[0]
    for b in range(nb):
        k = tracked_bodies[b]
        pos += wp.length(lib3(body_pos_relative, ee, b) - body_pos[e, k])
        rot += quat_error(lib4(body_quat_relative, ee, b), wxyz(body_quat[e, k]))
        lin += wp.length(lib3(lib_body_lin_vel, f, b) - body_lin_vel[e, k])
        ang += wp.length(lib3(lib_body_ang_vel, f, b) - body_ang_vel[e, k])
    metrics[e, 4] = pos / float(nb)
    metrics[e, 5] = rot / float(nb)
    metrics[e, 6] = lin / float(nb)
    metrics[e, 7] = ang / float(nb)
    jp, jv = float(0.0), float(0.0)
    nj = joints.shape[0]
    for j in range(nj):
        jp += wp.abs(lib_dof_pos[f, j] - joint_pos[e, joints[j]])
        jv += wp.abs(lib_dof_vel[f, j] - joint_vel[e, joints[j]])
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
        tracked=int_array(command.body_indexes.tolist(), dev),
        joints=int_array(joints, dev),
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
    command._update_relative_body_frames = types.MethodType(_update_relative_body_frames, command)  # noqa: SLF001
    command._update_metrics = types.MethodType(_update_metrics, command)  # noqa: SLF001


def _update_relative_body_frames(self):
    fused = self._fused  # noqa: SLF001
    if fused.check:
        fused.stock_frames()
        stock = self.body_pos_relative_w.clone(), self.body_quat_relative_w.clone()
    lib, data = self.motion_lib, self.robot.data
    wp.launch(
        _relative_body_frames_kernel,
        dim=self.num_envs,
        inputs=[
            cursor(self),
            ref(lib.body_pos_w),
            ref(lib.body_quat_w),
            ref(self._env.scene.env_origins),  # noqa: SLF001
            data.body_pos_w.warp,
            data.body_quat_w.warp,
            self.robot_anchor_body_index,
            self.motion_anchor_body_index,
        ],
        outputs=[wp.from_torch(self.body_pos_relative_w), wp.from_torch(self.body_quat_relative_w)],
        device=self.device,
    )
    if fused.check:
        n = len(METRICS)
        fused.worst[n] = torch.maximum(fused.worst[n], (self.body_pos_relative_w - stock[0]).abs().max())
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
        inputs=[
            cursor(self),
            ref(lib.body_pos_w),
            ref(lib.body_quat_w),
            ref(lib.body_lin_vel_w),
            ref(lib.body_ang_vel_w),
            ref(lib.dof_pos),
            ref(lib.dof_vel),
            ref(self._env.scene.env_origins),  # noqa: SLF001
            ref(self.body_pos_relative_w),
            ref(self.body_quat_relative_w),
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
    )
    if fused.check:
        n = len(METRICS)
        fused.worst[:n] = torch.maximum(fused.worst[:n], (fused.metrics - stock).abs().amax(dim=0))
        fused.steps += 1
        if fused.steps % 24 == 0:
            names = list(METRICS) + ["body_pos_relative_w", "body_quat_relative_w"]
            worst = ", ".join(f"{n}={v:.2e}" for n, v in zip(names, fused.worst.tolist()))
            print(f"[fused_command] max abs diff over {fused.steps} steps: {worst}")  # noqa: T201
