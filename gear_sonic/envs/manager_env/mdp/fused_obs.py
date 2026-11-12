"""Warp kernels for the observation terms that gather the reference motion or the robot's body frames.

The tokenizer group rebuilds the SMPL root orientation of every future frame from the motion library
(axis-angle to quaternion, Y-up to Z-up, SMPL base rotation removed) and expresses the SMPL joints, or the
root orientation relative to the robot anchor, in that frame; the policy and critic groups gather future
joint targets, root heights and orientations, and the robot's bodies, the reference anchor and the VR points
in anchor frames. In torch these are about 150 small kernels per control step. ``enable(env)`` swaps each
supported term function for one Warp kernel, one thread per env (and future frame), writing into a buffer
allocated once (the observation manager copies or adds noise to it as before). The quaternion products use
the same formula as the torch code.

Enable with ++manager_env.config.fused_obs=True. With ++manager_env.config.fused_obs_check=True the torch terms
also run and the largest differences are printed every 24 steps.
"""

from __future__ import annotations

import math

import torch
import warp as wp

from gear_sonic.envs.manager_env import mdp
from gear_sonic.envs.manager_env.mdp.fused_warp import (
    Cursor,
    cursor,
    env3,
    frame,
    future_frame,
    int_array,
    lib3,
    lib4,
    qapply,
    qinv,
    qmul,
    ref,
    rot6d,
    wxyz,
)
from gear_sonic.trl.utils import torch_transform


@wp.func
def _smpl_root_quat(smpl_pose: wp.array2d(dtype=float), f: wp.int64, y_up: int, y_to_z: wp.vec4) -> wp.vec4:
    """TrackingCommand.smpl_root_quat_w_multi_future for one frame: kornia's angle_axis_to_quaternion of the
    root axis-angle, Y-up to Z-up if needed, then remove_smpl_base_rot."""
    a = wp.vec3(smpl_pose[f, 0], smpl_pose[f, 1], smpl_pose[f, 2])
    theta_sq = wp.dot(a, a)
    theta = wp.sqrt(wp.max(theta_sq, 1.0e-6))
    half = theta * 0.5
    k = float(0.5)
    w = float(1.0)
    if theta_sq > 0.0:
        den = theta
        if wp.abs(den) < 1.0e-6:
            den = den + 1.0e-6
        k = wp.sin(half) / den
        w = wp.cos(half)
    q = wp.vec4(w, a[0] * k, a[1] * k, a[2] * k)
    if y_up != 0:
        q = qmul(y_to_z, q)
    # conjugate of the SMPL rest orientation (0.5, 0.5, 0.5, 0.5)
    return qmul(q, wp.vec4(0.5, -0.5, -0.5, -0.5))


@wp.func
def _write6(out: wp.array3d(dtype=float), e: int, k: int, r: wp.vec4):
    v = rot6d(r)
    for i in range(6):
        out[e, k, i] = v[i]


@wp.kernel
def _smpl_joints_local_kernel(
    c: Cursor,
    future_init: wp.array(dtype=float),
    smpl_pose: wp.array2d(dtype=float),
    smpl_joints: wp.array3d(dtype=float),
    y_up: int,
    y_to_z: wp.vec4,
    out: wp.array3d(dtype=float),
):
    e, k = wp.tid()
    f = future_frame(c, future_init, e, k)
    inv = qinv(_smpl_root_quat(smpl_pose, f, y_up, y_to_z))
    for j in range(smpl_joints.shape[1]):
        p = qapply(inv, lib3(smpl_joints, f, j))
        out[e, k, 3 * j + 0] = p[0]
        out[e, k, 3 * j + 1] = p[1]
        out[e, k, 3 * j + 2] = p[2]


@wp.kernel
def _smpl_root_ori_b_kernel(
    c: Cursor,
    future_init: wp.array(dtype=float),
    smpl_pose: wp.array2d(dtype=float),
    y_up: int,
    y_to_z: wp.vec4,
    body_quat: wp.array2d(dtype=wp.quatf),
    robot_anchor: int,
    out: wp.array3d(dtype=float),
):
    e, k = wp.tid()
    f = future_frame(c, future_init, e, k)
    q = _smpl_root_quat(smpl_pose, f, y_up, y_to_z)
    _write6(out, e, k, qmul(qinv(wxyz(body_quat[e, robot_anchor])), q))


@wp.kernel
def _joint_future_kernel(
    c: Cursor,
    future_init: wp.array(dtype=float),
    dof_pos: wp.array2d(dtype=float),
    dof_vel: wp.array2d(dtype=float),
    joints: wp.array(dtype=wp.int32),
    out: wp.array2d(dtype=float),
):
    """cat([joint_pos_multi_future, joint_vel_multi_future], dim=1) over ``joints``: all positions first."""
    e, k = wp.tid()
    f = future_frame(c, future_init, e, k)
    nj = joints.shape[0]
    vel_offset = future_init.shape[0] * nj
    for j in range(nj):
        out[e, k * nj + j] = dof_pos[f, joints[j]]
        out[e, vel_offset + k * nj + j] = dof_vel[f, joints[j]]


@wp.kernel
def _root_z_future_kernel(
    c: Cursor,
    future_init: wp.array(dtype=float),
    body_pos: wp.array3d(dtype=float),
    out: wp.array2d(dtype=float),
):
    """TrackingCommand.root_z_multi_future."""
    e, k = wp.tid()
    out[e, k] = body_pos[future_frame(c, future_init, e, k), 0, 2]


@wp.kernel
def _root_ori_b_future_kernel(
    c: Cursor,
    future_init: wp.array(dtype=float),
    lib_body_quat: wp.array3d(dtype=float),
    body_quat: wp.array2d(dtype=wp.quatf),
    robot_anchor: int,
    out: wp.array3d(dtype=float),
):
    """TrackingCommand.root_rot_dif_l_multi_future."""
    e, k = wp.tid()
    f = future_frame(c, future_init, e, k)
    _write6(out, e, k, qmul(qinv(wxyz(body_quat[e, robot_anchor])), lib4(lib_body_quat, f, 0)))


@wp.kernel
def _robot_body_b_kernel(
    body_pos: wp.array2d(dtype=wp.vec3f),
    body_quat: wp.array2d(dtype=wp.quatf),
    robot_anchor: int,
    tracked_bodies: wp.array(dtype=wp.int32),
    with_ori: int,
    out: wp.array2d(dtype=float),
):
    """robot_body_pos_b / robot_body_ori_b: subtract_frame_transforms from the robot anchor."""
    e = wp.tid()
    inv = qinv(wxyz(body_quat[e, robot_anchor]))
    anchor = body_pos[e, robot_anchor]
    for b in range(tracked_bodies.shape[0]):
        k = tracked_bodies[b]
        if with_ori == 0:
            p = qapply(inv, body_pos[e, k] - anchor)
            for i in range(3):
                out[e, 3 * b + i] = p[i]
        else:
            r = rot6d(qmul(inv, wxyz(body_quat[e, k])))
            for i in range(6):
                out[e, 6 * b + i] = r[i]


@wp.kernel
def _motion_anchor_b_kernel(
    c: Cursor,
    lib_body_pos: wp.array3d(dtype=float),
    lib_body_quat: wp.array3d(dtype=float),
    env_origins: wp.array2d(dtype=float),
    body_pos: wp.array2d(dtype=wp.vec3f),
    body_quat: wp.array2d(dtype=wp.quatf),
    robot_anchor: int,
    motion_anchor: int,
    with_ori: int,
    repeat: int,
    out: wp.array2d(dtype=float),
):
    """motion_anchor_pos_b / motion_anchor_ori_b: the reference anchor in the robot anchor frame."""
    e = wp.tid()
    f = frame(c, e)
    inv = qinv(wxyz(body_quat[e, robot_anchor]))
    if with_ori == 0:
        ref_pos = lib3(lib_body_pos, f, motion_anchor) + env3(env_origins, e)
        p = qapply(inv, ref_pos - body_pos[e, robot_anchor])
        for i in range(3):
            out[e, i] = p[i]
    else:
        r = rot6d(qmul(inv, lib4(lib_body_quat, f, motion_anchor)))
        for k in range(repeat):
            for i in range(6):
                out[e, 6 * k + i] = r[i]


@wp.kernel
def _vr_3point_kernel(
    c: Cursor,
    lib_body_pos: wp.array3d(dtype=float),
    lib_body_quat: wp.array3d(dtype=float),
    env_origins: wp.array2d(dtype=float),
    points: wp.array(dtype=wp.int32),
    offsets: wp.array3d(dtype=float),
    motion_anchor: int,
    with_orn: int,
    out: wp.array2d(dtype=float),
):
    """vr_3point_local_target / vr_3point_local_orn_target: the reference VR points (with their offsets) in the
    reference anchor frame."""
    e = wp.tid()
    f = frame(c, e)
    origin = env3(env_origins, e)
    inv = qinv(lib4(lib_body_quat, f, motion_anchor))
    anchor_pos = lib3(lib_body_pos, f, motion_anchor) + origin
    for p in range(points.shape[0]):
        b = points[p]
        q = lib4(lib_body_quat, f, b)
        if with_orn == 0:
            pos = lib3(lib_body_pos, f, b) + qapply(q, lib3(offsets, wp.int64(e), p)) + origin
            v = qapply(inv, pos - anchor_pos)
            for i in range(3):
                out[e, 3 * p + i] = v[i]
        else:
            r = qmul(inv, q)
            for i in range(4):
                out[e, 4 * p + i] = r[i]


class _WarpTerm:
    """An observation term evaluated by one Warp kernel, per env or per (env, future frame); ``check``
    compares it with the torch term on every call."""

    def __init__(self, env, stock, params, kind, check):
        self.stock, self.kind, self.check = stock, kind, check
        self.command = command = env.command_manager.get_term(params["command_name"])
        lib, dev, n = command.motion_lib, command.device, command.num_envs
        smpl = kind.startswith("smpl")
        init = command.smpl_future_time_steps_init if smpl else command.future_time_steps_init
        self._future_init = init[0].to(torch.float32).contiguous()
        self.future_init = wp.from_torch(self._future_init)
        frames = self.future_init.shape[0]
        flat = not params.get("non_flatten", False)
        if kind == "smpl_joints_local":
            num_joints = lib._motion_smpl_joints.shape[1]  # noqa: SLF001
            self.out = self.result = torch.zeros((n, frames, 3 * num_joints), device=dev)
        elif kind in ("smpl_root_ori_b", "root_ori_b_future"):
            self.out = torch.zeros((n, frames, 6), device=dev)
            self.result = self.out.view(n, -1) if flat else self.out
        elif kind in ("joint_future", "joint_future_lower"):
            joints = (
                command.lower_joint_isaaclab_indices
                if kind == "joint_future_lower"
                else range(lib.dof_pos.shape[-1])
            )
            self.joints = int_array(joints, dev)
            self.out = torch.zeros((n, 2 * frames * self.joints.shape[0]), device=dev)
            self.result = self.out if flat else self.out.view(n, frames, -1)
        elif kind == "root_z_future":
            self.out = torch.zeros((n, frames), device=dev)
            self.result = self.out if flat else self.out.view(n, frames, -1)
        elif kind in ("robot_body_pos_b", "robot_body_ori_b"):
            self.tracked = int_array(command.body_indexes, dev)
            width = 3 if kind == "robot_body_pos_b" else 6
            self.out = self.result = torch.zeros((n, width * len(command.body_indexes)), device=dev)
        elif kind == "motion_anchor_pos_b":
            self.out = self.result = torch.zeros((n, 3), device=dev)
        elif kind == "motion_anchor_ori_b":
            self.repeat = 1 if flat else command.num_future_frames
            self.out = torch.zeros((n, 6 * self.repeat), device=dev)
            self.result = self.out if flat else self.out.view(n, self.repeat, 6)
        elif kind in ("vr_3point_local_target", "vr_3point_local_orn_target"):
            self.points = int_array(command._vr_3point_idx_motion, dev)  # noqa: SLF001
            width = 3 if kind == "vr_3point_local_target" else 4
            self.out = self.result = torch.zeros((n, width * len(command.cfg.vr_3point_body)), device=dev)
        # the constant of TrackingCommand.smpl_root_ytoz_up, computed the same way
        y_to_z = torch_transform.angle_axis_to_quaternion(torch.tensor([[math.pi / 2, 0.0, 0.0]], device=dev))
        self.y_to_z = wp.vec4(*y_to_z[0].tolist())
        self.y_up = int(bool(getattr(lib, "smpl_y_up", False)))
        self.out_wp = wp.from_torch(self.out)
        self.steps, self.worst = 0, torch.zeros((), device=dev)

    def _launch_args(self):
        """(kernel, launch dim, inputs) for this term."""
        command, kind = self.command, self.kind
        lib, data, n = command.motion_lib, command.robot.data, command.num_envs
        if kind in ("robot_body_pos_b", "robot_body_ori_b"):
            return _robot_body_b_kernel, n, [
                data.body_pos_w.warp,
                data.body_quat_w.warp,
                command.robot_anchor_body_index,
                self.tracked,
                int(kind == "robot_body_ori_b"),
            ]
        c = cursor(command)
        origins = ref(command._env.scene.env_origins)  # noqa: SLF001
        if kind in ("motion_anchor_pos_b", "motion_anchor_ori_b"):
            return _motion_anchor_b_kernel, n, [
                c,
                ref(lib.body_pos_w),
                ref(lib.body_quat_w),
                origins,
                data.body_pos_w.warp,
                data.body_quat_w.warp,
                command.robot_anchor_body_index,
                command.motion_anchor_body_index,
                int(kind == "motion_anchor_ori_b"),
                getattr(self, "repeat", 1),
            ]
        if kind in ("vr_3point_local_target", "vr_3point_local_orn_target"):
            return _vr_3point_kernel, n, [
                c,
                ref(lib.body_pos_w),
                ref(lib.body_quat_w),
                origins,
                self.points,
                ref(command.vr_3point_body_offsets),
                command.motion_anchor_body_index,
                int(kind == "vr_3point_local_orn_target"),
            ]
        dim = (n, self.future_init.shape[0])
        if kind == "smpl_joints_local":
            smpl = [ref(lib._motion_smpl_poses), ref(lib._motion_smpl_joints)]  # noqa: SLF001
            return _smpl_joints_local_kernel, dim, [c, self.future_init, *smpl, self.y_up, self.y_to_z]
        if kind == "smpl_root_ori_b":
            return _smpl_root_ori_b_kernel, dim, [
                c,
                self.future_init,
                ref(lib._motion_smpl_poses),  # noqa: SLF001
                self.y_up,
                self.y_to_z,
                data.body_quat_w.warp,
                command.robot_anchor_body_index,
            ]
        if kind in ("joint_future", "joint_future_lower"):
            return _joint_future_kernel, dim, [
                c,
                self.future_init,
                ref(lib.dof_pos),
                ref(lib.dof_vel),
                self.joints,
            ]
        if kind == "root_z_future":
            return _root_z_future_kernel, dim, [c, self.future_init, ref(lib.body_pos_w)]
        return _root_ori_b_future_kernel, dim, [
            c,
            self.future_init,
            ref(lib.body_quat_w),
            data.body_quat_w.warp,
            command.robot_anchor_body_index,
        ]

    def __call__(self, env, **params):
        kernel, dim, inputs = self._launch_args()
        wp.launch(kernel, dim=dim, inputs=inputs, outputs=[self.out_wp], device=self.command.device)
        if self.check:
            stock = self.stock(env, **params)
            self.worst = torch.maximum(self.worst, (self.result - stock).abs().max())
            self.steps += 1
            if self.steps % 24 == 0:
                print(  # noqa: T201
                    f"[fused_obs] {self.kind}: max abs diff over {self.steps} steps {self.worst.item():.2e}"
                )
        return self.result


# term function -> (kind, accepted params, whether non_flatten=True is required)
_SUPPORTED = {
    mdp.smpl_joints_multi_future_local: ("smpl_joints_local", {"command_name", "non_flatten"}, True),
    mdp.smpl_root_ori_b_mf: ("smpl_root_ori_b", {"command_name", "non_flatten"}, True),
    mdp.command_multi_future: ("joint_future", {"command_name", "non_flatten"}, False),
    mdp.command_multi_future_lower_body: ("joint_future_lower", {"command_name"}, False),
    mdp.command_z_multi_future: ("root_z_future", {"command_name", "non_flatten"}, False),
    mdp.motion_anchor_ori_b_mf: ("root_ori_b_future", {"command_name", "non_flatten"}, False),
    mdp.robot_body_pos_b: ("robot_body_pos_b", {"command_name"}, False),
    mdp.robot_body_ori_b: ("robot_body_ori_b", {"command_name"}, False),
    mdp.motion_anchor_pos_b: ("motion_anchor_pos_b", {"command_name"}, False),
    mdp.motion_anchor_ori_b: ("motion_anchor_ori_b", {"command_name", "non_flatten"}, False),
    mdp.vr_3point_local_target: ("vr_3point_local_target", {"command_name"}, False),
    mdp.vr_3point_local_orn_target: ("vr_3point_local_orn_target", {"command_name"}, False),
}


def enable(env, check: bool = False) -> list[str]:
    """Swap supported observation terms for their Warp kernels; returns the swapped term names."""
    manager = env.observation_manager
    swapped = []
    for group, names in manager._group_obs_term_names.items():  # noqa: SLF001
        for name, term_cfg in zip(names, manager._group_obs_term_cfgs[group]):  # noqa: SLF001
            entry = _SUPPORTED.get(term_cfg.func)
            if entry is None:
                continue
            kind, accepted, needs_non_flatten = entry
            # unset parameters, and Isaac Lab 3.0's filled-in mask_out_z=False, are the kernels' defaults
            params = {
                k: v for k, v in term_cfg.params.items() if v is not None and not (k == "mask_out_z" and not v)
            }
            if set(params) - accepted or (needs_non_flatten and not params.get("non_flatten")):
                continue
            term_cfg.func = _WarpTerm(env, term_cfg.func, term_cfg.params, kind, check)
            swapped.append(f"{group}/{name}")
    print(f"[fused_obs] Warp terms: {swapped}")  # noqa: T201
    return swapped
