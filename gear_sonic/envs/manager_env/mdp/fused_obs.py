"""Warp kernels for the heaviest tokenizer observation terms (Direct-Warp).

The SMPL terms of the tokenizer group rebuild the SMPL root orientation of every future frame from the
motion library (axis-angle to quaternion, Y-up to Z-up, SMPL base rotation removed) and then express the
SMPL joints, or the root orientation relative to the robot anchor, in that frame: about 90 small torch kernels
per control step. ``enable(env)`` swaps the term functions for one Warp kernel each, one thread per env and
future frame, writing into a buffer allocated once (the observation manager copies or adds noise to it as
before). The quaternion products use the same formula as the torch code.

Enable with ++manager_env.config.fused_obs=True. With ++manager_env.config.fused_obs_check=True the torch terms
also run and the largest differences are printed every 24 steps.
"""

from __future__ import annotations

import math

import torch
import warp as wp

from gear_sonic.envs.manager_env import mdp
from gear_sonic.envs.manager_env.mdp.fused_rewards import _wxyz
from gear_sonic.trl.utils import torch_transform


@wp.func
def _qmul(a: wp.vec4, b: wp.vec4) -> wp.vec4:
    """(w, x, y, z) product with the operation order of math_wxyz.quat_mul / rotations.quat_mul."""
    ww = (a[3] + a[1]) * (b[1] + b[2])
    yy = (a[0] - a[2]) * (b[0] + b[3])
    zz = (a[0] + a[2]) * (b[0] - b[3])
    xx = ww + yy + zz
    qq = 0.5 * (xx + (a[3] - a[1]) * (b[1] - b[2]))
    return wp.vec4(
        qq - ww + (a[3] - a[2]) * (b[2] - b[3]),
        qq - xx + (a[1] + a[0]) * (b[1] + b[0]),
        qq - yy + (a[0] - a[1]) * (b[2] + b[3]),
        qq - zz + (a[3] + a[2]) * (b[0] - b[1]),
    )


@wp.func
def _qinv(q: wp.vec4) -> wp.vec4:
    """math_wxyz.quat_inv."""
    return wp.vec4(q[0], -q[1], -q[2], -q[3]) / wp.max(wp.dot(q, q), 1.0e-9)


@wp.func
def _qapply(q: wp.vec4, v: wp.vec3) -> wp.vec3:
    """math_wxyz.quat_apply."""
    xyz = wp.vec3(q[1], q[2], q[3])
    t = wp.cross(xyz, v) * 2.0
    return v + q[0] * t + wp.cross(xyz, t)


@wp.func
def _write_rot6d(out: wp.array3d(dtype=float), e: int, k: int, r: wp.vec4):
    """math_wxyz.matrix_from_quat, first two columns, row by row."""
    two_s = 2.0 / wp.dot(r, r)
    w, x, y, z = r[0], r[1], r[2], r[3]
    out[e, k, 0] = 1.0 - two_s * (y * y + z * z)
    out[e, k, 1] = two_s * (x * y - z * w)
    out[e, k, 2] = two_s * (x * y + z * w)
    out[e, k, 3] = 1.0 - two_s * (x * x + z * z)
    out[e, k, 4] = two_s * (x * z - y * w)
    out[e, k, 5] = two_s * (y * z + x * w)


@wp.func
def _smpl_root_quat(
    smpl_pose: wp.array2d(dtype=float), f: wp.int64, y_up: int, y_to_z: wp.vec4
) -> wp.vec4:
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
        q = _qmul(y_to_z, q)
    # conjugate of the SMPL rest orientation (0.5, 0.5, 0.5, 0.5)
    return _qmul(q, wp.vec4(0.5, -0.5, -0.5, -0.5))


@wp.func
def _future_frame(
    motion_ids: wp.array(dtype=wp.int64),
    starts: wp.array(dtype=wp.int64),
    time_steps: wp.array(dtype=wp.int64),
    num_steps: wp.array(dtype=wp.int64),
    length_starts: wp.array(dtype=wp.int64),
    future_init: wp.array(dtype=float),
    e: int,
    k: int,
) -> wp.int64:
    """TrackingCommand.(smpl_)future_time_steps (computed in float, then truncated) at the clip start."""
    step = wp.min(
        future_init[k] + float(time_steps[e] + starts[e]), float(num_steps[e] - wp.int64(1))
    )
    return length_starts[motion_ids[e]] + wp.int64(step)


@wp.kernel
def _smpl_joints_local_kernel(
    motion_ids: wp.array(dtype=wp.int64),
    starts: wp.array(dtype=wp.int64),
    time_steps: wp.array(dtype=wp.int64),
    num_steps: wp.array(dtype=wp.int64),
    length_starts: wp.array(dtype=wp.int64),
    future_init: wp.array(dtype=float),
    smpl_pose: wp.array2d(dtype=float),
    smpl_joints: wp.array3d(dtype=float),
    y_up: int,
    y_to_z: wp.vec4,
    out: wp.array3d(dtype=float),
):
    e, k = wp.tid()
    f = _future_frame(motion_ids, starts, time_steps, num_steps, length_starts, future_init, e, k)
    inv = _qinv(_smpl_root_quat(smpl_pose, f, y_up, y_to_z))
    for j in range(smpl_joints.shape[1]):
        p = _qapply(inv, wp.vec3(smpl_joints[f, j, 0], smpl_joints[f, j, 1], smpl_joints[f, j, 2]))
        out[e, k, 3 * j + 0] = p[0]
        out[e, k, 3 * j + 1] = p[1]
        out[e, k, 3 * j + 2] = p[2]


@wp.kernel
def _smpl_root_ori_b_kernel(
    motion_ids: wp.array(dtype=wp.int64),
    starts: wp.array(dtype=wp.int64),
    time_steps: wp.array(dtype=wp.int64),
    num_steps: wp.array(dtype=wp.int64),
    length_starts: wp.array(dtype=wp.int64),
    future_init: wp.array(dtype=float),
    smpl_pose: wp.array2d(dtype=float),
    y_up: int,
    y_to_z: wp.vec4,
    body_quat: wp.array2d(dtype=wp.quatf),
    robot_anchor: int,
    out: wp.array3d(dtype=float),
):
    e, k = wp.tid()
    f = _future_frame(motion_ids, starts, time_steps, num_steps, length_starts, future_init, e, k)
    q = _smpl_root_quat(smpl_pose, f, y_up, y_to_z)
    _write_rot6d(out, e, k, _qmul(_qinv(_wxyz(body_quat[e, robot_anchor])), q))


@wp.kernel
def _joint_future_kernel(
    motion_ids: wp.array(dtype=wp.int64),
    starts: wp.array(dtype=wp.int64),
    time_steps: wp.array(dtype=wp.int64),
    num_steps: wp.array(dtype=wp.int64),
    length_starts: wp.array(dtype=wp.int64),
    future_init: wp.array(dtype=float),
    dof_pos: wp.array2d(dtype=float),
    dof_vel: wp.array2d(dtype=float),
    joints: wp.array(dtype=int),
    out: wp.array2d(dtype=float),
):
    """cat([joint_pos_multi_future, joint_vel_multi_future], dim=1) over ``joints``: all positions first."""
    e, k = wp.tid()
    f = _future_frame(motion_ids, starts, time_steps, num_steps, length_starts, future_init, e, k)
    nj = joints.shape[0]
    vel_offset = future_init.shape[0] * nj
    for j in range(nj):
        out[e, k * nj + j] = dof_pos[f, joints[j]]
        out[e, vel_offset + k * nj + j] = dof_vel[f, joints[j]]


@wp.kernel
def _root_z_future_kernel(
    motion_ids: wp.array(dtype=wp.int64),
    starts: wp.array(dtype=wp.int64),
    time_steps: wp.array(dtype=wp.int64),
    num_steps: wp.array(dtype=wp.int64),
    length_starts: wp.array(dtype=wp.int64),
    future_init: wp.array(dtype=float),
    body_pos: wp.array3d(dtype=float),
    out: wp.array2d(dtype=float),
):
    """TrackingCommand.root_z_multi_future."""
    e, k = wp.tid()
    f = _future_frame(motion_ids, starts, time_steps, num_steps, length_starts, future_init, e, k)
    out[e, k] = body_pos[f, 0, 2]


@wp.kernel
def _root_ori_b_future_kernel(
    motion_ids: wp.array(dtype=wp.int64),
    starts: wp.array(dtype=wp.int64),
    time_steps: wp.array(dtype=wp.int64),
    num_steps: wp.array(dtype=wp.int64),
    length_starts: wp.array(dtype=wp.int64),
    future_init: wp.array(dtype=float),
    ref_body_quat: wp.array3d(dtype=float),
    body_quat: wp.array2d(dtype=wp.quatf),
    robot_anchor: int,
    out: wp.array3d(dtype=float),
):
    """TrackingCommand.root_rot_dif_l_multi_future."""
    e, k = wp.tid()
    f = _future_frame(motion_ids, starts, time_steps, num_steps, length_starts, future_init, e, k)
    ref = wp.vec4(
        ref_body_quat[f, 0, 0],
        ref_body_quat[f, 0, 1],
        ref_body_quat[f, 0, 2],
        ref_body_quat[f, 0, 3],
    )
    _write_rot6d(out, e, k, _qmul(_qinv(_wxyz(body_quat[e, robot_anchor])), ref))


vec6 = wp.types.vector(length=6, dtype=wp.float32)


@wp.func
def _rot6d(r: wp.vec4) -> vec6:
    """math_wxyz.matrix_from_quat, first two columns, row by row."""
    two_s = 2.0 / wp.dot(r, r)
    w, x, y, z = r[0], r[1], r[2], r[3]
    return vec6(
        1.0 - two_s * (y * y + z * z),
        two_s * (x * y - z * w),
        two_s * (x * y + z * w),
        1.0 - two_s * (x * x + z * z),
        two_s * (x * z - y * w),
        two_s * (y * z + x * w),
    )


@wp.kernel
def _robot_body_b_kernel(
    body_pos: wp.array2d(dtype=wp.vec3f),
    body_quat: wp.array2d(dtype=wp.quatf),
    robot_anchor: int,
    tracked_bodies: wp.array(dtype=int),
    with_ori: int,
    out: wp.array2d(dtype=float),
):
    """robot_body_pos_b / robot_body_ori_b: subtract_frame_transforms from the robot anchor."""
    e = wp.tid()
    inv = _qinv(_wxyz(body_quat[e, robot_anchor]))
    anchor = body_pos[e, robot_anchor]
    for b in range(tracked_bodies.shape[0]):
        k = tracked_bodies[b]
        if with_ori == 0:
            p = _qapply(inv, body_pos[e, k] - anchor)
            for i in range(3):
                out[e, 3 * b + i] = p[i]
        else:
            r = _rot6d(_qmul(inv, _wxyz(body_quat[e, k])))
            for i in range(6):
                out[e, 6 * b + i] = r[i]


@wp.kernel
def _motion_anchor_b_kernel(
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
    with_ori: int,
    repeat: int,
    out: wp.array2d(dtype=float),
):
    """motion_anchor_pos_b / motion_anchor_ori_b: the reference anchor in the robot anchor frame."""
    e = wp.tid()
    f = length_starts[motion_ids[e]] + starts[e] + time_steps[e]
    inv = _qinv(_wxyz(body_quat[e, robot_anchor]))
    if with_ori == 0:
        ref = wp.vec3(
            ref_body_pos[f, motion_anchor, 0] + env_origins[e, 0],
            ref_body_pos[f, motion_anchor, 1] + env_origins[e, 1],
            ref_body_pos[f, motion_anchor, 2] + env_origins[e, 2],
        )
        p = _qapply(inv, ref - body_pos[e, robot_anchor])
        for i in range(3):
            out[e, i] = p[i]
    else:
        ref_q = wp.vec4(
            ref_body_quat[f, motion_anchor, 0],
            ref_body_quat[f, motion_anchor, 1],
            ref_body_quat[f, motion_anchor, 2],
            ref_body_quat[f, motion_anchor, 3],
        )
        r = _rot6d(_qmul(inv, ref_q))
        for k in range(repeat):
            for i in range(6):
                out[e, 6 * k + i] = r[i]


@wp.kernel
def _vr_3point_kernel(
    motion_ids: wp.array(dtype=wp.int64),
    starts: wp.array(dtype=wp.int64),
    time_steps: wp.array(dtype=wp.int64),
    length_starts: wp.array(dtype=wp.int64),
    ref_body_pos: wp.array3d(dtype=float),
    ref_body_quat: wp.array3d(dtype=float),
    env_origins: wp.array2d(dtype=float),
    points: wp.array(dtype=int),
    offsets: wp.array3d(dtype=float),
    motion_anchor: int,
    with_orn: int,
    out: wp.array2d(dtype=float),
):
    """vr_3point_local_target / vr_3point_local_orn_target: the reference VR points (with their offsets) in the
    reference anchor frame."""
    e = wp.tid()
    f = length_starts[motion_ids[e]] + starts[e] + time_steps[e]
    origin = wp.vec3(env_origins[e, 0], env_origins[e, 1], env_origins[e, 2])
    a = motion_anchor
    inv = _qinv(
        wp.vec4(
            ref_body_quat[f, a, 0],
            ref_body_quat[f, a, 1],
            ref_body_quat[f, a, 2],
            ref_body_quat[f, a, 3],
        )
    )
    anchor_pos = (
        wp.vec3(ref_body_pos[f, a, 0], ref_body_pos[f, a, 1], ref_body_pos[f, a, 2]) + origin
    )
    for p in range(points.shape[0]):
        b = points[p]
        q = wp.vec4(
            ref_body_quat[f, b, 0],
            ref_body_quat[f, b, 1],
            ref_body_quat[f, b, 2],
            ref_body_quat[f, b, 3],
        )
        if with_orn == 0:
            offset = wp.vec3(offsets[e, p, 0], offsets[e, p, 1], offsets[e, p, 2])
            pos = wp.vec3(ref_body_pos[f, b, 0], ref_body_pos[f, b, 1], ref_body_pos[f, b, 2])
            pos = pos + _qapply(q, offset) + origin
            v = _qapply(inv, pos - anchor_pos)
            for i in range(3):
                out[e, 3 * p + i] = v[i]
        else:
            r = _qmul(inv, q)
            for i in range(4):
                out[e, 4 * p + i] = r[i]


def _ref(t: torch.Tensor, dtype=None):
    return wp.from_torch(t.contiguous() if dtype is None else t.to(dtype).contiguous())


class _WarpTerm:
    """An observation term evaluated by one Warp kernel, per env or per (env, future frame); ``check``
    compares it with the torch term on every call."""

    def __init__(self, env, stock, params, kind, check):
        self.stock, self.kind, self.check = stock, kind, check
        self.command = command = env.command_manager.get_term(params["command_name"])
        lib, dev, n = command.motion_lib, command.device, command.num_envs
        smpl = kind.startswith("smpl")
        init = command.smpl_future_time_steps_init if smpl else command.future_time_steps_init
        self.future_init = init[0].to(torch.float32).contiguous()
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
            self.joints = wp.array([int(j) for j in joints], dtype=int, device=dev)
            self.out = torch.zeros((n, 2 * frames * len(joints)), device=dev)
            self.result = self.out if flat else self.out.view(n, frames, -1)
        elif kind == "root_z_future":
            self.out = torch.zeros((n, frames), device=dev)
            self.result = self.out if flat else self.out.view(n, frames, -1)
        elif kind in ("robot_body_pos_b", "robot_body_ori_b"):
            self.tracked = wp.array(command.body_indexes.tolist(), dtype=int, device=dev)
            width = 3 if kind == "robot_body_pos_b" else 6
            self.out = self.result = torch.zeros((n, width * len(command.body_indexes)), device=dev)
        elif kind == "motion_anchor_pos_b":
            self.out = self.result = torch.zeros((n, 3), device=dev)
        elif kind == "motion_anchor_ori_b":
            self.repeat = 1 if flat else command.num_future_frames
            self.out = torch.zeros((n, 6 * self.repeat), device=dev)
            self.result = self.out if flat else self.out.view(n, self.repeat, 6)
        elif kind in ("vr_3point_local_target", "vr_3point_local_orn_target"):
            self.points = wp.array(
                command._vr_3point_idx_motion.tolist(), dtype=int, device=dev
            )  # noqa: SLF001
            width = 3 if kind == "vr_3point_local_target" else 4
            self.out = self.result = torch.zeros(
                (n, width * len(command.cfg.vr_3point_body)), device=dev
            )
        self.origins = wp.from_torch(env.scene.env_origins.contiguous())
        # the constant of TrackingCommand.smpl_root_ytoz_up, computed the same way
        y_to_z = torch_transform.angle_axis_to_quaternion(
            torch.tensor([[math.pi / 2, 0.0, 0.0]], device=dev)
        )
        self.y_to_z = wp.vec4(*y_to_z[0].tolist())
        self.steps, self.worst = 0, torch.zeros((), device=dev)

    def __call__(self, env, **params):
        command = self.command
        lib, data = command.motion_lib, command.robot.data
        inputs = [
            _ref(command.motion_ids, torch.int64),
            _ref(command.motion_start_time_steps, torch.int64),
            _ref(command.time_steps, torch.int64),
            _ref(command.motion_num_steps, torch.int64),
            _ref(lib.length_starts, torch.int64),
            wp.from_torch(self.future_init),
        ]
        kind = self.kind
        dim = (command.num_envs, self.future_init.shape[0])
        y_up = int(bool(getattr(lib, "smpl_y_up", False)))
        if kind in ("robot_body_pos_b", "robot_body_ori_b"):
            kernel, dim = _robot_body_b_kernel, command.num_envs
            inputs = [
                data.body_pos_w.warp,
                data.body_quat_w.warp,
                command.robot_anchor_body_index,
                self.tracked,
                int(kind == "robot_body_ori_b"),
            ]
        elif kind in ("motion_anchor_pos_b", "motion_anchor_ori_b"):
            kernel, dim = _motion_anchor_b_kernel, command.num_envs
            inputs = inputs[:3] + inputs[4:5]
            inputs += [
                _ref(lib.body_pos_w),
                _ref(lib.body_quat_w),
                self.origins,
                data.body_pos_w.warp,
                data.body_quat_w.warp,
                command.robot_anchor_body_index,
                command.motion_anchor_body_index,
                int(kind == "motion_anchor_ori_b"),
                getattr(self, "repeat", 1),
            ]
        elif kind in ("vr_3point_local_target", "vr_3point_local_orn_target"):
            kernel, dim = _vr_3point_kernel, command.num_envs
            inputs = inputs[:3] + inputs[4:5]
            inputs += [
                _ref(lib.body_pos_w),
                _ref(lib.body_quat_w),
                self.origins,
                self.points,
                _ref(command.vr_3point_body_offsets),
                command.motion_anchor_body_index,
                int(kind == "vr_3point_local_orn_target"),
            ]
        elif kind == "smpl_joints_local":
            kernel = _smpl_joints_local_kernel
            smpl_poses, smpl_joints = (
                lib._motion_smpl_poses,
                lib._motion_smpl_joints,
            )  # noqa: SLF001
            inputs += [_ref(smpl_poses), _ref(smpl_joints), y_up, self.y_to_z]
        elif kind == "smpl_root_ori_b":
            kernel = _smpl_root_ori_b_kernel
            inputs += [
                _ref(lib._motion_smpl_poses),  # noqa: SLF001
                y_up,
                self.y_to_z,
                data.body_quat_w.warp,
                command.robot_anchor_body_index,
            ]
        elif kind in ("joint_future", "joint_future_lower"):
            kernel = _joint_future_kernel
            inputs += [_ref(lib.dof_pos), _ref(lib.dof_vel), self.joints]
        elif kind == "root_z_future":
            kernel = _root_z_future_kernel
            inputs += [_ref(lib.body_pos_w)]
        else:
            kernel = _root_ori_b_future_kernel
            inputs += [
                _ref(lib.body_quat_w),
                data.body_quat_w.warp,
                command.robot_anchor_body_index,
            ]
        wp.launch(
            kernel,
            dim=dim,
            inputs=inputs,
            outputs=[wp.from_torch(self.out)],
            device=command.device,
            # the current torch stream, also when sync_free records the step as a CUDA graph
            stream=wp.stream_from_torch(torch.cuda.current_stream(command.device)),
        )
        if self.check:
            stock = self.stock(env, **params)
            self.worst = torch.maximum(self.worst, (self.result - stock).abs().max())
            self.steps += 1
            if self.steps % 24 == 0:
                print(  # noqa: T201
                    f"[fused_obs] {kind}: max abs diff over {self.steps} steps {self.worst.item():.2e}"
                )
        return self.result


# term function -> (kind, accepted params, whether non_flatten=True is required)
_SUPPORTED = {
    mdp.smpl_joints_multi_future_local: (
        "smpl_joints_local",
        {"command_name", "non_flatten"},
        True,
    ),
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
            params = {k: v for k, v in term_cfg.params.items() if v is not None}
            if set(params) - accepted or (needs_non_flatten and not params.get("non_flatten")):
                continue
            term_cfg.func = _WarpTerm(env, term_cfg.func, term_cfg.params, kind, check)
            swapped.append(f"{group}/{name}")
    print(f"[fused_obs] Warp terms: {swapped}")  # noqa: T201
    return swapped
