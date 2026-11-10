"""Fused Warp evaluation of the tracking reward terms (Direct-Warp step A').

The sonic_release reward set (rewards/tracking/base_5point_local_feet_acc) is twelve small terms.
Evaluated one by one in torch they cost about 200 kernel launches per control step.
FusedRewardManager computes all twelve raw values in one Warp kernel, reading robot state straight
from the Isaac Lab Warp buffers, then applies weights and episode sums exactly like
isaaclab.managers.RewardManager.compute. If the configured terms are not all supported (another
function, or a parameter the kernel does not model) it keeps the stock per-term path.

Enable with ++manager_env.config.fused_rewards=True. With
++manager_env.config.fused_rewards_check=True the stock terms are also evaluated every step and the
largest per-term difference is printed every 24 steps.
"""

from __future__ import annotations

from isaaclab.managers import RewardManager
import torch
import warp as wp

from gear_sonic.envs.manager_env import mdp
from gear_sonic.envs.manager_env.mdp.commands import _get_body_indexes

# Term kinds, in the order the kernel evaluates them.
(
    ANCHOR_POS,
    ANCHOR_ORI,
    BODY_POS,
    BODY_ORI,
    BODY_LIN_VEL,
    BODY_ANG_VEL,
    ACTION_RATE,
    JOINT_LIMIT,
    UNDESIRED_CONTACTS,
    ANTI_SHAKE,
    POINTS_LOCAL,
    JOINT_ACC,
) = range(12)
NUM_KINDS = 12

_KIND_OF_FUNC = {
    mdp.tracking_anchor_pos_error: ANCHOR_POS,
    mdp.tracking_anchor_ori_error: ANCHOR_ORI,
    mdp.tracking_relative_body_pos_error: BODY_POS,
    mdp.tracking_relative_body_ori_error: BODY_ORI,
    mdp.tracking_body_linvel_error: BODY_LIN_VEL,
    mdp.tracking_body_angvel_error: BODY_ANG_VEL,
    mdp.action_rate_l2: ACTION_RATE,
    mdp.joint_pos_limits: JOINT_LIMIT,
    mdp.undesired_contacts: UNDESIRED_CONTACTS,
    mdp.anti_shake_ang_vel_l2: ANTI_SHAKE,
    mdp.tracking_local_vr_5point_error: POINTS_LOCAL,
    mdp.joint_acc_l2: JOINT_ACC,
}
_GAUSSIAN_KINDS = (
    ANCHOR_POS,
    ANCHOR_ORI,
    BODY_POS,
    BODY_ORI,
    BODY_LIN_VEL,
    BODY_ANG_VEL,
    POINTS_LOCAL,
)


@wp.func
def _quat_mul(a: wp.vec4, b: wp.vec4) -> wp.vec4:
    """Hamilton product of (w, x, y, z) quaternions."""
    return wp.vec4(
        a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3],
        a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2],
        a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1],
        a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0],
    )


@wp.func
def _quat_error(q1: wp.vec4, q2: wp.vec4) -> float:
    """math_wxyz.quat_error_magnitude: norm of the axis-angle of q1 * conj(q2)."""
    d = _quat_mul(q1, wp.vec4(q2[0], -q2[1], -q2[2], -q2[3]))
    if d[0] < 0.0:
        d = -d
    mag = wp.sqrt(d[1] * d[1] + d[2] * d[2] + d[3] * d[3])
    half = wp.atan2(mag, d[0])
    angle = 2.0 * half
    s = 0.5 - angle * angle / 48.0
    if wp.abs(angle) > 1.0e-6:
        s = wp.sin(half) / angle
    return mag / s


@wp.func
def _quat_apply(q: wp.vec4, v: wp.vec3) -> wp.vec3:
    xyz = wp.vec3(q[1], q[2], q[3])
    t = 2.0 * wp.cross(xyz, v)
    return v + q[0] * t + wp.cross(xyz, t)


@wp.func
def _quat_inv(q: wp.vec4) -> wp.vec4:
    return wp.vec4(q[0], -q[1], -q[2], -q[3]) / wp.max(wp.dot(q, q), 1.0e-9)


@wp.func
def _wxyz(q: wp.quatf) -> wp.vec4:
    """Isaac Lab Warp buffers hold (x, y, z, w)."""
    return wp.vec4(q[3], q[0], q[1], q[2])


@wp.func
def _vec3(a: wp.array3d(dtype=float), e: int, i: int) -> wp.vec3:
    return wp.vec3(a[e, i, 0], a[e, i, 1], a[e, i, 2])


@wp.func
def _vec4(a: wp.array3d(dtype=float), e: int, i: int) -> wp.vec4:
    return wp.vec4(a[e, i, 0], a[e, i, 1], a[e, i, 2], a[e, i, 3])


@wp.kernel
def _tracking_rewards_kernel(
    # reference motion, torch tensors, quaternions (w, x, y, z)
    ref_anchor_pos: wp.array2d(dtype=float),
    ref_anchor_quat: wp.array2d(dtype=float),
    ref_body_pos: wp.array3d(dtype=float),
    ref_body_quat: wp.array3d(dtype=float),
    ref_body_lin_vel: wp.array3d(dtype=float),
    ref_body_ang_vel: wp.array3d(dtype=float),
    ref_points: wp.array3d(dtype=float),
    # robot state, Isaac Lab Warp buffers
    body_pos: wp.array2d(dtype=wp.vec3f),
    body_quat: wp.array2d(dtype=wp.quatf),
    body_lin_vel: wp.array2d(dtype=wp.vec3f),
    body_ang_vel: wp.array2d(dtype=wp.vec3f),
    joint_pos: wp.array2d(dtype=float),
    joint_limits: wp.array2d(dtype=wp.vec2f),
    joint_acc: wp.array2d(dtype=float),
    contact_forces: wp.array3d(dtype=wp.vec3f),
    action: wp.array2d(dtype=float),
    prev_action: wp.array2d(dtype=float),
    # robot body / joint / sensor indices
    anchor_body: int,
    tracked_bodies: wp.array(dtype=wp.int32),
    shake_bodies: wp.array(dtype=wp.int32),
    point_bodies: wp.array(dtype=wp.int32),
    point_offsets: wp.array(dtype=wp.vec3f),
    limit_joints: wp.array(dtype=wp.int32),
    acc_joints: wp.array(dtype=wp.int32),
    contact_bodies: wp.array(dtype=wp.int32),
    # term parameters
    variance: wp.array(dtype=float),
    contact_threshold: float,
    shake_threshold: float,
    columns: wp.array(dtype=wp.int32),
    # unweighted term values, (num_envs, num_terms)
    raw: wp.array2d(dtype=float),
):
    e = wp.tid()
    robot_anchor_pos = body_pos[e, anchor_body]
    robot_anchor_quat = _wxyz(body_quat[e, anchor_body])
    ref_anchor_p = wp.vec3(ref_anchor_pos[e, 0], ref_anchor_pos[e, 1], ref_anchor_pos[e, 2])
    ref_anchor_q = wp.vec4(
        ref_anchor_quat[e, 0], ref_anchor_quat[e, 1], ref_anchor_quat[e, 2], ref_anchor_quat[e, 3]
    )

    col = columns[ANCHOR_POS]
    if col >= 0:
        da = ref_anchor_p - robot_anchor_pos
        raw[e, col] = wp.exp(-wp.dot(da, da) / variance[ANCHOR_POS])

    col = columns[ANCHOR_ORI]
    if col >= 0:
        err = _quat_error(ref_anchor_q, robot_anchor_quat)
        raw[e, col] = wp.exp(-err * err / variance[ANCHOR_ORI])

    # tracked bodies: position / orientation (anchor-relative reference), linear / angular velocity
    num_tracked = tracked_bodies.shape[0]
    pos_sq = float(0.0)
    ori_sq = float(0.0)
    lin_sq = float(0.0)
    ang_sq = float(0.0)
    for i in range(num_tracked):
        b = tracked_bodies[i]
        dp = _vec3(ref_body_pos, e, i) - body_pos[e, b]
        pos_sq += wp.dot(dp, dp)
        err = _quat_error(_vec4(ref_body_quat, e, i), _wxyz(body_quat[e, b]))
        ori_sq += err * err
        dv = _vec3(ref_body_lin_vel, e, i) - body_lin_vel[e, b]
        lin_sq += wp.dot(dv, dv)
        dw = _vec3(ref_body_ang_vel, e, i) - body_ang_vel[e, b]
        ang_sq += wp.dot(dw, dw)
    n = float(num_tracked)
    col = columns[BODY_POS]
    if col >= 0:
        raw[e, col] = wp.exp(-(pos_sq / n) / variance[BODY_POS])
    col = columns[BODY_ORI]
    if col >= 0:
        raw[e, col] = wp.exp(-(ori_sq / n) / variance[BODY_ORI])
    col = columns[BODY_LIN_VEL]
    if col >= 0:
        raw[e, col] = wp.exp(-(lin_sq / n) / variance[BODY_LIN_VEL])
    col = columns[BODY_ANG_VEL]
    if col >= 0:
        raw[e, col] = wp.exp(-(ang_sq / n) / variance[BODY_ANG_VEL])

    col = columns[ACTION_RATE]
    if col >= 0:
        s = float(0.0)
        for j in range(action.shape[1]):
            dj = action[e, j] - prev_action[e, j]
            s += dj * dj
        raw[e, col] = s

    col = columns[JOINT_LIMIT]
    if col >= 0:
        s = float(0.0)
        for k in range(limit_joints.shape[0]):
            j = limit_joints[k]
            q = joint_pos[e, j]
            lim = joint_limits[e, j]
            s += -wp.min(q - lim[0], 0.0) + wp.max(q - lim[1], 0.0)
        raw[e, col] = s

    col = columns[UNDESIRED_CONTACTS]
    if col >= 0:
        count = float(0.0)
        for k in range(contact_bodies.shape[0]):
            c = contact_bodies[k]
            peak = float(0.0)
            for h in range(contact_forces.shape[1]):
                peak = wp.max(peak, wp.length(contact_forces[e, h, c]))
            if peak > contact_threshold:
                count += 1.0
        raw[e, col] = count

    col = columns[ANTI_SHAKE]
    if col >= 0:
        s = float(0.0)
        for k in range(shake_bodies.shape[0]):
            excess = wp.max(wp.length(body_ang_vel[e, shake_bodies[k]]) - shake_threshold, 0.0)
            s += excess * excess
        raw[e, col] = s / float(shake_bodies.shape[0])

    col = columns[POINTS_LOCAL]
    if col >= 0:
        ref_inv = _quat_inv(ref_anchor_q)
        robot_inv = _quat_inv(robot_anchor_quat)
        s = float(0.0)
        for k in range(point_bodies.shape[0]):
            b = point_bodies[k]
            robot_point = body_pos[e, b] + _quat_apply(_wxyz(body_quat[e, b]), point_offsets[k])
            ref_local = _quat_apply(ref_inv, _vec3(ref_points, e, k) - ref_anchor_p)
            robot_local = _quat_apply(robot_inv, robot_point - robot_anchor_pos)
            dl = robot_local - ref_local
            s += wp.dot(dl, dl)
        raw[e, col] = wp.exp(-(s / float(point_bodies.shape[0])) / variance[POINTS_LOCAL])

    col = columns[JOINT_ACC]
    if col >= 0:
        s = float(0.0)
        for k in range(acc_joints.shape[0]):
            a = joint_acc[e, acc_joints[k]]
            s += a * a
        raw[e, col] = s


def _int_array(values, device) -> wp.array:
    return wp.array([int(v) for v in values], dtype=wp.int32, device=device)


class FusedRewardManager(RewardManager):
    """RewardManager whose compute() evaluates the supported tracking terms in one Warp kernel."""

    def __init__(self, cfg: object, env, check: bool = False):
        super().__init__(cfg, env)
        self._check = check
        self._check_steps = 0
        self._check_max = torch.zeros(len(self._term_names), device=self.device)
        self._unsupported = self._prepare_fused()
        if self._unsupported:
            print(f"[FusedRewardManager] stock per-term path: {self._unsupported}")  # noqa: T201
        else:
            print(
                f"[FusedRewardManager] {len(self._term_names)} terms in one Warp kernel"
            )  # noqa: T201

    def _prepare_fused(self) -> str | None:
        """Map the configured terms onto kernel slots; returns why not, or None when all are supported."""
        dev = self.device
        columns = [-1] * NUM_KINDS
        variance = [1.0] * NUM_KINDS
        command_names = set()
        self._contact_threshold, self._shake_threshold = 1.0, 1.5
        self._sensor_name = None
        limit_cfg = acc_cfg = None
        shake_names = None
        for col, (name, term_cfg) in enumerate(zip(self._term_names, self._term_cfgs)):
            kind = _KIND_OF_FUNC.get(term_cfg.func)
            if kind is None or columns[kind] != -1:
                return f"term '{name}' ({getattr(term_cfg.func, '__name__', term_cfg.func)})"
            params = term_cfg.params
            if (
                kind in (BODY_POS, BODY_ORI, BODY_LIN_VEL, BODY_ANG_VEL)
                and params.get("body_names") is not None
            ):
                return f"term '{name}' uses a body subset"
            if "command_name" in params:
                command_names.add(params["command_name"])
            if kind in _GAUSSIAN_KINDS:
                variance[kind] = float(params["std"]) ** 2
            elif kind == UNDESIRED_CONTACTS:
                self._contact_threshold = float(params["threshold"])
                self._sensor_name = params["sensor_cfg"].name
                contact_ids = params["sensor_cfg"].body_ids
                if isinstance(contact_ids, slice):
                    contact_ids = range(self._env.scene.sensors[self._sensor_name].num_bodies)
            elif kind == ANTI_SHAKE:
                self._shake_threshold = float(params.get("threshold", 1.5))
                shake_names = params.get("body_names")
            elif kind == JOINT_LIMIT:
                limit_cfg = params["asset_cfg"]
            elif kind == JOINT_ACC:
                acc_cfg = params["asset_cfg"]
            # zero-weight terms stay 0 in the raw buffer, like the stock path's skip
            columns[kind] = col if term_cfg.weight != 0.0 else -1
        if len(command_names) > 1:
            return f"several commands {command_names}"
        self._command = (
            self._env.command_manager.get_term(command_names.pop()) if command_names else None
        )
        robot = self._command.robot if self._command is not None else self._env.scene["robot"]
        self._robot = robot
        for cfg in (limit_cfg, acc_cfg):
            if cfg is not None and cfg.name != "robot":
                return f"joint term on asset '{cfg.name}'"

        def joint_ids(cfg):
            if cfg is None or isinstance(cfg.joint_ids, slice):
                return range(robot.num_joints)
            return cfg.joint_ids

        tracked = self._command.body_indexes if self._command is not None else torch.zeros(0)
        self._columns = _int_array(columns, dev)
        self._variance = wp.array(variance, dtype=float, device=dev)
        self._tracked_bodies = _int_array(tracked.tolist(), dev)
        shake = _get_body_indexes(self._command, shake_names) if self._command is not None else []
        self._shake_bodies = _int_array([int(tracked[i]) for i in shake], dev)
        self._point_bodies = _int_array(
            self._command.reward_point_body_indices if self._command else [], dev
        )
        offsets = self._command.reward_point_body_offsets[0] if self._command else torch.zeros(0, 3)
        self._point_offsets = wp.array(offsets.cpu().numpy(), dtype=wp.vec3f, device=dev)
        self._limit_joints = _int_array(joint_ids(limit_cfg), dev)
        self._acc_joints = _int_array(joint_ids(acc_cfg), dev)
        self._contact_bodies = _int_array(
            contact_ids if columns[UNDESIRED_CONTACTS] >= 0 else [], dev
        )
        self._weights = torch.tensor([float(c.weight) for c in self._term_cfgs], device=dev)
        self._raw = torch.zeros((self.num_envs, len(self._term_names)), device=dev)
        self._raw_wp = wp.from_torch(self._raw)
        self._empty3 = wp.zeros((self.num_envs, 1, 1), dtype=float, device=dev)
        return None

    def compute(self, dt: float) -> torch.Tensor:
        command = self._command
        if (
            self._unsupported
            or command is None
            or command.use_ref_motion_root_quat_w_as_anchor
            or getattr(command, "_offline", False)
        ):
            return super().compute(dt)

        def ref(t):
            return wp.from_torch(t.contiguous())

        data = self._robot.data
        if self._sensor_name is not None:
            contact = self._env.scene.sensors[
                self._sensor_name
            ].data.net_normal_forces_w_history.warp
        else:
            contact = wp.zeros((self.num_envs, 1, 1), dtype=wp.vec3f, device=self.device)
        action_manager = self._env.action_manager
        wp.launch(
            _tracking_rewards_kernel,
            dim=self.num_envs,
            inputs=[
                ref(command.anchor_pos_w),
                ref(command.anchor_quat_w),
                ref(command.body_pos_relative_w),
                ref(command.body_quat_relative_w),
                ref(command.body_lin_vel_w),
                ref(command.body_ang_vel_w),
                (
                    ref(command.reward_point_body_pos_w)
                    if self._point_bodies.shape[0]
                    else self._empty3
                ),
                data.body_pos_w.warp,
                data.body_quat_w.warp,
                data.body_lin_vel_w.warp,
                data.body_ang_vel_w.warp,
                data.joint_pos.warp,
                data.soft_joint_pos_limits.warp,
                data.joint_acc.warp,
                contact,
                ref(action_manager.action),
                ref(action_manager.prev_action),
                command.robot_anchor_body_index,
                self._tracked_bodies,
                self._shake_bodies,
                self._point_bodies,
                self._point_offsets,
                self._limit_joints,
                self._acc_joints,
                self._contact_bodies,
                self._variance,
                self._contact_threshold,
                self._shake_threshold,
                self._columns,
            ],
            outputs=[self._raw_wp],
            device=self.device,
        )
        if self._check:
            self._compare_with_stock()
        # same bookkeeping as RewardManager.compute
        torch.mul(self._raw, self._weights, out=self._step_reward)
        torch.sum(self._step_reward, dim=1, out=self._reward_buf).mul_(dt)
        self._episode_sum_buf.add_(self._step_reward, alpha=dt)
        return self._reward_buf

    def _compare_with_stock(self):
        """Parity: largest |fused - stock| per term, relative to max(1, |stock|), printed every 24 steps."""
        for col, term_cfg in enumerate(self._term_cfgs):
            if term_cfg.weight == 0.0:
                continue
            stock = term_cfg.func(self._env, **term_cfg.params).to(torch.float32)
            err = ((self._raw[:, col] - stock).abs() / stock.abs().clamp(min=1.0)).max()
            self._check_max[col] = torch.maximum(self._check_max[col], err)
        self._check_steps += 1
        if self._check_steps % 24 == 0:
            worst = ", ".join(
                f"{n}={v:.2e}" for n, v in zip(self._term_names, self._check_max.tolist())
            )
            print(
                f"[FusedRewardManager] parity over {self._check_steps} steps, max rel err: {worst}"
            )  # noqa: T201
