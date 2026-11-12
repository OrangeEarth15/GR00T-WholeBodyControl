"""Fused Warp evaluation of the tracking reward terms.

The sonic_release reward set (rewards/tracking/base_5point_local_feet_acc) is twelve small terms.
Evaluated one by one in torch they cost about 200 kernel launches per control step.
FusedRewardManager computes all twelve raw values in one Warp kernel, reading the reference motion at the
current frame from the motion library and the robot state from the Isaac Lab Warp buffers, then applies
weights and episode sums exactly like isaaclab.managers.RewardManager.compute. If the configured terms are
not all supported (another function, or a parameter the kernel does not model) it keeps the stock per-term
path. Weights and parameters changed at run time (a schedule) are picked up on the next step.

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
    quat_error,
    ref,
    wxyz,
)

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


@wp.kernel
def _tracking_rewards_kernel(
    # reference motion: the motion library at the command's frame, quaternions (w, x, y, z)
    c: Cursor,
    lib_body_pos: wp.array3d(dtype=float),
    lib_body_quat: wp.array3d(dtype=float),
    lib_body_lin_vel: wp.array3d(dtype=float),
    lib_body_ang_vel: wp.array3d(dtype=float),
    env_origins: wp.array2d(dtype=float),
    body_pos_relative: wp.array3d(dtype=float),
    body_quat_relative: wp.array3d(dtype=float),
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
    # body / joint / sensor indices: robot order unless named *_motion
    robot_anchor: int,
    motion_anchor: int,
    tracked_bodies: wp.array(dtype=wp.int32),
    shake_bodies: wp.array(dtype=wp.int32),
    point_bodies: wp.array(dtype=wp.int32),
    point_bodies_motion: wp.array(dtype=wp.int32),
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
    f = frame(c, e)
    ee = wp.int64(e)
    origin = env3(env_origins, e)
    robot_anchor_pos = body_pos[e, robot_anchor]
    robot_anchor_quat = wxyz(body_quat[e, robot_anchor])
    ref_anchor_pos = lib3(lib_body_pos, f, motion_anchor) + origin
    ref_anchor_quat = lib4(lib_body_quat, f, motion_anchor)

    col = columns[ANCHOR_POS]
    if col >= 0:
        da = ref_anchor_pos - robot_anchor_pos
        raw[e, col] = wp.exp(-wp.dot(da, da) / variance[ANCHOR_POS])

    col = columns[ANCHOR_ORI]
    if col >= 0:
        err = quat_error(ref_anchor_quat, robot_anchor_quat)
        raw[e, col] = wp.exp(-err * err / variance[ANCHOR_ORI])

    # tracked bodies: position / orientation (anchor-relative reference), linear / angular velocity
    num_tracked = tracked_bodies.shape[0]
    pos_sq = float(0.0)
    ori_sq = float(0.0)
    lin_sq = float(0.0)
    ang_sq = float(0.0)
    for i in range(num_tracked):
        b = tracked_bodies[i]
        dp = lib3(body_pos_relative, ee, i) - body_pos[e, b]
        pos_sq += wp.dot(dp, dp)
        err = quat_error(lib4(body_quat_relative, ee, i), wxyz(body_quat[e, b]))
        ori_sq += err * err
        dv = lib3(lib_body_lin_vel, f, i) - body_lin_vel[e, b]
        lin_sq += wp.dot(dv, dv)
        dw = lib3(lib_body_ang_vel, f, i) - body_ang_vel[e, b]
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
            cb = contact_bodies[k]
            peak = float(0.0)
            for h in range(contact_forces.shape[1]):
                peak = wp.max(peak, wp.length(contact_forces[e, h, cb]))
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
        ref_inv = qinv(ref_anchor_quat)
        robot_inv = qinv(robot_anchor_quat)
        s = float(0.0)
        for k in range(point_bodies.shape[0]):
            b = point_bodies[k]
            m = point_bodies_motion[k]
            # TrackingCommand.reward_point_body_pos_w, same operation order
            ref_point = lib3(lib_body_pos, f, m) + qapply(lib4(lib_body_quat, f, m), point_offsets[k])
            ref_local = qapply(ref_inv, ref_point + origin - ref_anchor_pos)
            robot_point = body_pos[e, b] + qapply(wxyz(body_quat[e, b]), point_offsets[k])
            robot_local = qapply(robot_inv, robot_point - robot_anchor_pos)
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


def term_signature(term_cfgs) -> list:
    """The run-time adjustable part of the term configs: weights and scalar parameters."""
    return [
        (
            getattr(c, "weight", None),
            {k: v for k, v in c.params.items() if isinstance(v, (bool, int, float, str))},
        )
        for c in term_cfgs
    ]


class FusedRewardManager(RewardManager):
    """RewardManager whose compute() evaluates the supported tracking terms in one Warp kernel."""

    def __init__(self, cfg: object, env, check: bool = False):
        super().__init__(cfg, env)
        self._check = check
        self._check_steps = 0
        self._check_max = torch.zeros(len(self._term_names), device=self.device)
        self._prepare_fused()
        if self._unsupported:
            print(f"[FusedRewardManager] stock per-term path: {self._unsupported}")  # noqa: T201
        else:
            print(f"[FusedRewardManager] {len(self._term_names)} terms in one Warp kernel")  # noqa: T201

    def _prepare_fused(self):
        """Map the configured terms onto kernel slots; sets ``_unsupported`` to why not, or None."""
        self._signature = term_signature(self._term_cfgs)
        self._unsupported = self._map_terms()

    def _map_terms(self) -> str | None:
        dev = self.device
        columns = [-1] * NUM_KINDS
        variance = [1.0] * NUM_KINDS
        command_names = set()
        self._contact_threshold, self._shake_threshold = 1.0, 1.5
        self._sensor_name = None
        contact_ids = []
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
                    contact_ids = range(self._env.scene.sensors[self._sensor_name].num_sensors)
            elif kind == ANTI_SHAKE:
                self._shake_threshold = float(params.get("threshold", 1.5))
                shake_names = params.get("body_names")
            elif kind == JOINT_LIMIT:
                limit_cfg = params["asset_cfg"]
            elif kind == JOINT_ACC:
                acc_cfg = params["asset_cfg"]
            # zero-weight terms stay 0 in the raw buffer, like the stock path's skip
            columns[kind] = col if term_cfg.weight != 0.0 else -1
        if len(command_names) != 1:
            return f"commands {command_names}"
        command = self._env.command_manager.get_term(command_names.pop())
        if command.use_ref_motion_root_quat_w_as_anchor or getattr(command, "_offline", False):
            return "the command reads the anchor from the reference motion"
        for cfg in (limit_cfg, acc_cfg):
            if cfg is not None and cfg.name != "robot":
                return f"joint term on asset '{cfg.name}'"

        def joint_ids(cfg):
            if cfg is None or isinstance(cfg.joint_ids, slice):
                return range(command.robot.num_joints)
            return cfg.joint_ids

        self._command = command
        tracked = command.body_indexes
        points = columns[POINTS_LOCAL] >= 0
        self._columns = int_array(columns, dev)
        self._variance = wp.array(variance, dtype=float, device=dev)
        self._tracked_bodies = int_array(tracked.tolist(), dev)
        self._shake_bodies = int_array(tracked[_get_body_indexes(command, shake_names)].tolist(), dev)
        self._point_bodies = int_array(command.reward_point_body_indices if points else [], dev)
        self._point_bodies_motion = int_array(
            command.reward_point_body_indices_motion if points else [], dev
        )
        self._point_offsets = wp.array(
            command.reward_point_body_offsets[0].cpu().numpy() if points else [],
            dtype=wp.vec3f,
            device=dev,
        )
        self._limit_joints = int_array(joint_ids(limit_cfg), dev)
        self._acc_joints = int_array(joint_ids(acc_cfg), dev)
        self._contact_bodies = int_array(contact_ids, dev)
        self._no_contacts = wp.zeros((self.num_envs, 1, 1), dtype=wp.vec3f, device=dev)
        self._weights = torch.tensor([float(c.weight) for c in self._term_cfgs], device=dev)
        self._raw = torch.zeros((self.num_envs, len(self._term_names)), device=dev)
        self._raw_wp = wp.from_torch(self._raw)
        return None

    def compute(self, dt: float) -> torch.Tensor:
        if term_signature(self._term_cfgs) != self._signature:
            self._prepare_fused()
        if self._unsupported:
            return super().compute(dt)
        command = self._command
        lib, data = command.motion_lib, command.robot.data
        if self._sensor_name is not None:
            contact = self._env.scene.sensors[self._sensor_name].data.net_normal_forces_w_history.warp
        else:
            contact = self._no_contacts
        action_manager = self._env.action_manager
        wp.launch(
            _tracking_rewards_kernel,
            dim=self.num_envs,
            inputs=[
                cursor(command),
                ref(lib.body_pos_w),
                ref(lib.body_quat_w),
                ref(lib.body_lin_vel_w),
                ref(lib.body_ang_vel_w),
                ref(self._env.scene.env_origins),
                ref(command.body_pos_relative_w),
                ref(command.body_quat_relative_w),
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
                command.motion_anchor_body_index,
                self._tracked_bodies,
                self._shake_bodies,
                self._point_bodies,
                self._point_bodies_motion,
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
            worst = ", ".join(f"{n}={v:.2e}" for n, v in zip(self._term_names, self._check_max.tolist()))
            print(f"[FusedRewardManager] parity over {self._check_steps} steps, max rel err: {worst}")  # noqa: T201
