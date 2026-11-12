"""Warp helpers shared by the fused term kernels (fused_rewards, fused_terminations, fused_command, fused_obs).

Quaternions are (w, x, y, z), as in gear_sonic.isaac_utils.math_wxyz; Isaac Lab's Warp buffers hold (x, y, z, w)
and are read through ``wxyz``. ``qmul`` keeps the operation order of math_wxyz.quat_mul, so the kernels round
like the torch terms. ``Cursor`` carries the tracking command's frame indices in their own dtypes, so handing
them to a kernel copies nothing.
"""

from __future__ import annotations

import torch
import warp as wp

vec6 = wp.types.vector(length=6, dtype=wp.float32)


@wp.struct
class Cursor:
    """The reference frame of env e is ``length_starts[motion_ids[e]] + starts[e] + time_steps[e]``."""

    motion_ids: wp.array(dtype=wp.int64)
    starts: wp.array(dtype=wp.int32)
    time_steps: wp.array(dtype=wp.int64)
    num_steps: wp.array(dtype=wp.int32)
    length_starts: wp.array(dtype=wp.int64)
    num_frames: wp.array(dtype=wp.int64)


def cursor(command) -> Cursor:
    """The command's current frame indices as Warp views of its tensors."""
    lib = command.motion_lib
    c = Cursor()
    c.motion_ids = wp.from_torch(command.motion_ids)
    c.starts = wp.from_torch(command.motion_start_time_steps)
    c.time_steps = wp.from_torch(command.time_steps)
    c.num_steps = wp.from_torch(command.motion_num_steps)
    c.length_starts = wp.from_torch(lib.length_starts)
    c.num_frames = wp.from_torch(lib._motion_num_frames)  # noqa: SLF001
    return c


def ref(t: torch.Tensor) -> wp.array:
    return wp.from_torch(t.contiguous())


def int_array(values, device) -> wp.array:
    """An index list or tensor as a Warp int32 array (one host copy for a tensor)."""
    if isinstance(values, torch.Tensor):
        values = values.tolist()
    return wp.array([int(v) for v in values], dtype=wp.int32, device=device)


@wp.func
def frame(c: Cursor, e: int) -> wp.int64:
    return c.length_starts[c.motion_ids[e]] + wp.int64(c.starts[e]) + c.time_steps[e]


@wp.func
def future_frame(c: Cursor, future_init: wp.array(dtype=float), e: int, k: int) -> wp.int64:
    """TrackingCommand.(smpl_)future_time_steps: computed in float from the clip start, then truncated."""
    step = wp.min(
        future_init[k] + float(c.time_steps[e] + wp.int64(c.starts[e])), float(c.num_steps[e] - 1)
    )
    return c.length_starts[c.motion_ids[e]] + wp.int64(step)


@wp.func
def lib3(a: wp.array3d(dtype=float), f: wp.int64, b: int) -> wp.vec3:
    return wp.vec3(a[f, b, 0], a[f, b, 1], a[f, b, 2])


@wp.func
def lib4(a: wp.array3d(dtype=float), f: wp.int64, b: int) -> wp.vec4:
    return wp.vec4(a[f, b, 0], a[f, b, 1], a[f, b, 2], a[f, b, 3])


@wp.func
def env3(a: wp.array2d(dtype=float), e: int) -> wp.vec3:
    return wp.vec3(a[e, 0], a[e, 1], a[e, 2])


@wp.func
def wxyz(q: wp.quatf) -> wp.vec4:
    return wp.vec4(q[3], q[0], q[1], q[2])


@wp.func
def qmul(a: wp.vec4, b: wp.vec4) -> wp.vec4:
    """math_wxyz.quat_mul, same operation order."""
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
def qinv(q: wp.vec4) -> wp.vec4:
    """math_wxyz.quat_inv."""
    return wp.vec4(q[0], -q[1], -q[2], -q[3]) / wp.max(wp.dot(q, q), 1.0e-9)


@wp.func
def qapply(q: wp.vec4, v: wp.vec3) -> wp.vec3:
    """math_wxyz.quat_apply."""
    xyz = wp.vec3(q[1], q[2], q[3])
    t = wp.cross(xyz, v) * 2.0
    return v + q[0] * t + wp.cross(xyz, t)


@wp.func
def quat_error(q1: wp.vec4, q2: wp.vec4) -> float:
    """math_wxyz.quat_error_magnitude: the angle of q1 * conj(q2)."""
    d = qmul(q1, wp.vec4(q2[0], -q2[1], -q2[2], -q2[3]))
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
def rot6d(r: wp.vec4) -> vec6:
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
