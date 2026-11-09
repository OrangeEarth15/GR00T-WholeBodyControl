# Isaac Lab 3.0 and Newton (experimental)

This branch runs SONIC training, evaluation and ONNX export on Isaac Lab 3.0 (GA, v3.0.0), with either
PhysX or the Newton MJWarp solver. The default is still PhysX with the settings the released
checkpoint was trained with; Newton is selected per run.

## What changes compared with Isaac Lab 2.3

- Launch: `isaaclab.app.launch_simulation` instead of `AppLauncher`. On Newton no Kit app starts.
- Quaternions: Isaac Lab 3.0 data is `(x, y, z, w)`. SONIC math stays `(w, x, y, z)`: the 2.3 helpers
  live in `gear_sonic/isaac_utils/math_wxyz.py`, and robot data is converted where it is read
  (`wxyz_from_xyzw`) and written (`xyzw_from_wxyz`). Observations, the ONNX inputs and the deploy
  side are unchanged.
- Joint and body order are pinned to the Isaac Lab 2.3 PhysX order with explicit name lists in
  `robots/g1.py` (`joint_ordering`, `body_ordering`). Newton's native order is depth-first.
- Robot asset: the 3.0 URDF importer ignores `replace_cylinders_with_capsules`, so the G1 USD is
  converted once with Isaac Lab 2.3 and loaded through `manager_env.config.robot.frozen_usd`.
- `AdditiveUniformNoiseCfg` is gone in 3.0 (`UniformNoiseCfg`), and CoM randomization uses the
  backend-agnostic Isaac Lab term. Eval maps both names when it reads a 2.x checkpoint config.

## Setup

1. Install Isaac Lab 3.0 (v3.0.0) with uv. Newton training needs no Isaac Sim.
2. Install SONIC into the same environment: `pip install -e "gear_sonic[training]"` (Python 3.12,
   numpy 2).
3. Make the G1 USD once. The conversion step needs an Isaac Lab 2.3 install; the second step only
   needs `pxr`:

   ```sh
   python gear_sonic/scripts/isaaclab3/freeze_g1_usd.py --out_dir <dir>              # Isaac Lab 2.3
   python gear_sonic/scripts/isaaclab3/newtonize_usd.py <dir>/g1_sonic.usd <dir>/g1_sonic_newton.usd
   ```

   `newtonize_usd.py` moves the collision APIs from the importer's Xform onto the child Mesh;
   without it Newton skips the 18 hand and wrist mesh colliders.

## Run

Append these overrides to the usual training, eval or export commands:

```sh
++manager_env.config.physics_backend=newton \
++manager_env.config.robot.frozen_usd=<dir>/g1_sonic_newton.usd
```

| Override | Default | Notes |
|---|---|---|
| `manager_env.config.physics_backend` | `physx` | `newton` selects MJWarp with Newton's collision pipeline |
| `manager_env.config.newton.num_substeps` | 1 | solver substeps per 5 ms physics step; 1 is faster than 2 and, for the released checkpoint, closer to PhysX |
| `manager_env.config.newton.ground_friction` | 0.3 | Newton combines friction with max(); the ground sits at the low end of the robot's randomized range |
| `manager_env.config.newton.use_cuda_graph` | True | the physics step is a CUDA graph |
| `manager_env.config.newton.gap` | 0.01 | contact detection distance [m]; most of the trimesh collision cost scales with it (2 mm: physics 1.39x faster on the probe) |
| `manager_env.config.fused_rewards` | False | evaluate the tracking reward set in one Warp kernel; `fused_rewards_check=True` compares it with the stock terms |

## Known differences on Newton

- Friction is combined with max() instead of the multiply mode used with PhysX.
- MJWarp does not enforce joint velocity limits. Effort limits do apply: Newton turns the implicit
  PD drives into MuJoCo position actuators with `forcerange = ±effort_limit_sim`.
- `undesired_contacts` reads normal contact forces in Isaac Lab 3.0 (total force in 2.3).
- Contacts within the shape gap (1 cm by default) are generated; the gap drives most of the
  collision cost on the trimesh terrain.

## Measured on one H100 PCIe, 4096 envs (2026-11-09)

| Stack | s / iteration | Eval success rate, released checkpoint, 4096 motions |
|---|---|---|
| Isaac Lab 2.3.2 + PhysX | 5.92 | 97.2% |
| Isaac Lab 3.0 + Newton, 2 substeps | 4.08 | 95.0% |
| Isaac Lab 3.0 + Newton, 1 substep | 3.80 | 96.4% |

On motions both stacks track, the released checkpoint's local MPJPE differs from PhysX by
+0.05 mm with 1 substep and +0.36 mm with 2 substeps.

ONNX exports from both stacks are identical for the released checkpoint. `torch.onnx.export` is
called with `dynamo=False`, because torch 2.9 and later default to the dynamo exporter, which
rejects the dict inputs used here.
