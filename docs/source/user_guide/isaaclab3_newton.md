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

1. Run `bash install_scripts/install_isaaclab3.sh [<Isaac Lab dir>]`. It clones Isaac Lab v3.0.0 unless the
   directory exists, creates its `.venv` from Isaac Lab's `uv.lock` (Python 3.12, torch 2.11 cu130, warp 1.17,
   newton 1.6.1), and installs `gear_sonic[training]` into it with the versions in
   `install_scripts/isaaclab3_constraints.txt`. Newton training needs no Isaac Sim; `ISAACSIM=1` also installs
   Isaac Sim 6.2 for `physics_backend=physx`. Do not run `uv sync` in the Isaac Lab directory afterwards: it
   restores Isaac Lab's Hydra 1.4, and SONIC's entry points need Hydra 1.3; run the script again instead.
   The script also works on aarch64 hosts: open3d, which only the mesh utilities load (on first use), is
   installed on x86_64 only.
2. Activate the environment: `source <Isaac Lab dir>/.venv/bin/activate`.
3. Fetch the G1 USD with `git lfs pull` (`gear_sonic/data/assets/robot_description/usd/g1/g1_sonic_newton.usd`).
   Isaac Lab 3.0 loads it by default instead of importing the URDF: its URDF importer ignores
   `replace_cylinders_with_capsules`, so this is the asset Isaac Lab 2.3.2 converted for training, with the mesh
   collision APIs moved onto the Mesh prims (otherwise Newton skips the 18 hand and wrist mesh colliders).
   `++manager_env.config.robot.frozen_usd=<path>` loads another USD, `=null` imports the URDF. To make it again
   (the first step needs an Isaac Lab 2.3 install, the second only `pxr`):

   ```sh
   python gear_sonic/scripts/isaaclab3/freeze_g1_usd.py --out_dir <dir>              # Isaac Lab 2.3
   python gear_sonic/scripts/isaaclab3/newtonize_usd.py <dir>/g1_sonic.usd <dir>/g1_sonic_newton.usd
   ```

## Run

Append these overrides to the usual training, eval or export commands:

```sh
++manager_env.config.physics_backend=newton
```

| Override | Default | Notes |
|---|---|---|
| `manager_env.config.physics_backend` | `physx` | `newton` selects MJWarp with Newton's collision pipeline |
| `manager_env.config.newton.num_substeps` | 1 | solver substeps per 5 ms physics step; 1 is faster than 2 and, for the released checkpoint, closer to PhysX |
| `manager_env.config.newton.ground_friction` | 0.3 | Newton combines friction with max(); the ground sits at the low end of the robot's randomized range |
| `manager_env.config.newton.use_cuda_graph` | True | the physics step is a CUDA graph |
| `manager_env.config.newton.gap` | 0.01 | contact detection distance [m]; most of the trimesh collision cost scales with it. 0.002 made training iterations 1.15x faster (1 substep) and left the released checkpoint's eval unchanged (96.1% vs 96.4%, within error) |
| `manager_env.config.newton.graph_decimation` | False | capture all physics steps of a control step in one CUDA graph. Isaac Lab 3.0 does this by itself only for robots with non-implicit actuators. Same throughput and eval on its own (the GPU had no gaps between the per-step graphs), but `sim.step` no longer blocks the CPU, which overlapping post-processing with physics needs. Changes the update rate of sensor histories (contact forces) and of finite-difference joint accelerations from per physics step to per control step |
| `manager_env.config.sync_free_step` | False | reset terminated envs through masks instead of `reset_buf.nonzero()`, so `env.step` has no host sync; see below |
| `manager_env.config.post_processing_graph` | False | with `sync_free_step`: record the post-processing as one CUDA graph and replay it; see below |
| `manager_env.config.fused_rewards` | False | evaluate the tracking reward set in one Warp kernel; `fused_rewards_check=True` compares it with the stock terms |
| `manager_env.config.fused_terminations` | False | evaluate the five tracking terminations in one Warp kernel; `fused_terminations_check=True` counts mismatches with the stock terms |
| `manager_env.config.fused_command` | False | the tracking command's relative body frames and metrics as one Warp kernel each; `fused_command_check=True` compares them with the torch versions |
| `manager_env.config.fused_obs` | False | 13 tokenizer and critic observation terms as Warp kernels (SMPL, multi-future joints / root height / root orientation, anchor and body frames, VR points); `fused_obs_check=True` compares them with the stock terms |
| `algo.config.actor.backbone.dense_routing` | False | run every encoder on the whole minibatch in the PPO update and evaluate the latent alignment losses under row masks: static shapes, no host syncs |
| `algo.config.align_linear` | False | zero-pad the matmuls of Linear layers whose sizes are not multiples of 4, so cuBLAS uses its aligned TF32 kernels; parameters keep their shapes; see below |
| `algo.config.bf16_autocast` | False | policy and critic under bf16 autocast, output layers in FP32; see below |

## Overlapping post-processing with physics

Isaac Lab finds the envs to reset with `reset_buf.nonzero()`, which waits for the step's physics. The reset,
command and observation work after it (about 1300 small kernels per control step) then runs while the GPU is
idle. `sync_free_step` applies every reset-path update to all envs under the reset mask, through Isaac Lab 3.0's
`*_mask` writes, so the CPU queues that work while the GPU still runs the physics:

```sh
export CUDA_SCALE_LAUNCH_QUEUES=4x   # before CUDA starts: the default launch queues fill up mid-step
... ++manager_env.config.sync_free_step=True ++manager_env.config.newton.graph_decimation=True
```

On one H100 PCIe (1 substep, 2 mm gap, 4096 envs) a training iteration goes from 3.37 s to 2.84 s and a control
step from 101 ms to 65 ms, of which the GPU idles 2 ms; the step is now bound by the physics on the GPU.
`graph_decimation` adds about 0.5% on top of the other two. Rollout inference has no host syncs either: every
encoder runs on the whole batch and rows are picked with `torch.where` (always on, bitwise-identical actions).

In eval mode the masked reset is exact against the default path on 130 state tensors
(`sonic_il3/p3/test_sync_free.py` in the migration workspace), and the released checkpoint's eval is unchanged
(96.2% vs 96.1-96.2%). `enable` refuses tasks that use what it does not implement: curriculum, recorder or
reset-mode event terms, stateful observation terms, interval events other than `push_by_setting_velocity`,
scene objects, or tracking without a time_out termination.

### Replaying the post-processing as a CUDA graph

Without host syncs and with fixed shapes, the post-processing can be recorded once with `torch.cuda.graph`
and replayed, so the CPU submits it with one call instead of about 1300 launches:

```sh
... ++manager_env.config.sync_free_step=True ++manager_env.config.post_processing_graph=True
```

Warp work inside it (Isaac Lab's `*_mask` writes, lazy articulation data, Newton's forward kinematics) is
recorded on the same stream. Attributes that the step reassigns are copied back into the buffers the graph
reads; attributes that code outside the step reassigns (adaptive sampling statistics, a reloaded motion batch
of the same size) are copied in before each replay. A change of shape, or of the command term's Python flags
(evaluation mode), records another graph. From the same state and random state, replays match the eager
sync-free step bit for bit (`sonic_il3/p3/test_graph.py`: 279 outputs, torch states and simulator / sensor
buffers over 60 steps of training, an adaptive sampling update, evaluation, and a motion reload).

On one H100 PCIe (1 substep, 2 mm gap, 4096 envs) the training iteration on the trimesh terrain stays at
2.82 s: the physics on the GPU sets the pace there. On a plane the collection goes from 0.80 s to 0.66 s per
iteration (a control step from 33 ms to 27.5 ms), and the step becomes GPU-bound there too.

### Warp kernels for the heaviest terms

A per-term trace of the sync-free step puts the post-processing at about 3.6 ms of GPU kernel time and 1450
kernels per control step (4096 envs). `fused_rewards`, `fused_terminations`, `fused_command` and `fused_obs`
evaluate the heaviest groups as Warp kernels, one thread per env (and future frame), reading the reference
motion straight from the motion library and the robot state from Isaac Lab's Warp buffers. Each has a
`*_check` switch that also runs the torch terms and prints the largest differences.

```sh
... ++manager_env.config.fused_rewards=True ++manager_env.config.fused_terminations=True \
    ++manager_env.config.fused_command=True ++manager_env.config.fused_obs=True
```

In check mode (256 envs, 96 steps) the observation terms are within 7.2e-7 of the torch terms (the pure
gathers are equal), the metrics within 1.4e-6, the relative body frames within 7.6e-6 m and 3e-7, the
rewards within 6.6e-7 relative, and the terminations are identical. With all four on, graph replays still
match the eager step bit for bit (282 items over 300 steps). The critic's `motion_anchor_pos_b` stays a torch
term: Isaac Lab 3.0 adds its default `mask_out_z` to the term's parameters, which the Warp version does not
implement.

With the post-processing graph, the four switches cut its GPU time per control step from 3.9 ms (1225
kernels) to 2.7 ms (518 kernels). On one H100 PCIe (1 substep, 2 mm gap, 4096 envs) the collection takes
1.58 s instead of 1.62 s per iteration on the trimesh terrain (2.78 s instead of 2.81 s per iteration) and
0.63 s instead of 0.66 s on a plane (1.82 s instead of 1.86 s), about 26 ms per control step there.
A 500-iteration fine-tune of the released checkpoint with the graph and all four switches on evaluates like
the same fine-tune without them, both with `sync_free_step` and `graph_decimation` (Newton 96.9% vs 96.6%,
PhysX 97.9% vs 98.2% over 4096 motions, within the eval's rerun noise of about 0.6 points).

Warp code that runs inside the post-processing has to launch on Warp's current stream, which the graph
capture sets to the capture stream. Do not pass a `wp.stream_from_torch(...)` wrapper per launch: when the
wrapper is freed during the capture, Warp resets its current stream to the legacy stream, and later Warp
memsets (Newton clearing its forward-kinematics requests) run once instead of being recorded.

### Host syncs in the PPO update

In the update each encoder runs on its rows of the minibatch. Picking those rows with boolean masks synced
with the host at every use, and again in the backward pass on the autograd thread (about 840 syncs per
iteration). The routing now turns the masks into row indices once per forward and selects with them: the same
rows in the same order, bitwise-equal action means, aux losses and gradients, and no syncs in the forward or
backward indexing. On one H100 PCIe (4096 envs, trimesh, post-processing graph and fused terms on) the
update takes 1.11 s instead of 1.17 s per iteration.

`algo.config.actor.backbone.dense_routing=True` instead runs every encoder on the whole minibatch, as rollout
inference does, and evaluates the latent alignment losses under row masks: static shapes and no syncs from the
routing, the same values and gradients up to float rounding (FP32: 1.6e-6 relative; with the trainer's TF32
matmuls the encoder weight gradients round differently by up to 2e-4). Every encoder then encodes all 24576
rows of a minibatch, about 1.6 times the encoder work, and the update gets slower (1.20 s to 1.31 s per
iteration), so it is off by default. Compiling the encoders and decoders with `torch.compile` did not help
either (1.163 s against 1.165-1.171 s per iteration).

### Matmul shapes and bf16 in the update

About a fifth of the update's GPU time went to cuBLAS `align1` TF32 kernels, picked for four Linear layers
whose sizes are not multiples of 4: the action decoder (994 inputs, 29 outputs), the teleop encoder's input
layer (267) and the critic's input layer (1645). `algo.config.align_linear=True` pads their inputs, weights and
outputs with zeros to multiples of 4 inside the forward; the parameters, checkpoints and ONNX exports keep their
shapes. The update takes 1.04 s instead of 1.12 s per iteration (2.61 s against 2.69 s per iteration).

`algo.config.bf16_autocast=True` runs the policy and critic forward and backward passes under bf16 autocast.
The last Linear layer of each network (action means, values, latent heads) stays in FP32, as do the losses,
optimizer and parameters. With `align_linear` the update takes 0.72 s instead of 1.04 s (2.28 s against
2.61 s per iteration, two runs each on one node). It changes the numerics, so it is off by default until a
fine-tune with it matches the FP32 one in the eval.

## Known differences on Newton

- Friction is combined with max() instead of the multiply mode used with PhysX.
- MJWarp does not enforce joint velocity limits. Effort limits do apply: Newton turns the implicit
  PD drives into MuJoCo position actuators with `forcerange = ±effort_limit_sim`.
- `undesired_contacts` reads normal contact forces in Isaac Lab 3.0 (total force in 2.3).
- Contacts within the shape gap (1 cm by default) are generated; the gap drives most of the
  collision cost on the trimesh terrain.

## Fine-tuning the released checkpoint on one GPU

The released checkpoint was trained on 128 GPUs, and its stored learning rate is the adaptive minimum (1e-5).
Fine-tuning it on one GPU with the default PPO settings (actor learning rate 2e-5, adaptive range [1e-5, 2e-4],
desired KL 0.01) makes it worse on either backend: on PhysX the eval success rate drops from 97.2% to 93.6% after
1000 iterations. A step about five times smaller keeps it improving:

```sh
++algo.config.actor_learning_rate=5e-6 ++algo.config.adaptive_lr_min=2e-6 \
++algo.config.adaptive_lr_max=2e-5 ++algo.config.desired_kl=0.003
```

With these settings, 500 iterations on Newton (2 substeps) lowered the local MPJPE on PhysX by 1.9 mm without
losing success rate (98.1%, against 97.2% and 97.5% in two evals of the released checkpoint; reruns of the same
eval differ by up to about 0.6 points, so compare MPJPE rather than small success-rate gaps). Leg and global position
errors grow (+3 mm and +16 mm at 500
iterations) and keep drifting in longer runs, so compare the checkpoints from 500 to 1000 iterations and check the
legs before deploying one.

When resuming with `+resume=True`, set `algo.config.num_learning_iterations` to the absolute iteration count
(for example 41550 + 2000): training stops once the global step reaches it.

## Measured on one H100 PCIe, 4096 envs (2026-11-09)

| Stack | s / iteration | Eval success rate, released checkpoint, 4096 motions |
|---|---|---|
| Isaac Lab 2.3.2 + PhysX | 5.92 | 97.2% |
| Isaac Lab 3.0 + Newton, 2 substeps | 4.08 | 95.0% |
| Isaac Lab 3.0 + Newton, 1 substep | 3.80 | 96.4% |
| + 2 mm gap, `sync_free_step`, `graph_decimation` | 2.84 | 96.2% |
| + `post_processing_graph` and the four `fused_*` switches | 2.78 | — |
| + row-index routing and the total-norm gradient check in the update (always on) | 2.69 | — |
| + `align_linear` | 2.61 | — |
| + `bf16_autocast` | 2.28 | — |

The last four rows were measured on another node of the same type (2.82 s there for the row above them).

On motions both stacks track, the released checkpoint's local MPJPE differs from PhysX by
+0.05 mm with 1 substep and +0.36 mm with 2 substeps.

ONNX exports from both stacks are identical for the released checkpoint. `torch.onnx.export` is
called with `dynamo=False`, because torch 2.9 and later default to the dynamo exporter, which
rejects the dict inputs used here.
