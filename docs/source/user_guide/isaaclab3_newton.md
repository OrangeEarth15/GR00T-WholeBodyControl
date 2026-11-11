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
| `manager_env.config.action_delay` | None | `[min, max]` in physics steps (5 ms): the joints apply the action the policy chose `d` physics steps earlier, `d` drawn per env at every episode start; the policy still observes its latest action. `[4, 4]` is a fixed one-control-step delay, e.g. for an eval; with `graph_decimation` only whole control steps; see below |
| `algo.config.update_graph` | False | capture each PPO minibatch step as one CUDA graph and replay it, on several GPUs with DDP's collectives in the graph; needs `dense_routing` and torch's fused AdamW; see below |

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

Three more syncs per minibatch did not affect the result and are gone: the aux-loss sums started from
`torch.tensor(0.0, device=...)`, a copy from pageable host memory that waits for the stream; the loss was checked
for NaN/Inf in every minibatch, although a non-finite loss also makes the gradients non-finite (the skipped-step
message now prints the loss); and the policy checked its std on the host and sampled through `torch.normal`, which
reads the std back as well (`rsample()` under `no_grad` draws the same values). 202 syncs per update remain, ten
per minibatch: the routing's row indices, the KL-adaptive learning rate and the gradient check. With bf16 the update
takes 0.714 s instead of 0.728 s (2.30 s against 2.32 s per iteration, two runs each on one node).

### Matmul shapes and bf16 in the update

About a fifth of the update's GPU time went to cuBLAS `align1` TF32 kernels, picked for four Linear layers
whose sizes are not multiples of 4: the action decoder (994 inputs, 29 outputs), the teleop encoder's input
layer (267) and the critic's input layer (1645). `algo.config.align_linear=True` pads their inputs, weights and
outputs with zeros to multiples of 4 inside the forward; the parameters, checkpoints and ONNX exports keep their
shapes. The update takes 1.04 s instead of 1.12 s per iteration (2.61 s against 2.69 s per iteration).

`algo.config.bf16_autocast=True` runs the policy and critic forward and backward passes under bf16 autocast.
The last Linear layer of each network (action means, values, latent heads) stays in FP32, as do the losses,
optimizer and parameters. With `align_linear` the update takes 0.72 s instead of 1.04 s (2.28 s against
2.61 s per iteration, two runs each on one node). A 500-iteration fine-tune of the released checkpoint with it
(small-step recipe below, every switch on) evaluated like the same fine-tune without bf16: success 96.51% against
96.61% on Newton and 97.85% against 98.22% on PhysX, local MPJPE +0.00 mm and +0.12 mm on motions both track,
less than the 0.45 mm between two FP32 fine-tunes. It changes the numerics, so it stays off by default.

### The update as one CUDA graph

With bf16 the update is bound by kernel launches: in nsys it takes 908 ms, of which the GPU is busy 508 ms (56%),
against 1181 ms and 80% in FP32. Dense routing then costs nothing (0.712 s against 0.713 s per update, two runs
each), and it makes the update's shapes static. `algo.config.update_graph=True` captures one minibatch step
(forward, losses, backward, gradient check, clipping, optimizer step) as a CUDA graph after three eager warm-up
minibatches, and replays it for every later minibatch:

```sh
... ++algo.config.align_linear=True ++algo.config.bf16_autocast=True \
    ++algo.config.actor.backbone.dense_routing=True ++algo.config.update_graph=True
```

The step's two host decisions run on the device instead. The KL-adaptive learning rate updates a tensor that the
optimizer reads, copied back to `args.learning_rate` once per update. A non-finite gradient norm sets the
`found_inf` flag of torch's fused AdamW, which then skips the step on the device; skipped steps are printed once
per update. It needs one micro-batch per minibatch, dense routing, no symmetry or image-augmentation losses, and the
fused AdamW (`adamw_torch_fused`, transformers' default with torch >= 2.8), and refuses to start otherwise. On
several GPUs the graph also records DDP's buffer broadcast and gradient all-reduce and the gather of the KL, as
PyTorch's notes on CUDA graphs with DDP describe: NCCL's async error handling is turned off
(`TORCH_NCCL_ASYNC_ERROR_HANDLING=0`, set by `train_agent_trl.py`; torchrun sets it to 1), at least 11 DDP
iterations run eagerly before the capture, and `accelerator.prepare` runs on a side stream, because building DDP fixes
the stream of every parameter's gradient accumulator and the captured backward cannot depend on the legacy stream.
On two GPUs the parameters after an update are bitwise equal on both ranks to those of the eager DDP path; on four
H100 PCIe the update takes 0.738 s instead of 0.845 s per iteration (2.29 s against 2.40 s, two runs each). On a real rollout (20 minibatches) the parameters after the
update are bitwise equal to those of the eager path (`sonic_il3/p3/test_update_graph.py`). On one H100 PCIe (4096
envs, every switch, `align_linear`, bf16) the update takes 0.632 s instead of 0.714 s per iteration (2.23 s against
2.31 s per iteration, two runs each on one node). In nsys the GPU is now busy 96% of the update (597 of 624 ms),
and two host syncs per update remain (the learning rate and the skipped-step count, read once at its end). The
GPU time grew by about 90 ms over the eager update with row-index routing: that is the encoder work dense routing
adds, which the eager update had hidden in its launch gaps. A 500-iteration fine-tune of the released checkpoint
with it (the bf16 recipe above plus dense routing) evaluated like the one without: success 96.90% against 96.51% on
Newton and 97.90% against 97.85% on PhysX, local MPJPE +0.22 mm and +0.31 mm on motions both track, within the
0.45 mm between two FP32 fine-tunes; no step was skipped.

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
eval differ by up to about 0.6 points, so compare MPJPE rather than small success-rate gaps). With the default
position rewards, leg and global position errors grow (+3 to 4 mm and +12 to 19 mm at 500 iterations, about half of
the legs' share also when fine-tuning on PhysX) and keep drifting in longer runs. A heavier root-position term and a
tighter body-position term hold them:

```sh
++manager_env.rewards.tracking_anchor_pos.weight=1.5 ++manager_env.rewards.tracking_anchor_pos.params.std=0.2 \
++manager_env.rewards.tracking_relative_body_pos.params.std=0.1
```

(defaults: weight 0.5 and std 0.3 for `tracking_anchor_pos`, std 0.3 for `tracking_relative_body_pos`). Two
500-iteration fine-tunes with them (small step, 1 substep, 2 mm gap, every switch, bf16) evaluated at 96.85% and
97.12% on Newton and 98.19% on PhysX, local MPJPE 2.1 to 3.1 mm lower than the released checkpoint, legs only +1.1
to +1.9 mm and no global drift (−9 to +1 mm); after 1000 iterations the global position still did not drift (legs
+1.8 to +2.0 mm). Tightening the std alone halves the leg drift but doubles the global drift; the root term needs
the weight. In MuJoCo sim2sim with the C++ deploy (50 training motions) these fine-tunes track with 3.7 to 4.2 mm
lower local MPJPE than the released checkpoint and foot errors within 0.5 to 2.1 mm of it (the default rewards:
2.4 to 3.0 mm and +3 to 6 mm).

When resuming with `+resume=True`, set `algo.config.num_learning_iterations` to the absolute iteration count
(for example 41550 + 2000): training stops once the global step reaches it.

On four GPUs (`accelerate launch --multi_gpu --num_processes 4`, 4096 envs each) the default step size works:
2000 iterations on Newton (1 substep, 2 mm gap, every switch above) end at 96.68% on Newton and 97.44% on PhysX
(one GPU with the default step: 92.4% and 92.9%), with 0.9-1.1 mm lower local MPJPE. Legs (+6 mm) and global position
(+36 to 38 mm) drift about as much as with the small step over 2000 iterations, so watch them in any longer run.
On one node with eight H100 PCIe (no NVLink) the throughput is 95% of linear on four GPUs and 91% on eight (2.57 s
per iteration on one GPU, 2.83 s on eight); the extra time is the gradient all-reduce in the update. With
`bf16_autocast` the same node ran one to four GPUs at 94% of linear, but eight GPUs hung in all seven tries: five in
the first update (also with `TORCH_DISTRIBUTED_DEBUG=DETAIL`, `CUDA_MODULE_LOADING=EAGER`, `NCCL_P2P_DISABLE=1`
and the default launch queues; some ranks stalled in a bf16 GEMM in the backward pass, the others in the gradient
check after it), and after 7 and 16 iterations with dense routing, eager and with `update_graph`. FP32 on eight GPUs
ran 2000 iterations without a hang. The cause is not known yet; use FP32 on eight GPUs until it is.

The policy has little margin for action delay: evaluated with the joints one control step (20 ms) behind the
policy, the released checkpoint's success rate drops from 96.1% to 91.0% on Newton, and two steps fail nearly every
motion; Newton fine-tunes behave the same. `manager_env.config.action_delay=[min, max]` delays the joints by `min` to
`max` physics steps (5 ms each), drawn per env at every episode start; `[4, 4]` evaluates at a fixed one-control-step
delay. Fine-tunes with the position rewards above (one GPU, small step, 500 iterations), with and without a random
delay in training, evaluated on Newton (success rate):

| Delay in training | 0 ms | 10 ms | 20 ms | 40 ms | Global position, no delay |
|---|---|---|---|---|---|
| none | 96.9% | 95.7% | 89.7% | 1.9% | +1 mm |
| 0 to 10 ms | 96.4% | 96.0% | 92.1% | 9.8% | +19 mm |
| 0 to 20 ms | 96.0% | 95.8% | 93.6% | 45.7% | +40 mm |

(released checkpoint: 96.1%, 95.2%, 91.0%, 0.7%; global position relative to it). A random delay in training buys
margin for long delays at the cost of no-delay tracking, mostly the global position; a heavier root term (weight 3.0)
halves that drift but cuts the gain in local MPJPE by a third, and 1000 iterations make the drift larger. At 10 ms all of them track
about equally, so measure the robot's latency from observation to motor command before choosing.

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
| + `dense_routing` and `update_graph` | 2.23 | — |

The four rows from the routing to bf16 were measured on another node of the same type (2.82 s there for the row
above them), the last one on a third node (2.31 s there without `update_graph`).

On motions both stacks track, the released checkpoint's local MPJPE differs from PhysX by
+0.05 mm with 1 substep and +0.36 mm with 2 substeps.

The speed switches work in the eval as well (`sync_free_step`, `post_processing_graph`, `graph_decimation` and the four
`fused_*` switches, with `CUDA_SCALE_LAUNCH_QUEUES=4x`): the released checkpoint's 4096-motion eval on Newton took
11.1 min instead of 13.2 min (16.9 instead of 20.3 min with startup) on one H100 PCIe, at 96.36% against 96.12%
and the same local MPJPE (−0.015 ± 0.025 mm) on motions both runs track, within the eval's rerun noise.

The eval command merges `terminations=tracking/eval` into the training terminations, so the training-only
`foot_pos_xyz` (an ankle more than 0.2 m from the reference) still ends episodes. Without it
(`++manager_env.config.train_only_terminations=[foot_pos_xyz]`) the released checkpoint scores 98.39% instead of
96.14% on Newton (1 substep, 2 mm gap) and 99.15% instead of 97.17% on Isaac Lab 2.3.2 + PhysX, with the same local
MPJPE on motions both evals track.

评测可选 `eval_motion_keys_file=/absolute/path/keys.json`（默认 `null`）。文件是非空 JSON 字符串列表，
例如 `["motion_a", "motion_b"]`；名称按字面匹配，按文件顺序加载。指定后使用完整列表，忽略
`max_unique_motions` 的随机抽样上限。重复、缺失或被已有 filter/remove 规则排除的 key 会报错。
不指定时沿用原来的筛选与随机抽样。此参数只提供固定列表能力，正式动作集和条数仍由 SONIC team 决定。
固定 key 不保证物理逐位确定；比较仍需重复评测、按动作名配对。

ONNX exports from both stacks are identical for the released checkpoint. `torch.onnx.export` is
called with `dynamo=False`, because torch 2.9 and later default to the dynamo exporter, which
rejects the dict inputs used here.
