"""The PPO minibatch update as one CUDA graph (``algo.config.update_graph``).

With ``bf16_autocast`` the update is bound by kernel launches (in nsys the GPU is busy about half of it).
Replaying the whole minibatch step, forward, losses, backward, gradient check, clipping and optimizer step, as one
graph removes the launch cost. The step must then make no host decision, so two of them move to the device:

- the KL-adaptive learning rate: the same rule on a device copy of ``args.learning_rate``, written into the
  learning rate tensor the optimizer reads, and back into ``args.learning_rate`` once per update;
- the NaN/Inf gradient check: instead of skipping ``optimizer.step()``, set ``found_inf``, which torch's fused
  AdamW reads to skip the step on the device; skipped steps are counted and printed once per update.

Requirements, checked at setup: one micro-batch per minibatch, dense encoder routing (static shapes), no symmetry
or image-augmentation losses, no log-std parameterization or multi-critic advantage weights, and torch's fused
AdamW (transformers' default optimizer with torch >= 2.8). The first minibatches run eagerly on a side stream
(warm-up); then one is captured, and every later one copies its rows into the captured input buffers and replays.

On several GPUs the graph also records the step's NCCL collectives: DDP's buffer broadcast in the forward and its
gradient all-reduce in the backward, and the gather of the KL for the learning rate. As PyTorch's notes on CUDA
graphs with DDP require, NCCL's async error handling is off (train_agent_trl.py sets
TORCH_NCCL_ASYNC_ERROR_HANDLING=0 before the process group starts), at least 11 DDP iterations run eagerly before
the capture, and DDP is built on a side stream (ppo_trainer.py wraps accelerator.prepare): building it fixes the
stream of every parameter's gradient accumulator, and on the legacy stream the captured backward would have to wait
on the capture.
"""
import os

import torch

from gear_sonic.trl.utils import rl

# mini-batch key -> rollout_data key, as _get_mb_rollout_data slices them
ROLLOUT_KEYS = {
    "mb_advantage": "advantages",
    "mb_logprobs": "logprobs",
    "mb_return": "returns",
    "mb_values": "values",
    "mb_dones": "dones",
    "mb_actions": "actions",
    "mb_old_mu": "old_mu_batch",
    "mb_old_sigma": "old_sigma_batch",
    "mb_padding_mask": "padding_mask",
    "mb_padding_mask_p1": "padding_mask_p1",
}


class UpdateGraph:
    def __init__(self, trainer, warmup=3):
        args = trainer.args
        if trainer.accelerator.num_processes > 1:
            if os.environ.get("TORCH_NCCL_ASYNC_ERROR_HANDLING") != "0":
                raise ValueError("update_graph on several GPUs needs TORCH_NCCL_ASYNC_ERROR_HANDLING=0")
            warmup = max(warmup, 11)  # DDP times its first iterations with events, which a capture cannot record
        if args.per_device_train_batch_size != args.local_mini_batch_size:
            raise ValueError("update_graph needs one micro-batch per minibatch")
        if trainer.use_symmetry or trainer.compute_imgaug_bc_loss:
            raise ValueError("update_graph does not support the symmetry or image-augmentation losses")
        # both read the host in the step: a NaN check on log_std, a host-to-device copy of the weights
        if trainer.multi_critic_advantage_weights is not None or any(
            getattr(m, "use_log_std", False) for m in trainer.policy_model.modules()
        ):
            raise ValueError("update_graph does not support use_log_std or multi_critic_advantage_weights")
        if not any(getattr(m, "dense_routing", False) for m in trainer.policy_model.modules()):
            raise ValueError("update_graph needs static shapes: set algo.config.actor.backbone.dense_routing=True")
        optimizer = getattr(trainer.optimizer, "optimizer", trainer.optimizer)
        if not all(group.get("fused") for group in optimizer.param_groups):
            raise ValueError("update_graph needs torch's fused AdamW (optim=adamw_torch_fused)")
        self.trainer = trainer
        self.warmup = warmup
        self.graph = None
        self.inputs = None
        self.loss_dict = None
        device = trainer.accelerator.device
        self.lr = torch.tensor(args.learning_rate, dtype=torch.float64, device=device)  # args.learning_rate
        self.group_lr = torch.tensor(args.learning_rate, device=device)  # what the optimizer reads
        for group in optimizer.param_groups:
            group["lr"] = self.group_lr  # the constant scheduler's step() fills it in place
            group["capturable"] = True  # the fused kernel is the same; step() checks the flag under capture
        self.found_inf = torch.zeros((), device=device)
        self.skipped = torch.zeros((), device=device)
        self.side = torch.cuda.Stream()  # one warm-up stream: the allocator keeps blocks per stream
        optimizer.found_inf = self.found_inf
        trainer._adjust_learning_rate_based_on_kl = self._adjust_learning_rate

    def _adjust_learning_rate(self, kl_mean, optimizer):  # noqa: ARG002
        """TRLPPOTrainer._adjust_learning_rate_based_on_kl on the device."""
        t = self.trainer
        if t.desired_kl is None:
            return
        # compare kl_mean in its own dtype, as the host rule does; the rate itself stays in float64
        lower = torch.clamp(self.lr / 1.5, min=t.adaptive_lr_min)
        higher = torch.clamp(self.lr * 1.5, max=t.adaptive_lr_max)
        lr = torch.where(
            kl_mean > t.desired_kl * 2.0,
            lower,
            torch.where((kl_mean < t.desired_kl / 2.0) & (kl_mean > 0.0), higher, self.lr),
        )
        self.lr.copy_(lr)
        self.group_lr.copy_(lr)

    def _step(self):
        """One minibatch update without host syncs: what the graph records."""
        t = self.trainer
        x = self.inputs
        x["episode_attnmask"] = rl.compute_episode_attnmask(x["mb_dones"])
        loss_dict = t._compute_loss(t._forward_model(t.model, x), x)
        t.accelerator.backward(loss_dict["loss"])
        params = [p for p in t.model.parameters() if p.grad is not None]
        norm = torch.nn.utils.get_total_norm([p.grad for p in params])
        nonfinite = ~torch.isfinite(norm)
        self.found_inf.copy_(nonfinite)
        self.skipped.add_(nonfinite)
        if t.args.max_grad_norm is not None and t.args.max_grad_norm > 0:
            # clip_grad_norm_ with the norm already computed (no grad scaler or FSDP here)
            torch.nn.utils.clip_grads_with_norm_(params, t.args.max_grad_norm, norm)
        t.optimizer.step()
        return loss_dict

    def _load(self, rollout_data, inds):
        x = self.inputs
        for key, value in rollout_data["all_obs_dict"].items():
            torch.index_select(value, 0, inds, out=x["mb_obs_dict"][key])
        for key, name in ROLLOUT_KEYS.items():
            torch.index_select(rollout_data[name], 0, inds, out=x[key])
        x["micro_batch_inds"].copy_(inds)

    def minibatch(self, rollout_data, inds):
        """Update on the rollout rows inds; returns the loss dict for the stats buffers."""
        t = self.trainer
        if self.inputs is None:
            self.inputs = t._get_mb_rollout_data(rollout_data, inds.clone())
            refreshed = {"micro_batch_inds", "mb_obs_dict", "episode_attnmask", *ROLLOUT_KEYS}
            if set(self.inputs) != refreshed:
                raise ValueError(f"update_graph would replay stale minibatch keys: {set(self.inputs) ^ refreshed}")
        else:
            self._load(rollout_data, inds)
        if self.graph is not None:
            self.graph.replay()
            return self.loss_dict
        if self.warmup > 0:
            self.warmup -= 1
            self.side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(self.side):
                loss_dict = self._step()
                t.optimizer.zero_grad(set_to_none=True)
            torch.cuda.current_stream().wait_stream(self.side)
            return loss_dict
        t.optimizer.zero_grad(set_to_none=True)  # the captured backward then writes the gradients, not adds
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.loss_dict = self._step()
        self.graph.replay()
        return self.loss_dict

    def end_update(self):
        """After the last minibatch of an update: the learning rate back to the host, skipped steps reported."""
        lr, skipped = torch.stack((self.lr, self.skipped.double())).tolist()  # one sync
        self.trainer.args.learning_rate = lr
        if skipped:
            print(f"NaN/Inf gradients: skipped {int(skipped)} optimizer steps in this update")  # noqa: T201
            self.skipped.zero_()
