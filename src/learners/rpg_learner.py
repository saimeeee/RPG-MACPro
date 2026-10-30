"""RPG continual QMIX learner (paper equations 3--12).

The replay buffer contains only the current task. At a task boundary we retain
parameter anchors/importance, not old transitions. Hypernetwork targets are
refreshed for *all* previous task embeddings at the latest boundary (Eq. 10).
"""

import copy
import math
from pathlib import Path
import random

import numpy as np
import torch as th
from torch.optim import RMSprop

from modules.mixers.rpg_mixer import RPGMixer


def _clone_mac(mac):
    """Copy parameters without copying non-leaf recurrent autograd tensors."""
    hidden, entropy = mac.hidden_states, mac.last_entropy
    try:
        mac.hidden_states = mac.last_entropy = None
        return copy.deepcopy(mac)
    finally:
        mac.hidden_states, mac.last_entropy = hidden, entropy


def _map_tensors(value, device):
    if isinstance(value, th.Tensor):
        return value.to(device)
    if isinstance(value, dict):
        return {k: _map_tensors(v, device) for k, v in value.items()}
    if isinstance(value, list):
        return [_map_tensors(v, device) for v in value]
    if isinstance(value, tuple):
        return tuple(_map_tensors(v, device) for v in value)
    return value


class RPGQLearner:
    """Train one task at a time; ``end_task`` must precede ``begin_task``."""

    CHECKPOINT_VERSION = 1

    def __init__(self, mac, logger, args):
        self.mac, self.logger, self.args = mac, logger, args
        self.task2decomposer = mac.task2decomposer
        self.mixer = RPGMixer(mac.surrogate_decomposer, args)
        self.mixer.to(next(mac.parameters()).device)
        self.target_mac = _clone_mac(mac)
        self.target_mixer = copy.deepcopy(self.mixer)
        # Include frozen, preallocated task embeddings so newly activated task
        # embeddings are optimised without rebuilding the optimiser.
        self.params = list(mac.parameters()) + list(self.mixer.parameters())
        self.optimiser = RMSprop(
            self.params, lr=getattr(args, "lr", 0.0005),
            alpha=getattr(args, "optim_alpha", 0.99),
            eps=getattr(args, "optim_eps", 0.00001),
        )
        self.current_task = None
        self.completed_tasks = []
        self.relation_anchors = {}
        self.relation_importance = {}
        self.hyper_targets = {}
        self.return_samples = []
        self.gamma_p = 0.0
        self.last_target_update_episode = 0
        self.log_stats_t = -getattr(args, "learner_log_interval", 2000) - 1
        self.train_steps = 0
        self.progress = {}  # Optional driver metadata, persisted with the model.
        self._freeze_targets()

    @property
    def device(self):
        return next(self.mac.parameters()).device

    def _freeze_targets(self):
        for p in list(self.target_mac.parameters()) + list(self.target_mixer.parameters()):
            p.requires_grad_(False)

    def begin_task(self, task, previous_task=None):
        if self.current_task == task and task not in self.completed_tasks:
            return  # Resuming an active task must not reset its state.
        if self.current_task is not None and self.current_task not in self.completed_tasks:
            raise RuntimeError("Call end_task on the active task before changing tasks")
        if task in self.completed_tasks:
            raise ValueError("Training a completed task again is not this continual protocol")
        expected_previous = self.completed_tasks[-1] if self.completed_tasks else None
        if previous_task is None:
            previous_task = expected_previous
        if previous_task != expected_previous:
            raise ValueError("previous_task must be the most recently completed task")
        self.mac.begin_task(task, previous_task)
        self.current_task = task
        self.return_samples = []
        self.gamma_p = 0.0
        self._update_targets()

    def observe_return(self, normalized_return):
        """Eq. 9, mean over the first N returns, then fixed for this task.

        The caller explicitly normalizes environment returns to [0, 1]. The
        supplied paper does not specify that normalization; it is an experiment
        configuration choice, not a learned min/max from future episodes.
        """
        value = float(normalized_return)
        if not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError("normalized_return must be a finite value in [0, 1]")
        count = int(getattr(self.args, "rpg_return_probe_episodes", 8))
        if count <= 0:
            raise ValueError("rpg_return_probe_episodes must be positive")
        if len(self.return_samples) < count:
            self.return_samples.append(value)
            self.gamma_p = sum(self.return_samples) / len(self.return_samples)
        return self.gamma_p

    def _td_loss(self, batch, task):
        if batch.max_seq_length < 2:
            raise ValueError("A training episode needs a transition and its next state")
        rewards = batch["reward"][:, :-1]
        actions = batch["actions"][:, :-1].long()
        terminated = batch["terminated"][:, :-1].float()
        # Requiring the next stored observation excludes final bootstrap rows
        # and padding even if filled marks every observed state as in MACPro.
        mask = batch["filled"][:, :-1].float() * batch["filled"][:, 1:].float()
        alive = th.cat((th.ones_like(terminated[:, :1]), 1 - terminated[:, :-1]), dim=1)
        mask = mask * alive.cumprod(dim=1)
        if mask.sum().item() == 0:
            raise ValueError("No valid transitions in training batch")

        self.mac.init_hidden(batch.batch_size, task)
        outputs, entropies = [], []
        for t in range(batch.max_seq_length):
            outputs.append(self.mac.forward(batch, task, t=t))
            if t < batch.max_seq_length - 1:
                entropies.append(self.mac.last_entropy)
        outputs = th.stack(outputs, dim=1)
        chosen = th.gather(outputs[:, :-1], 3, actions).squeeze(3)
        chosen_total = self.mixer(chosen, batch["state"][:, :-1], self.task2decomposer[task])

        with th.no_grad():
            self.target_mac.init_hidden(batch.batch_size, task)
            target_outputs = th.stack([
                self.target_mac.forward(batch, task, t=t)
                for t in range(batch.max_seq_length)
            ], dim=1)[:, 1:]
            available = batch["avail_actions"][:, 1:].bool()
            target_outputs = target_outputs.masked_fill(~available, -1e9)
            if getattr(self.args, "double_q", True):
                greedy = outputs[:, 1:].detach().masked_fill(~available, -1e9).argmax(3, keepdim=True)
                target_q = th.gather(target_outputs, 3, greedy).squeeze(3)
            else:
                target_q = target_outputs.max(3).values
            # Padding/dead agents can have no available action. Avoid enormous
            # artificial values from entering the mixer in those rows.
            target_q = th.where(available.any(3), target_q, th.zeros_like(target_q))
            target_total = self.target_mixer(target_q, batch["state"][:, 1:], self.task2decomposer[task])
            targets = rewards + getattr(self.args, "gamma", 0.99) * (1 - terminated) * target_total

        td_error = chosen_total - targets
        td_loss = (td_error.square() * mask).sum() / mask.sum()
        entropy = th.stack(entropies, dim=1)
        entropy_mask = mask.expand_as(entropy)
        attention_loss = (entropy * entropy_mask).sum() / entropy_mask.sum()
        details = {
            "td_error_abs": (td_error.abs() * mask).sum() / mask.sum(),
            "q_taken_mean": (chosen_total * mask).sum() / mask.sum(),
            "target_mean": (targets * mask).sum() / mask.sum(),
        }
        return td_loss, attention_loss, details

    def regularization_losses(self):
        """Return the already coefficient-weighted Eq. 8 and Eq. 10 terms."""
        reference = next(self.mac.parameters())
        relation = reference.new_zeros(())
        hyper = reference.new_zeros(())
        params = dict(self.mac.agent.relation_capturer.named_parameters())
        for task in self.completed_tasks:
            for name, param in params.items():
                relation = relation + (
                    self.relation_importance[task][name]
                    * (param - self.relation_anchors[task][name]).square()
                ).sum()
        relation = relation * (float(getattr(self.args, "rpg_lambda_rc", 100.0)) / 2)
        if self.hyper_targets:
            for task, target in self.hyper_targets.items():
                hyper = hyper + (self.mac.agent.generated_parameters(task) - target).square().sum()
            hyper = hyper * (float(getattr(self.args, "rpg_lambda_hy", 0.01)) / len(self.hyper_targets))
        return relation, hyper

    def train(self, batch, t_env, episode_num, task):
        if task != self.current_task or task in self.completed_tasks:
            raise RuntimeError("begin_task must activate the task before training")
        td_loss, attention_loss, details = self._td_loss(batch, task)
        relation, hyper = self.regularization_losses()
        total = (td_loss + float(getattr(self.args, "rpg_alpha_att", 0.001)) * attention_loss
                 + self.gamma_p * relation + hyper)
        if not th.isfinite(total).item():
            raise FloatingPointError("Non-finite RPG loss")
        self.optimiser.zero_grad(set_to_none=True)
        total.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(self.params, getattr(self.args, "grad_norm_clip", 10.0))
        if not th.isfinite(grad_norm).item():
            self.optimiser.zero_grad(set_to_none=True)
            raise FloatingPointError("Non-finite RPG gradient")
        self.optimiser.step()
        self.train_steps += 1
        if episode_num - self.last_target_update_episode >= getattr(self.args, "target_update_interval", 200):
            self._update_targets()
            self.last_target_update_episode = episode_num
        metrics = {
            "loss": total.item(), "td_loss": td_loss.item(),
            "attention_loss": attention_loss.item(), "relation_loss": relation.item(),
            "hyper_loss": hyper.item(), "gamma_p": self.gamma_p,
            "grad_norm": float(grad_norm),
            "weighted_attention_loss": float(getattr(self.args, "rpg_alpha_att", 0.001)) * attention_loss.item(),
            "weighted_relation_loss": self.gamma_p * relation.item(),
            "weighted_hyper_loss": hyper.item(),
            "calibration_episodes": len(self.return_samples),
            **{key: value.item() for key, value in details.items()},
        }
        if self.logger is not None and t_env - self.log_stats_t >= getattr(self.args, "learner_log_interval", 2000):
            for key, value in metrics.items():
                self.logger.log_stat("rpg/" + key, value, t_env)
            self.log_stats_t = t_env
        # Do not retain the just-consumed computation graph between episodes.
        self.mac.hidden_states = self.mac.last_entropy = None
        return metrics

    def end_task(self, task, buffer):
        """Estimate Eq. 7 from individual sampled episode TD gradients.

        An episode is the sampling unit because the capturer is recurrent. We
        average squared theta-times-gradient values, then divide *all* tensors
        by a single global maximum; squared batch-mean gradients would cancel.
        """
        if task != self.current_task or task in self.completed_tasks:
            raise RuntimeError("end_task requires an active, unfinished task")
        available = int(getattr(buffer, "episodes_in_buffer", buffer.batch_size))
        count = min(int(getattr(self.args, "rpg_importance_samples",
                                getattr(self.args, "rpg_importance_episodes", 32))), available)
        if count <= 0:
            raise ValueError("Importance estimation requires non-empty replay and positive sample count")
        sample = buffer.sample(count) if hasattr(buffer, "sample") else buffer[:count]
        named = dict(self.mac.agent.relation_capturer.named_parameters())
        importance = {name: th.zeros_like(param) for name, param in named.items()}
        for index in range(count):
            episode = sample[index:index + 1]
            episode = episode[:, :int(episode.max_t_filled().item())]
            episode.to(self.device)
            td_loss, _, _ = self._td_loss(episode, task)
            grads = th.autograd.grad(td_loss, tuple(named.values()), allow_unused=True)
            for (name, param), grad in zip(named.items(), grads):
                if grad is not None:
                    importance[name].add_((param.detach() * grad.detach()).square() / count)
        maximum = max((value.max().item() for value in importance.values()), default=0.0)
        if not math.isfinite(maximum):
            raise FloatingPointError("Non-finite relation parameter importance")
        if maximum > 0:
            importance = {name: value / maximum for name, value in importance.items()}
        self.relation_importance[task] = importance
        self.relation_anchors[task] = {name: param.detach().clone() for name, param in named.items()}
        self.completed_tasks.append(task)
        with th.no_grad():
            # Latest-boundary reference, including older tasks whose generated
            # heads may have changed during the task that just completed.
            self.hyper_targets = {
                old: self.mac.agent.generated_parameters(old).detach().clone()
                for old in self.completed_tasks
            }
        self.optimiser.zero_grad(set_to_none=True)
        self.mac.hidden_states = self.mac.last_entropy = None
        return {"importance_episodes": count, "importance_max_before_normalization": maximum,
                "importance_by_parameter": {
                    name: {"mean": value.mean().item(), "max": value.max().item(),
                           "nonzero_fraction": (value > 0).float().mean().item()}
                    for name, value in importance.items()},
                "gamma_p": self.gamma_p, "return_samples": list(self.return_samples)}

    def _update_targets(self):
        self.target_mac.load_state(self.mac)
        self.target_mixer.load_state_dict(self.mixer.state_dict())
        self._freeze_targets()

    def to(self, device):
        self.mac.to(device)
        self.target_mac.to(device)
        self.mixer.to(device)
        self.target_mixer.to(device)
        for attr in ("relation_anchors", "relation_importance", "hyper_targets"):
            setattr(self, attr, _map_tensors(getattr(self, attr), device))
        for param, state in self.optimiser.state.items():
            self.optimiser.state[param] = _map_tensors(state, device)
        return self

    def cuda(self):
        return self.to("cuda")

    def save_models(self, path):
        """Save training and continual-learning state, including target lag."""
        destination = Path(path)
        destination.mkdir(parents=True, exist_ok=True)
        np_state = np.random.get_state()
        checkpoint = {
            "version": self.CHECKPOINT_VERSION,
            "agent": self.mac.agent.state_dict(),
            "target_agent": self.target_mac.agent.state_dict(),
            "mixer": self.mixer.state_dict(),
            "target_mixer": self.target_mixer.state_dict(),
            "optimiser": self.optimiser.state_dict(),
            "current_task": self.current_task, "completed_tasks": self.completed_tasks,
            "relation_anchors": self.relation_anchors,
            "relation_importance": self.relation_importance,
            "hyper_targets": self.hyper_targets,
            "return_samples": self.return_samples, "gamma_p": self.gamma_p,
            "last_target_update_episode": self.last_target_update_episode,
            "log_stats_t": self.log_stats_t, "train_steps": self.train_steps,
            "progress": self.progress,
            "torch_rng": th.get_rng_state(),
            "cuda_rng": th.cuda.get_rng_state_all() if th.cuda.is_available() else [],
            "python_rng": random.getstate(),
            "numpy_rng": (np_state[0], np_state[1].tolist(), *np_state[2:]),
        }
        # Atomic replacement avoids treating a partial write as a checkpoint.
        temporary = destination / "learner.th.tmp"
        th.save(checkpoint, temporary)
        temporary.replace(destination / "learner.th")

    def load_models(self, path):
        checkpoint = th.load(Path(path) / "learner.th", map_location=self.device, weights_only=True)
        if checkpoint.get("version") != self.CHECKPOINT_VERSION:
            raise ValueError("Unsupported RPG checkpoint version")
        self.mac.agent.load_state_dict(checkpoint["agent"])
        self.target_mac.agent.load_state_dict(checkpoint["target_agent"])
        self.mixer.load_state_dict(checkpoint["mixer"])
        self.target_mixer.load_state_dict(checkpoint["target_mixer"])
        self.optimiser.load_state_dict(checkpoint["optimiser"])
        for attr in ("current_task", "completed_tasks", "relation_anchors", "relation_importance",
                     "hyper_targets", "return_samples", "gamma_p", "last_target_update_episode",
                     "log_stats_t", "train_steps", "progress"):
            setattr(self, attr, checkpoint[attr])
        self._freeze_targets()
        self.mac.hidden_states = self.mac.last_entropy = None
        self.target_mac.hidden_states = self.target_mac.last_entropy = None
        th.set_rng_state(checkpoint["torch_rng"].cpu())
        if th.cuda.is_available() and checkpoint["cuda_rng"]:
            th.cuda.set_rng_state_all([state.cpu() for state in checkpoint["cuda_rng"]])
        random.setstate(checkpoint["python_rng"])
        np_state = checkpoint["numpy_rng"]
        np.random.set_state((np_state[0], np.asarray(np_state[1], dtype=np.uint32), *np_state[2:]))
