"""RPG relation capturer and conditional hyper decision-maker.

Equations refer to Yao et al., IJCAI 2025. Network widths and the two-layer
generated decision MLPs are explicit reproduction choices; the supplied main
paper does not specify these architectural hyperparameters.
"""

import math

import torch
from torch import nn
from torch.nn import functional as F


def _stack_entities(parts, reference, width):
    """MACPro decomposers return one tensor per entity."""
    if isinstance(parts, torch.Tensor):
        return parts
    if parts:
        return torch.stack(parts, dim=-2)
    return reference.new_zeros(*reference.shape[:-1], 0, width)


def compact_actions(actions, n_ego):
    """Represent variable target actions by one shared interaction bit."""
    return torch.cat((actions[..., :n_ego],
                      actions[..., n_ego:].sum(dim=-1, keepdim=True)), dim=-1)


class RelationCapturer(nn.Module):
    """Separate teammate/other cross-attention followed by a shared GRU."""

    def __init__(self, own_dim, ally_dim, other_dim, args):
        super().__init__()
        self.hidden_dim = getattr(args, "rnn_hidden_dim", 64)
        self.entity_dim = getattr(args, "entity_embed_dim", 64)
        self.attn_dim = getattr(args, "attn_embed_dim", 32)
        self.query = nn.Linear(own_dim, self.attn_dim)
        self.own_value = nn.Linear(own_dim, self.entity_dim)
        self.ally_key = nn.Linear(ally_dim, self.attn_dim)
        self.ally_value = nn.Linear(ally_dim, self.entity_dim)
        self.other_key = nn.Linear(other_dim, self.attn_dim)
        self.other_value = nn.Linear(other_dim, self.entity_dim)
        self.rnn = nn.GRUCell(3 * self.entity_dim, self.hidden_dim)

    @staticmethod
    def attend(query, keys, values, valid):
        """Equations (1)-(3), including empty and wholly invisible sets."""
        if keys.shape[-2] == 0:
            return values.new_zeros(*query.shape[:-1], values.shape[-1]), query.new_zeros(query.shape[:-1])
        logits = (keys * query.unsqueeze(-2)).sum(-1) / math.sqrt(query.shape[-1])
        valid = valid.bool()
        # A finite sentinel avoids softmax(-inf, ..., -inf) -> NaN.
        logits = logits.masked_fill(~valid, torch.finfo(logits.dtype).min)
        weights = torch.softmax(logits, dim=-1) * valid.to(logits.dtype)
        weights = weights / weights.sum(-1, keepdim=True).clamp_min(1e-12)
        attended = (weights.unsqueeze(-1) * values).sum(dim=-2)
        entropy = -(weights * weights.clamp_min(1e-12).log()).sum(dim=-1)
        return attended, entropy

    def forward(self, own, allies, others, ally_valid, other_valid, hidden):
        query = self.query(own)
        ally_pattern, ally_entropy = self.attend(
            query, self.ally_key(allies), self.ally_value(allies), ally_valid)
        other_pattern, other_entropy = self.attend(
            query, self.other_key(others), self.other_value(others), other_valid)
        pattern = torch.cat((self.own_value(own), ally_pattern, other_pattern), -1)
        h = self.rnn(pattern, hidden.reshape(-1, self.hidden_dim))
        return h, ally_entropy + other_entropy


class HyperDecisionMaker(nn.Module):
    """Generate every weight and bias of ego and interaction decision MLPs."""

    def __init__(self, embedding_dim, hidden_dim, entity_dim, n_ego, args):
        super().__init__()
        decision_dim = getattr(args, "rpg_head_hidden_dim", hidden_dim)
        hyper_dim = getattr(args, "rpg_hyper_hidden_dim", 64)
        self.ego_shapes = [(decision_dim, hidden_dim), (decision_dim,),
                           (n_ego, decision_dim), (n_ego,)]
        self.interaction_shapes = [(decision_dim, hidden_dim + entity_dim),
                                   (decision_dim,), (1, decision_dim), (1,)]
        self.shapes = self.ego_shapes + self.interaction_shapes
        self.sizes = [math.prod(shape) for shape in self.shapes]
        self.net = nn.Sequential(nn.Linear(embedding_dim, hyper_dim), nn.ReLU(),
                                 nn.Linear(hyper_dim, sum(self.sizes)))
        # Generated weights use fan-in scaling. Bias generation stays learnable
        # but starts at zero, avoiding explosive initial TD targets.
        output = self.net[-1]
        with torch.no_grad():
            offset = 0
            for shape, size in zip(self.shapes, self.sizes):
                if len(shape) == 2:
                    output.weight[offset:offset + size].normal_(
                        0.0, 1.0 / math.sqrt(hyper_dim * shape[1]))
                else:
                    output.weight[offset:offset + size].zero_()
                output.bias[offset:offset + size].zero_()
                offset += size

    def forward(self, embedding):
        return self.net(embedding)

    def apply_generated(self, generated, hidden, other_features, has_interactions):
        tensors = [chunk.reshape(shape) for chunk, shape in
                   zip(generated.split(self.sizes), self.shapes)]
        w1, b1, w2, b2, iw1, ib1, iw2, ib2 = tensors
        ego_q = F.linear(F.relu(F.linear(hidden, w1, b1)), w2, b2)
        if not has_interactions:
            return ego_q
        repeated_h = hidden.unsqueeze(-2).expand(-1, other_features.shape[-2], -1)
        interaction_input = torch.cat((repeated_h, other_features), dim=-1)
        interaction_q = F.linear(
            F.relu(F.linear(interaction_input, iw1, ib1)), iw2, ib2).squeeze(-1)
        return torch.cat((ego_q, interaction_q), dim=-1)


class RPGAgent(nn.Module):
    """One shared relation model with a generated decision-maker per task."""

    def __init__(self, task2decomposer, args):
        super().__init__()
        if not task2decomposer:
            raise ValueError("RPG requires at least one task decomposer")
        self.task2decomposer = task2decomposer
        self.args = args
        self.task_keys = {task: "t{}".format(i) for i, task in enumerate(task2decomposer)}
        base = next(iter(task2decomposer.values()))
        self.n_ego = base.n_actions_no_attack
        self.hidden_dim = getattr(args, "rnn_hidden_dim", 64)
        self.entity_dim = getattr(args, "entity_embed_dim", 64)
        self.embedding_dim = getattr(args, "rpg_task_embed_dim", 32)
        self.alpha_init = getattr(args, "rpg_alpha_init", 0.5)
        self.embedding_std = getattr(args, "rpg_embedding_std", 1.0)
        if not 0 <= self.alpha_init <= 1:
            raise ValueError("rpg_alpha_init must lie in [0, 1]")
        signature = (base.own_obs_dim, base.obs_nf_al, base.obs_nf_en, self.n_ego)
        for task, decomposer in task2decomposer.items():
            actual = (decomposer.own_obs_dim, decomposer.obs_nf_al,
                      decomposer.obs_nf_en, decomposer.n_actions_no_attack)
            if actual != signature:
                raise ValueError("Entity feature layouts must agree across tasks; {} has {} vs {}".format(
                    task, actual, signature))
            interaction_count = decomposer.n_actions - self.n_ego
            if interaction_count not in (0, decomposer.n_enemies):
                raise ValueError("RPG expects zero or one interaction action per other entity")
        self.relation_capturer = RelationCapturer(
            base.own_obs_dim + self.n_ego + 1, base.obs_nf_al, base.obs_nf_en + 1, args)
        # This entity projection is part of the relation feature extractor and
        # receives return-aware importance regularization together with the GRU.
        self.relation_capturer.other_embedding = nn.Linear(base.obs_nf_en + 1, self.entity_dim)
        self.hypernetwork = HyperDecisionMaker(
            self.embedding_dim, self.hidden_dim, self.entity_dim, self.n_ego, args)
        self.task_embeddings = nn.ParameterDict({
            key: nn.Parameter(torch.randn(self.embedding_dim) * self.embedding_std,
                              requires_grad=False)
            for key in self.task_keys.values()})
        self.active_task = None
        self.begun_tasks = []

    def begin_task(self, task, previous_task=None):
        if task not in self.task_keys:
            raise KeyError("Unknown RPG task: {}".format(task))
        if task not in self.begun_tasks:
            if previous_task is None and self.begun_tasks:
                previous_task = self.begun_tasks[-1]
            with torch.no_grad():
                new = self.task_embeddings[self.task_keys[task]]
                if previous_task is not None:
                    if previous_task not in self.begun_tasks:
                        raise ValueError("Previous task must have been initialized")
                    previous = self.task_embeddings[self.task_keys[previous_task]]
                    new.copy_(self.alpha_init * previous + (1 - self.alpha_init) * new)
            self.begun_tasks.append(task)
        self.active_task = task
        for name, key in self.task_keys.items():
            self.task_embeddings[key].requires_grad_(name == task)

    def get_extra_state(self):
        return {"task_keys": self.task_keys, "active_task": self.active_task,
                "begun_tasks": list(self.begun_tasks)}

    def set_extra_state(self, state):
        if state["task_keys"] != self.task_keys:
            raise ValueError("Checkpoint task identities/order differ from configured tasks")
        self.active_task = state["active_task"]
        self.begun_tasks = list(state["begun_tasks"])
        for task, key in self.task_keys.items():
            self.task_embeddings[key].requires_grad_(task == self.active_task)

    def generated_parameters(self, task):
        return self.hypernetwork(self.task_embeddings[self.task_keys[task]])

    def init_hidden(self):
        return self.relation_capturer.query.weight.new_zeros(1, self.hidden_dim)

    def forward(self, obs, last_actions, hidden, task, embedding_task=None):
        decomposer = self.task2decomposer[task]
        own, enemy_parts, ally_parts = decomposer.decompose_obs(obs)
        allies = _stack_entities(ally_parts, own, decomposer.obs_nf_al)
        others = _stack_entities(enemy_parts, own, decomposer.obs_nf_en)
        if hasattr(decomposer, "decompose_obs_mask"):
            ally_valid, other_valid = decomposer.decompose_obs_mask(obs)
        else:
            ally_valid = allies.abs().sum(-1) > 0
            other_valid = others.abs().sum(-1) > 0
        if last_actions is None:
            last_actions = obs.new_zeros(obs.shape[0], decomposer.n_actions)
        has_interactions = decomposer.n_actions > self.n_ego
        previous_target = (last_actions[..., self.n_ego:] if has_interactions
                           else obs.new_zeros(obs.shape[0], decomposer.n_enemies))
        others = torch.cat((others, previous_target.unsqueeze(-1)), -1)
        own = torch.cat((own, compact_actions(last_actions, self.n_ego)), -1)
        hidden, entropy = self.relation_capturer(
            own, allies, others, ally_valid, other_valid, hidden)
        entity_features = F.relu(self.relation_capturer.other_embedding(others))
        generated = self.generated_parameters(task if embedding_task is None else embedding_task)
        q = self.hypernetwork.apply_generated(generated, hidden, entity_features, has_interactions)
        return q, hidden, entropy
