"""Variable-entity monotonic QMIX with multi-head state attention.

The supplied paper describes multi-head attention in the mixer but omits the
appendix specification. This module preserves QMIX monotonicity explicitly:
state-dependent weights are nonnegative and never depend on agent Q-values.
"""

import torch
from torch import nn
from torch.nn import functional as F

from modules.agents.rpg_agent import _stack_entities, compact_actions


class RPGMixer(nn.Module):
    def __init__(self, surrogate_decomposer, args):
        super().__init__()
        self.entity_dim = getattr(args, "entity_embed_dim", 64)
        self.mixing_dim = getattr(args, "mixing_embed_dim", 32)
        self.n_heads = getattr(args, "rpg_mixer_heads", 4)
        if self.entity_dim % self.n_heads:
            raise ValueError("entity_embed_dim must be divisible by rpg_mixer_heads")
        self.state_last_action = surrogate_decomposer.state_last_action
        self.n_ego = surrogate_decomposer.n_actions_no_attack
        self.timestep_dim = getattr(surrogate_decomposer, "timestep_number_state_dim", 0)
        ally_dim = surrogate_decomposer.state_nf_al
        if self.state_last_action:
            ally_dim += self.n_ego + 1
        self.ally_encoder = nn.Linear(ally_dim, self.entity_dim)
        self.other_encoder = nn.Linear(surrogate_decomposer.state_nf_en, self.entity_dim)
        self.attention = nn.MultiheadAttention(self.entity_dim, self.n_heads, batch_first=True)
        self.norm = nn.LayerNorm(self.entity_dim)
        context_dim = self.entity_dim + self.timestep_dim
        # Per-agent contexts let credit assignment vary between agents while the
        # same network supports any team size.
        self.hyper_w1 = nn.Linear(self.entity_dim + context_dim, self.mixing_dim)
        self.hyper_b1 = nn.Linear(context_dim, self.mixing_dim)
        self.hyper_w2 = nn.Linear(context_dim, self.mixing_dim)
        self.value = nn.Sequential(nn.Linear(context_dim, self.mixing_dim), nn.ReLU(),
                                   nn.Linear(self.mixing_dim, 1))

    def forward(self, agent_qs, states, task_decomposer):
        batch, time, n_agents = agent_qs.shape
        ally_parts, other_parts, action_parts, timestep = task_decomposer.decompose_state(states)
        allies = _stack_entities(ally_parts, states, task_decomposer.state_nf_al)
        others = _stack_entities(other_parts, states, task_decomposer.state_nf_en)
        if hasattr(task_decomposer, "decompose_state_mask"):
            ally_valid, other_valid = task_decomposer.decompose_state_mask(states)
        else:
            ally_valid = allies.abs().sum(-1) > 0
            other_valid = others.abs().sum(-1) > 0
        if self.state_last_action:
            actions = _stack_entities(action_parts, states, task_decomposer.n_actions)
            allies = torch.cat((allies, compact_actions(actions, self.n_ego)), dim=-1)
        tokens = torch.cat((self.ally_encoder(allies), self.other_encoder(others)), dim=-2)
        valid = torch.cat((ally_valid, other_valid), dim=-1).bool()
        n_entities = tokens.shape[-2]
        tokens = tokens.reshape(batch * time, n_entities, self.entity_dim)
        valid = valid.reshape(batch * time, n_entities)
        tokens = tokens * valid.unsqueeze(-1)
        safe_valid = valid.clone()
        safe_valid[~valid.any(-1), 0] = True
        attended, _ = self.attention(tokens, tokens, tokens, key_padding_mask=~safe_valid,
                                     need_weights=False)
        contextual = self.norm(tokens + attended) * valid.unsqueeze(-1)
        pooled = contextual.sum(-2) / valid.sum(-1, keepdim=True).clamp_min(1)
        if self.timestep_dim:
            if timestep.shape[-1] != self.timestep_dim:
                raise ValueError("State timestep feature layout does not match mixer")
            pooled = torch.cat((pooled, timestep.reshape(batch * time, self.timestep_dim)), dim=-1)
        ally_context = contextual[:, :n_agents]
        local_context = torch.cat((ally_context, pooled.unsqueeze(1).expand(-1, n_agents, -1)), -1)
        w1 = self.hyper_w1(local_context).abs()
        b1 = self.hyper_b1(pooled).unsqueeze(1)
        hidden = F.elu(torch.bmm(agent_qs.reshape(batch * time, 1, n_agents), w1) + b1)
        w2 = self.hyper_w2(pooled).abs().unsqueeze(-1)
        total = torch.bmm(hidden, w2) + self.value(pooled).unsqueeze(-1)
        return total.reshape(batch, time, 1)
