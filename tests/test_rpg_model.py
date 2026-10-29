"""Behavioral contracts of the paper's entity-scalable RPG architecture."""

from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from modules.agents.rpg_agent import RPGAgent
from modules.mixers.rpg_mixer import RPGMixer


class EntityDecomposer:
    """A small entity layout, independent of the LBF/SMAC installations."""

    own_obs_dim = obs_nf_al = obs_nf_en = state_nf_al = state_nf_en = 3
    n_actions_no_attack = 6
    state_last_action = state_timestep_number = False
    timestep_number_state_dim = 0

    def __init__(self, n_agents, n_enemies, interaction_actions=True):
        self.n_agents, self.n_enemies = n_agents, n_enemies
        self.n_actions = 6 + (n_enemies if interaction_actions else 0)
        self.obs_dim = 3 * (n_agents + n_enemies)

    def decompose_obs(self, obs):
        own = obs[:, :3]
        allies = [obs[:, 3 + 3 * i:6 + 3 * i] for i in range(self.n_agents - 1)]
        start = 3 * self.n_agents
        others = [obs[:, start + 3 * i:start + 3 * (i + 1)] for i in range(self.n_enemies)]
        return own, others, allies

    def decompose_obs_mask(self, obs):
        entities = obs.reshape(*obs.shape[:-1], self.n_agents + self.n_enemies, 3)
        return entities[:, 1:self.n_agents, 2] > 0, entities[:, self.n_agents:, 2] > 0

    def decompose_state(self, states):
        allies = [states[..., 3 * i:3 * (i + 1)] for i in range(self.n_agents)]
        start = 3 * self.n_agents
        others = [states[..., start + 3 * i:start + 3 * (i + 1)] for i in range(self.n_enemies)]
        return allies, others, [], states[..., :0]

    def decompose_state_mask(self, states):
        entities = states.reshape(*states.shape[:-1], self.n_agents + self.n_enemies, 3)
        return entities[..., :self.n_agents, 2] > 0, entities[..., self.n_agents:, 2] > 0


def settings(**updates):
    values = dict(rnn_hidden_dim=12, entity_embed_dim=8, attn_embed_dim=4,
                  rpg_task_embed_dim=5, rpg_hyper_hidden_dim=9, rpg_head_hidden_dim=11,
                  rpg_mixer_heads=2, mixing_embed_dim=7, rpg_alpha_init=0.7)
    values.update(updates)
    return SimpleNamespace(**values)


def visible_observations(decomposer, batch=2):
    obs = torch.randn(batch, decomposer.obs_dim)
    obs.reshape(batch, -1, 3)[..., 2] = 1
    return obs


def test_entity_permutations_preserve_relations_and_permute_target_actions():
    torch.manual_seed(1)
    decomp = EntityDecomposer(4, 3)
    agent = RPGAgent({"task": decomp}, settings())
    agent.begin_task("task")
    obs = visible_observations(decomp)
    previous = torch.zeros(2, decomp.n_actions)
    previous[:, 7] = 1  # A target-specific last action must move with the entity.
    hidden = torch.randn(2, 12)
    q, h, entropy = agent(obs, previous, hidden, "task")
    ally_order = torch.tensor([2, 0, 1])
    other_order = torch.tensor([1, 2, 0])
    entities = obs.reshape(2, 7, 3)
    permuted = torch.cat((entities[:, :1], entities[:, 1:4][:, ally_order],
                          entities[:, 4:][:, other_order]), dim=1).reshape(2, -1)
    permuted_previous = torch.cat((previous[:, :6], previous[:, 6:][:, other_order]), dim=-1)
    pq, ph, pe = agent(permuted, permuted_previous, hidden, "task")
    torch.testing.assert_close(ph, h)
    torch.testing.assert_close(pe, entropy)
    torch.testing.assert_close(pq[:, :6], q[:, :6])
    torch.testing.assert_close(pq[:, 6:], q[:, 6:][:, other_order])


@pytest.mark.parametrize("n_agents,n_others", [(1, 0), (1, 3), (4, 0), (4, 3)])
def test_empty_or_invisible_entity_sets_are_finite_with_zero_entropy(n_agents, n_others):
    decomp = EntityDecomposer(n_agents, n_others)
    agent = RPGAgent({"task": decomp}, settings())
    agent.begin_task("task")
    obs = torch.zeros(2, decomp.obs_dim)
    q, h, entropy = agent(obs, None, torch.zeros(2, 12), "task")
    assert q.shape == (2, decomp.n_actions)
    assert torch.isfinite(q).all() and torch.isfinite(h).all()
    assert torch.equal(entropy, torch.zeros_like(entropy))
    (q.square().sum() + h.square().sum() + entropy.sum()).backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in agent.parameters())


def test_masked_entities_cannot_change_relation_history_or_ego_q():
    decomp = EntityDecomposer(3, 2)
    agent = RPGAgent({"task": decomp}, settings())
    agent.begin_task("task")
    obs = torch.zeros(2, decomp.obs_dim)
    altered = obs.reshape(2, -1, 3).clone()
    altered[:, 1:, :2] = torch.randn_like(altered[:, 1:, :2]) * 100
    first = agent(obs, None, torch.zeros(2, 12), "task")
    second = agent(altered.reshape(2, -1), None, torch.zeros(2, 12), "task")
    torch.testing.assert_close(first[1], second[1])
    torch.testing.assert_close(first[0][:, :6], second[0][:, :6])


def test_shared_model_accepts_changing_entity_and_action_counts_without_new_weights():
    tasks = {"small": EntityDecomposer(2, 1), "large": EntityDecomposer(5, 4)}
    agent = RPGAgent(tasks, settings())
    parameter_ids = [id(p) for p in agent.parameters()]
    for task, decomp in tasks.items():
        agent.begin_task(task)
        q, h, _ = agent(visible_observations(decomp), None, torch.zeros(2, 12), task)
        assert q.shape == (2, decomp.n_actions) and h.shape == (2, 12)
        assert [id(p) for p in agent.parameters()] == parameter_ids


def test_lbf_food_entities_do_not_create_extra_actions():
    decomp = EntityDecomposer(3, 4, interaction_actions=False)
    agent = RPGAgent({"foraging": decomp}, settings())
    agent.begin_task("foraging")
    q, _, _ = agent(visible_observations(decomp), None, torch.zeros(2, 12), "foraging")
    assert q.shape == (2, 6)


def test_generated_heads_backpropagate_into_hypernetwork_embedding_and_capturer():
    torch.manual_seed(2)
    decomp = EntityDecomposer(3, 2)
    agent = RPGAgent({"old": decomp, "new": decomp}, settings())
    agent.begin_task("old")
    old = agent.task_embeddings[agent.task_keys["old"]].detach().clone()
    gaussian = agent.task_embeddings[agent.task_keys["new"]].detach().clone()
    agent.begin_task("new", "old")
    current = agent.task_embeddings[agent.task_keys["new"]]
    torch.testing.assert_close(current, 0.7 * old + 0.3 * gaussian)
    q, _, entropy = agent(visible_observations(decomp), None, torch.zeros(2, 12), "new")
    (q.square().sum() + 0.01 * entropy.sum()).backward()
    assert current.grad is not None and current.grad.abs().sum() > 0
    assert agent.task_embeddings[agent.task_keys["old"]].grad is None
    for module in (agent.hypernetwork, agent.relation_capturer):
        assert sum(float(p.grad.abs().sum()) for p in module.parameters() if p.grad is not None) > 0


def test_state_dict_preserves_task_identity_and_frozen_embeddings():
    tasks = {"a": EntityDecomposer(2, 1), "b": EntityDecomposer(4, 3)}
    agent = RPGAgent(tasks, settings())
    agent.begin_task("a")
    agent.begin_task("b")
    restored = RPGAgent(tasks, settings())
    restored.load_state_dict(agent.state_dict())
    assert restored.active_task == "b" and restored.begun_tasks == ["a", "b"]
    assert not restored.task_embeddings[restored.task_keys["a"]].requires_grad
    assert restored.task_embeddings[restored.task_keys["b"]].requires_grad
    for task in tasks:
        torch.testing.assert_close(restored.generated_parameters(task), agent.generated_parameters(task))


@pytest.mark.parametrize("n_agents,n_others", [(1, 0), (2, 1), (5, 4)])
def test_mixer_is_monotone_and_supports_variable_entity_counts(n_agents, n_others):
    torch.manual_seed(3)
    base = EntityDecomposer(2, 1)
    decomp = EntityDecomposer(n_agents, n_others)
    mixer = RPGMixer(base, settings())
    states = torch.randn(2, 3, (n_agents + n_others) * 3)
    states.reshape(2, 3, -1, 3)[..., 2] = 1
    states[0, 0] = 0  # Terminal state with no surviving entities.
    qs = torch.randn(2, 3, n_agents, requires_grad=True)
    totals = mixer(qs, states, decomp)
    derivative = torch.autograd.grad(totals.sum(), qs)[0]
    assert totals.shape == (2, 3, 1) and torch.isfinite(totals).all()
    assert torch.isfinite(derivative).all() and (derivative >= 0).all()
    torch.testing.assert_close(mixer(qs + 0.5, states, decomp).clamp_min(totals),
                               mixer(qs + 0.5, states, decomp))


def test_mixer_is_invariant_to_joint_state_and_q_entity_permutations():
    torch.manual_seed(4)
    decomp = EntityDecomposer(3, 2)
    mixer = RPGMixer(decomp, settings())
    state = torch.randn(2, 4, 5, 3)
    state[..., 2] = 1
    qs = torch.randn(2, 4, 3)
    ally_order, enemy_order = [2, 0, 1], [1, 0]
    permuted = torch.cat((state[:, :, :3][:, :, ally_order],
                          state[:, :, 3:][:, :, enemy_order]), dim=2)
    original_total = mixer(qs, state.reshape(2, 4, -1), decomp)
    permuted_total = mixer(qs[:, :, ally_order], permuted.reshape(2, 4, -1), decomp)
    torch.testing.assert_close(original_total, permuted_total, atol=1e-5, rtol=1e-5)
