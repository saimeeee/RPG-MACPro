"""Offline tensor checks using MACPro's actual SC2 decomposition code.

These tests do not instantiate or emulate StarCraft II and are not game
experiments. AST extraction avoids importing the optional PySC2/protobuf/game
dependencies while executing the original SC2Decomposer class and reading the
original literal map registry from the checked-out MACPro source.
"""

import ast
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from modules.agents.rpg_agent import RPGAgent
from modules.mixers.rpg_mixer import RPGMixer


TRAIN_MAPS = ["5m_vs_6m", "12m", "5m", "8m_vs_9m"]
GENERALIZATION_MAPS = ["3m", "8m", "10m", "10m_vs_11m"]


@pytest.fixture(scope="module")
def original_decomposer_class():
    registry_file = ROOT / "src/envs/smac/smac/env/starcraft2/maps/smac_maps.py"
    registry_ast = ast.parse(registry_file.read_text(encoding="utf-8"))
    registry_node = next(
        node for node in registry_ast.body if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "map_param_registry"
                for target in node.targets)
    )
    map_registry = ast.literal_eval(registry_node.value)
    decomposer_file = ROOT / "src/modules/decomposers/sc2_decomposer.py"
    decomposer_ast = ast.parse(decomposer_file.read_text(encoding="utf-8"))
    class_node = next(node for node in decomposer_ast.body
                      if isinstance(node, ast.ClassDef) and node.name == "SC2Decomposer")
    namespace = {"th": torch, "np": np,
                 "get_map_params": lambda name: map_registry[name]}
    exec(compile(ast.Module(body=[class_node], type_ignores=[]),
                 str(decomposer_file), "exec"), namespace)
    return namespace["SC2Decomposer"]


def decomposer(cls, map_name, **updates):
    options = dict(map_name=map_name, obs_own_health=True, obs_all_health=True,
                   obs_instead_of_state=False, obs_last_action=False,
                   obs_pathing_grid=False, obs_terrain_height=False,
                   obs_timestep_number=False, state_last_action=True,
                   state_timestep_number=False)
    options.update(updates)
    return cls(SimpleNamespace(env_args=options))


def model_settings():
    return SimpleNamespace(rnn_hidden_dim=12, entity_embed_dim=8, attn_embed_dim=4,
                           rpg_mixer_heads=2, mixing_embed_dim=7,
                           rpg_task_embed_dim=5, rpg_hyper_hidden_dim=9,
                           rpg_head_hidden_dim=11)


@pytest.mark.parametrize("map_name", TRAIN_MAPS + GENERALIZATION_MAPS)
def test_original_smac_layout_accepts_variable_team_and_action_dimensions(
        original_decomposer_class, map_name):
    torch.manual_seed(29)
    # Only training task embeddings exist. Unseen map shapes must work with an
    # explicitly chosen trained embedding, without adding parameter tensors.
    known = {name: decomposer(original_decomposer_class, name) for name in TRAIN_MAPS}
    agent = RPGAgent(known, model_settings())
    agent.begin_task(TRAIN_MAPS[0])
    parameter_ids = [id(parameter) for parameter in agent.parameters()]
    current = decomposer(original_decomposer_class, map_name)
    agent.task2decomposer[map_name] = current
    mixer = RPGMixer(known[TRAIN_MAPS[0]], model_settings())

    batch = 2
    obs = torch.rand(batch * current.n_agents, current.obs_dim)
    last_actions = torch.zeros(batch * current.n_agents, current.n_actions)
    q, hidden, entropy = agent(obs, last_actions, torch.zeros(batch * current.n_agents, 12),
                               map_name, embedding_task=TRAIN_MAPS[0])
    assert q.shape == (batch * current.n_agents, 6 + current.n_enemies)
    assert hidden.shape == (batch * current.n_agents, 12)
    assert entropy.shape == (batch * current.n_agents,)
    assert [id(parameter) for parameter in agent.parameters()] == parameter_ids
    assert set(agent.task_keys) == set(TRAIN_MAPS)

    states = torch.rand(batch, 1, current.state_dim)
    states[0] = 0  # Wholly empty terminal state must remain numerically safe.
    totals = mixer(q[:, 0].reshape(batch, 1, current.n_agents), states, current)
    assert totals.shape == (batch, 1, 1)
    assert torch.isfinite(totals).all()
    (totals.square().sum() + entropy.mean() * 0.01).backward()
    for module in (agent.hypernetwork, agent.relation_capturer, mixer):
        gradients = [parameter.grad for parameter in module.parameters() if parameter.grad is not None]
        assert gradients and all(torch.isfinite(gradient).all() for gradient in gradients)
        assert sum(gradient.abs().sum().item() for gradient in gradients) > 0


@pytest.mark.parametrize("state_last_action", [False, True])
def test_original_smac_timestep_offset_respects_optional_action_state(
        original_decomposer_class, state_last_action):
    current = decomposer(original_decomposer_class, "5m", state_last_action=state_last_action,
                         state_timestep_number=True)
    state = torch.rand(2, 3, current.state_dim)
    state[..., -1] = 0.375
    allies, others, previous_actions, timestep = current.decompose_state(state)
    assert len(allies) == current.n_agents and len(others) == current.n_enemies
    assert len(previous_actions) == (current.n_agents if state_last_action else 0)
    assert timestep.shape == (2, 3, 1)
    torch.testing.assert_close(timestep, torch.full((2, 3, 1), 0.375))
    mixer = RPGMixer(current, model_settings())
    q = torch.rand(2, 3, current.n_agents, requires_grad=True)
    total = mixer(q, state, current)
    gradient = torch.autograd.grad(total.sum(), q)[0]
    assert torch.isfinite(total).all() and torch.isfinite(gradient).all()
    assert (gradient >= 0).all()
