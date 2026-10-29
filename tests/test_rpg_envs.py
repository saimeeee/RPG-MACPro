"""Tests use the real official LBF environment, not a substituted simulator."""

from pathlib import Path
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rpg_envs import LBFAdapter, LBFDecomposer, build_env


def task(**overrides):
    result = dict(name="lbf-test", players=2, field_size=[8, 8], max_food=2,
                  sight=2, max_episode_steps=5, force_coop=False, max_player_level=3)
    result.update(overrides)
    return result


def test_lbf_real_environment_reproducible_sequence_and_shapes():
    first, dec = build_env(task(), "lbf", seed=14)
    second, _ = build_env(task(), "lbf", seed=14)
    try:
        states = []
        for _ in range(2):
            obs, state = first.reset()
            obs2, state2 = second.reset()
            np.testing.assert_array_equal(obs, obs2)
            np.testing.assert_array_equal(state, state2)
            assert obs.shape == (2, 12)
            assert state.shape == (12,)
            assert np.isfinite(obs).all() and np.isfinite(state).all()
            assert first.get_avail_actions().shape == (2, 6)
            states.append(state)
        assert not np.array_equal(states[0], states[1])
        assert dec.n_enemies == 2 and dec.n_interaction_actions == 0
    finally:
        first.close()
        second.close()


def test_lbf_timeout_bootstraps_but_success_does_not():
    env = LBFAdapter(task(max_episode_steps=1), seed=6)
    try:
        env.reset()
        reward, done, info = env.step([0, 0])
        assert reward == 0 and done and info["episode_limit"]
        assert not info["battle_won"]
        env.reset()
        env._env.field[:] = 0
        reward, done, info = env.step([0, 0])
        assert done and not info["episode_limit"] and info["battle_won"]
    finally:
        env.close()


def test_lbf_local_observation_does_not_expose_hidden_entities():
    env = LBFAdapter(task(sight=1), seed=11)
    try:
        env.reset()
        # Far apart players and food are visible centrally, but not to player 0.
        env._env.players[0].position = (0, 0)
        env._env.players[1].position = (7, 7)
        env._env.field[:] = 0
        env._env.field[5, 5] = 2
        env._env._gen_valid_moves()
        env.step([0, 0])
        obs = torch.from_numpy(env.get_obs())
        ally_mask, food_mask = env.decomposer.decompose_obs_mask(obs)
        assert not ally_mask[0].any() and not food_mask[0].any()
        assert torch.count_nonzero(obs[0, 3:]) == 0
        players, foods = env.decomposer.decompose_state_mask(
            torch.from_numpy(env.get_state()).view(1, 1, -1))
        assert players.all() and foods.sum() == 1
    finally:
        env.close()


def test_lbf_sixth_action_really_loads_food_and_sums_team_reward():
    env = LBFAdapter(task(max_food=1, force_coop=True), seed=8)
    try:
        env.reset()
        env._env.players[0].position = (3, 2)
        env._env.players[1].position = (3, 4)
        for player in env._env.players:
            player.level = 1
        env._env.field[:] = 0
        env._env.field[3, 3] = 2
        env._env._food_spawned = 2
        env._env._gen_valid_moves()
        reward, done, info = env.step([5, 5])
        assert reward == pytest.approx(1.0)
        assert done and info["battle_won"] and not info["episode_limit"]
        assert not env.decomposer.decompose_obs_mask(torch.from_numpy(env.get_obs()))[1].any()
    finally:
        env.close()


def test_decomposer_supports_single_agent_and_arbitrary_batch_axes():
    dec = LBFDecomposer(1, 2)
    obs = torch.tensor([[1., 1., 1., 0., 0., .5, 0., 0., 0.]])
    own, foods, allies = dec.decompose_obs(obs)
    assert own.shape == (1, 3) and len(foods) == 2 and allies == []
    ally_mask, food_mask = dec.decompose_obs_mask(obs)
    assert ally_mask.shape == (1, 0) and food_mask.tolist() == [[True, False]]
    actions = torch.zeros(2, 3, 6)
    actions[..., 5] = 1
    ego, interaction, compact = dec.decompose_action_info(actions)
    assert ego.shape == (2, 3, 6) and compact.shape == (2, 3, 7)
    assert interaction.shape == (2, 3, 2) and not interaction.any()
    assert not compact[..., -1].any()


def test_sc2_missing_installation_fails_before_importing_game_dependencies(monkeypatch, tmp_path):
    monkeypatch.setenv("SC2PATH", str(tmp_path / "missing"))
    with pytest.raises(FileNotFoundError, match="SC2PATH"):
        build_env("5m_vs_6m", "sc2", seed=0)
