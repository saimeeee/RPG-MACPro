"""Runner and command-line regressions for the continual RPG pipeline."""

import copy
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace as SN

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from components.episode_buffer import ReplayBuffer
from rpg_run import RPGRunner, load_config, make_scheme, run_experiment, training_clock


class EndingEnv:
    episode_limit = 4

    def __init__(self, timeout):
        self.timeout = timeout
        self.t = 0

    def reset(self):
        self.t = 0

    def get_obs(self):
        return np.full((2, 9), self.t + 1, dtype=np.float32)

    def get_state(self):
        return np.full(9, self.t + 1, dtype=np.float32)

    def get_avail_actions(self):
        return np.ones((2, 6), dtype=np.int32)

    def step(self, actions):
        self.t += 1
        done = self.t == 2
        return (7.0 if done else 0.0), done, {"episode_limit": self.timeout and done}


class ZeroMAC:
    def init_hidden(self, batch_size, task):
        pass

    def select_actions(self, batch, t_ep, t_env, task, **kwargs):
        return torch.zeros(1, 2, dtype=torch.long)


@pytest.mark.parametrize("timeout", [False, True])
def test_runner_preserves_final_reward_and_timeout_bootstrap(timeout):
    info = {"n_agents": 2, "n_actions": 6, "state_shape": 9, "obs_shape": 9}
    scheme, groups, preprocess = make_scheme(info)
    runner = RPGRunner(EndingEnv(timeout), "a", scheme, groups, preprocess, "cpu")
    episode, result = runner.run(ZeroMAC())
    assert result["return"] == 7
    assert result["length"] == 2
    assert episode["reward"][0, 1, 0].item() == 7
    assert episode["terminated"][0, 1, 0].item() == (0 if timeout else 1)
    assert episode["filled"][0, :, 0].tolist() == [1, 1, 1, 0, 0]
    assert episode["state"][0, 2, 0].item() == 3  # Final next state was retained.

    replay = ReplayBuffer(scheme, groups, 2, 5, preprocess=preprocess)
    replay.insert_episode_batch(episode)
    sample = replay.sample(1)
    sample = sample[:, :int(sample.max_t_filled().item())]
    mask = sample["filled"][:, :-1] * sample["filled"][:, 1:]
    assert sample.max_seq_length == 3
    assert mask.sum().item() == 2
    assert (sample["reward"][:, :-1] * mask).sum().item() == 7


def cli(output, checkpoint=None, evaluate=False):
    return SN(config="qmix_rpg", task_config="rpg_lbf_smoke", seed=None,
              device="cpu", test_nepisode=None, steps_per_task=None,
              set=[], checkpoint=str(checkpoint) if checkpoint else None,
              output=str(output), evaluate=evaluate, zero_shot_embedding=None)


@pytest.fixture(scope="module")
def trained_run(tmp_path_factory):
    folder = tmp_path_factory.mktemp("driver")
    command = cli(folder / "original")
    config = load_config(command)
    config.update(seed=37, test_nepisode=1, test_interval=100,
                  rpg_importance_samples=1, rpg_return_probe_episodes=1,
                  task_steps=[12, 12], batch_size=2, buffer_size=4,
                  rnn_hidden_dim=8, entity_embed_dim=8, attn_embed_dim=4,
                  mixing_embed_dim=4, rpg_task_embed_dim=4,
                  rpg_hyper_hidden_dim=8, rpg_head_hidden_dim=8,
                  rpg_mixer_heads=2, target_update_interval=2)
    for spec in config["tasks"]:
        spec["max_episode_steps"] = 4
    config["generalization_tasks"] = [dict(config["tasks"][-1], name="unseen_4p", players=4)]
    output = run_experiment(config, command)
    return folder, output


def _assert_tensor_tree_equal(first, second):
    if isinstance(first, torch.Tensor):
        torch.testing.assert_close(first, second, rtol=0, atol=0)
    elif isinstance(first, dict):
        assert first.keys() == second.keys()
        for key in first:
            _assert_tensor_tree_equal(first[key], second[key])
    elif isinstance(first, (list, tuple)):
        assert len(first) == len(second)
        for a, b in zip(first, second):
            _assert_tensor_tree_equal(a, b)
    else:
        assert first == second


def test_task_boundary_resume_reproduces_training_parameters(trained_run):
    folder, original = trained_run
    command = cli(folder / "resumed", original / "checkpoints/task_00")
    output = run_experiment(load_config(command), command)
    reference = torch.load(original / "checkpoints/task_01/learner.th", weights_only=True)
    resumed = torch.load(output / "checkpoints/task_01/learner.th", weights_only=True)
    for key in ("agent", "mixer", "target_agent", "target_mixer", "optimiser",
                "relation_anchors", "relation_importance", "hyper_targets", "torch_rng"):
        _assert_tensor_tree_equal(reference[key], resumed[key])
    assert reference["train_steps"] == resumed["train_steps"]
    assert reference["completed_tasks"] == resumed["completed_tasks"]
    assert reference["progress"]["evaluation_env_states"] == resumed["progress"]["evaluation_env_states"]
    assert reference["progress"]["boundary_rows"] == resumed["progress"]["boundary_rows"]


def test_checkpoint_rejects_changed_seed(trained_run):
    folder, original = trained_run
    command = cli(folder / "bad_seed", original / "checkpoints/task_00")
    command.seed = 99
    with pytest.raises(ValueError, match="original seed"):
        load_config(command)


def test_unseen_task_uses_explicit_trained_embedding(trained_run):
    folder, original = trained_run
    command = cli(folder / "zero_shot", original / "checkpoints/task_01", evaluate=True)
    command.zero_shot_embedding = "lbf_3p_2f"
    output = run_experiment(load_config(command), command)
    rows = [json.loads(line) for line in (output / "evaluations.jsonl").read_text().splitlines()]
    assert len(rows) == 1
    assert rows[0]["task"] == "unseen_4p"
    assert rows[0]["embedding_task"] == "lbf_3p_2f"
    assert rows[0]["stage"] == "zero_shot"
    assert not (output / "updates.jsonl").exists()


def test_main_cli_evaluates_a_saved_boundary_without_training(trained_run):
    folder, original = trained_run
    destination = folder / "cli_evaluation"
    result = subprocess.run(
        [sys.executable, str(ROOT / "src/main.py"), "--config=qmix_rpg", "--evaluate",
         "--checkpoint", str(original / "checkpoints/task_01"),
         "--output", str(destination), "--test-nepisode", "1"],
        cwd=ROOT, capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    summary = json.loads((destination / "summary.json").read_text())
    rows = [json.loads(line) for line in (destination / "evaluations.jsonl").read_text().splitlines()]
    assert summary["mode"] == "evaluation"
    assert summary["tasks"] == ["lbf_2p_1f", "lbf_3p_2f"]
    assert len(rows) == 2
    assert all(row["stage"] == "checkpoint" for row in rows)
    assert not (destination / "updates.jsonl").exists()


def test_exploration_clock_is_explicit_and_legacy_compatible():
    assert training_clock({}, 100, 1000100) == 100
    assert training_clock({'reset_epsilon': True}, 100, 1000100) == 100
    assert training_clock({'reset_epsilon': False}, 100, 1000100) == 1000100


@pytest.mark.parametrize('option', ['obs_agent_id=true', 'reset_epsilon=1',
                                  'rpg_return_calibration=unsupported'])
def test_unsupported_or_ambiguous_protocol_config_is_rejected(tmp_path, option):
    command = cli(tmp_path / 'invalid')
    command.set = [option]
    with pytest.raises(ValueError):
        load_config(command)


def test_legacy_checkpoints_keep_the_same_training_protocol(trained_run):
    folder, original = trained_run
    config = json.loads((original / 'config.json').read_text())
    config.pop('reset_epsilon')
    config.pop('rpg_return_calibration')
    output = run_experiment(config, cli(folder / 'legacy_keys'))
    reference = torch.load(original / 'checkpoints/task_01/learner.th', weights_only=True)
    legacy = torch.load(output / 'checkpoints/task_01/learner.th', weights_only=True)
    for key in ('agent', 'mixer', 'optimiser', 'relation_importance', 'torch_rng', 'return_samples'):
        _assert_tensor_tree_equal(reference[key], legacy[key])


def test_training_episode_calibration_adds_no_extra_greedy_rollouts(trained_run):
    folder, original = trained_run
    config = json.loads((original / 'config.json').read_text())
    config.update(rpg_return_calibration='training_episodes', reset_epsilon=False,
                  rpg_return_probe_episodes=2)
    output = run_experiment(config, cli(folder / 'training_calibration'))
    episodes = [json.loads(x) for x in (output / 'episodes.jsonl').read_text().splitlines()]
    probes = [json.loads(x) for x in (output / 'calibration.jsonl').read_text().splitlines()]
    summary = json.loads((output / 'summary.json').read_text())
    assert all(x['phase'] == 'training' for x in episodes)
    assert sum(x['length'] for x in episodes) == summary['environment_steps']
    assert len(probes) == 4
    for spec in config['tasks']:
        task = spec['name']
        first = [x for x in episodes if x['task'] == task][:2]
        actual = [x for x in probes if x['task'] == task]
        assert [x['return'] for x in actual] == [x['return'] for x in first]
        assert [x['complete'] for x in actual] == [False, True]
        expected = np.mean([np.clip((x['return'] - config['return_min']) /
                                   (config['return_max'] - config['return_min']), 0, 1) for x in first])
        assert actual[-1]['gamma_p'] == pytest.approx(expected)
    runtime = json.loads((output / 'runtime.json').read_text())
    assert 'src/controllers/rpg_controller.py' in runtime['source_sha256']
    assert 'src/components/episode_buffer.py' in runtime['source_sha256']
    assert runtime['model_parameters']['agent_total'] > 0
    updates = [json.loads(x) for x in (output / 'updates.jsonl').read_text().splitlines()]
    for row in updates:
        assert row['loss'] == pytest.approx(row['td_loss'] + row['weighted_attention_loss'] +
                                          row['weighted_relation_loss'] + row['weighted_hyper_loss'], abs=1e-6)
