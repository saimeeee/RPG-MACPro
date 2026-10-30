"""Changing unseen map names must preserve all learned/optimizer/random state."""
import copy
import json
import os
import shutil
from pathlib import Path
import sys
from types import SimpleNamespace as NS

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from migrate_rpg_curriculum import BUDGETS, NEW_TASKS, OLD_TASKS, assert_equal, migrate, transform
from test_rpg_learner import TinyDecomposer, args, batch
from controllers.rpg_controller import RPGMAC
from learners.rpg_learner import RPGQLearner


def make_learner(names):
    config = args()
    decomposers = {name: TinyDecomposer(2 + i, 1 + i) for i, name in enumerate(names)}
    task_args = {name: NS(n_agents=d.n_agents, obs_last_action=True) for name, d in decomposers.items()}
    return RPGQLearner(RPGMAC(names, {}, task_args, config, decomposers), None, config)


@pytest.fixture
def source(tmp_path):
    learner = make_learner(OLD_TASKS)
    task = OLD_TASKS[0]
    learner.begin_task(task)
    learner.observe_return(.5)
    data = batch(learner.task2decomposer[task])
    learner.train(data, 8, 1, task)
    learner.end_task(task, data)
    # Synthetic counters model the first full-budget boundary; no simulated benchmark result is claimed.
    learner.progress = dict(next_stage=1, total_steps=1000029, episode_num=10000,
                            boundary_rows=[dict(task=task, step=1000029)], evaluation_env_states={})
    folder = tmp_path / 'source'
    learner.save_models(folder)
    driver = {'config': {'env': 'sc2', 'tasks': OLD_TASKS, 'task_steps': BUDGETS,
                         'seed': 7, 'buffer_size': 5000, 'lr': .001}, 'progress': learner.progress}
    (folder / 'driver.json').write_text(json.dumps(driver))
    return folder, driver, torch.load(folder / 'learner.th', weights_only=True)


def test_migration_preserves_weights_optimizer_rng_and_old_task_output(source, tmp_path):
    folder, driver, payload = source
    original_bytes = (folder / 'learner.th').read_bytes()
    destination = tmp_path / 'converted'
    provenance = migrate(folder, destination, 7)
    converted = torch.load(destination / 'learner.th', weights_only=True)
    assert provenance['tensor_optimizer_target_rng_equality_verified']
    assert (folder / 'learner.th').read_bytes() == original_bytes
    for name in ('optimiser', 'mixer', 'target_mixer', 'relation_anchors', 'relation_importance',
                 'hyper_targets', 'torch_rng', 'cuda_rng', 'numpy_rng', 'python_rng', 'progress'):
        assert_equal(payload[name], converted[name])
    restored = make_learner(NEW_TASKS)
    restored.load_models(destination)
    original = make_learner(OLD_TASKS)
    original.load_models(folder)
    torch.testing.assert_close(original.mac.agent.generated_parameters(OLD_TASKS[0]),
                               restored.mac.agent.generated_parameters(NEW_TASKS[0]), rtol=0, atol=0)
    # The restored optimizer can continue on the new second task without resetting its step count.
    restored.begin_task('13m')
    restored.train(batch(restored.task2decomposer['13m']), 1000035, 10001, '13m')
    assert restored.train_steps == payload['train_steps'] + 1
    assert restored.mac.agent.task_keys['13m'] == 't1'
    assert restored.mac.agent.task_keys['4m'] == 't2'
    with pytest.raises(FileExistsError):
        migrate(folder, destination, 7)


@pytest.mark.parametrize('change', ['stage', 'trained_future', 'seed', 'buffer', 'steps', 'agent_future'])
def test_converter_rejects_incompatible_or_already_trained_tasks(source, change):
    _, driver, payload = source
    driver, payload = copy.deepcopy(driver), copy.deepcopy(payload)
    if change == 'stage': driver['progress']['next_stage'] = 2
    elif change == 'trained_future': payload['completed_tasks'].append('12m')
    elif change == 'seed': driver['config']['seed'] = 8
    elif change == 'buffer': driver['config']['buffer_size'] = 256
    elif change == 'steps': driver['config']['task_steps'][0] = 500000
    elif change == 'agent_future': payload['agent']['_extra_state']['begun_tasks'].append('12m')
    with pytest.raises((ValueError, AssertionError)):
        transform(driver, payload, 7)


@pytest.mark.skipif(os.name != 'posix', reason='Server supervisor uses Linux process groups')
def test_supervisor_converts_each_seed_and_tracks_success_without_touching_other_jobs(source, tmp_path, monkeypatch):
    import transition_rpg_marines as supervisor
    folder, driver, _ = source
    root, control = tmp_path / 'experiment', tmp_path / 'control'
    identities, stopped, commands = {}, [], []
    manifest = {'settings': {'tasks': OLD_TASKS, 'task_steps': BUDGETS}, 'seeds': {}}
    for seed in (0, 1, 2):
        attempt = root / ('seed_' + str(seed)) / 'attempts/attempt_000'
        checkpoint = attempt / 'checkpoints/task_00'
        shutil.copytree(folder, checkpoint)
        seed_driver = copy.deepcopy(driver)
        seed_driver['config']['seed'] = seed
        (checkpoint / 'driver.json').write_text(json.dumps(seed_driver))
        pid = 1000 + seed
        manifest['seeds'][str(seed)] = {'attempts': [{'path': str(attempt), 'pid': pid}]}
        identities[pid] = {'start_ticks': seed, 'command': 'python src/main.py --output=' + str(attempt), 'pgid': pid}
    (root / 'manifest.json').write_text(json.dumps(manifest))
    monkeypatch.setattr(supervisor, 'identity', lambda pid: identities.get(pid))
    monkeypatch.setattr(supervisor, 'stop_source', lambda pid, expected: stopped.append((pid, expected)))
    monkeypatch.setattr(supervisor.time, 'sleep', lambda seconds: None)

    class Child:
        def __init__(self, command, **kwargs):
            commands.append(command)
            self.pid = 2000 + len(commands)
            output = Path(next(arg.split('=', 1)[1] for arg in command if arg.startswith('--output=')))
            checkpoint = Path(next(arg.split('=', 1)[1] for arg in command if arg.startswith('--checkpoint=')))
            assert json.loads((checkpoint / 'driver.json').read_text())['config']['tasks'] == NEW_TASKS
            output.mkdir()
            (output / 'summary.json').write_text(json.dumps({'completed_tasks': NEW_TASKS, 'environment_steps': sum(BUDGETS)}))

        def poll(self):
            return 0

    monkeypatch.setattr(supervisor.subprocess, 'Popen', Child)
    monkeypatch.setattr(sys, 'argv', ['transition', '--source-output', str(root), '--control', str(control)])
    assert supervisor.main() == 0
    result = json.loads((control / 'manifest.json').read_text())
    assert result['status'] == 'completed'
    assert all(item['status'] == 'completed' for item in result['seeds'].values())
    assert [pid for pid, _ in stopped] == [1000, 1001, 1002]
    assert len(commands) == 3
    assert all('--task-config=rpg_marines_macpro' in command for command in commands)


@pytest.mark.skipif(os.name != 'posix', reason='Server supervisor uses Linux process groups')
def test_supervisor_refuses_reused_or_mismatched_process_id(monkeypatch):
    import transition_rpg_marines as supervisor
    expected = {'start_ticks': 10, 'command': 'authorized', 'pgid': 99}
    monkeypatch.setattr(supervisor, 'identity', lambda pid: dict(expected, start_ticks=11))
    def forbidden_signal(*args):
        pytest.fail('A mismatched process must never be signalled')
    monkeypatch.setattr(supervisor.os, 'killpg', forbidden_signal)
    with pytest.raises(RuntimeError, match='identity'):
        supervisor.stop_source(99, expected)
