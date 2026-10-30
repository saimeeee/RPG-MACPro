"""Safety and ordering checks for the server-side A/B priority handoff."""
import os
from pathlib import Path
import sys
import subprocess
import time

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import prioritize_rpg_b as handoff
from test_rpg_curriculum_migration import make_learner
from test_rpg_learner import batch


@pytest.fixture
def boundary(tmp_path):
    learner = make_learner(handoff.TASKS)
    for i, task in enumerate(handoff.TASKS[:2]):
        learner.begin_task(task, handoff.TASKS[i - 1] if i else None)
        learner.observe_return(.5)
        data = batch(learner.task2decomposer[task])
        learner.train(data, 8 * (i + 1), i + 1, task)
        learner.end_task(task, data)
    learner.progress = {'next_stage': 2, 'total_steps': 1500030, 'episode_num': 20000,
                        'boundary_rows': [], 'evaluation_env_states': {}}
    folder = tmp_path / 'boundary'
    learner.save_models(folder)
    config = dict(seed=0, tasks=handoff.TASKS, task_steps=handoff.BUDGETS)
    handoff.write_json(folder / 'driver.json', dict(config=config, progress=learner.progress))
    return folder, config, learner


def test_snapshot_preserves_full_state_and_can_train_next_map(boundary, tmp_path):
    source, config, learner = boundary
    item = dict(checkpoint=str(source), saved_checkpoint=str(tmp_path / 'saved'),
                config=config, target_stage=1)
    proof = handoff.preserve_checkpoint(item)
    assert proof['next_stage'] == 2
    for name in ('driver.json', 'learner.th'):
        assert (source / name).read_bytes() == (Path(item['saved_checkpoint']) / name).read_bytes()
    resumed = make_learner(handoff.TASKS)
    resumed.load_models(item['saved_checkpoint'])
    assert resumed.completed_tasks == handoff.TASKS[:2]
    for old in handoff.TASKS[:2]:
        torch.testing.assert_close(learner.mac.agent.generated_parameters(old),
                                   resumed.mac.agent.generated_parameters(old), rtol=0, atol=0)
    resumed.begin_task('4m', '13m')
    resumed.train(batch(resumed.task2decomposer['4m']), 1500040, 20001, '4m')
    assert resumed.train_steps == learner.train_steps + 1
    with pytest.raises(FileExistsError):
        handoff.preserve_checkpoint(item)


@pytest.mark.parametrize('bad', ['stage', 'config', 'learner', 'missing_optimizer'])
def test_bad_boundary_is_rejected(boundary, bad):
    source, config, _ = boundary
    if bad in ('stage', 'config'):
        driver = handoff.read_json(source / 'driver.json')
        if bad == 'stage':
            driver['progress']['next_stage'] = 1
        else:
            driver['config']['seed'] = 9
        handoff.write_json(source / 'driver.json', driver)
    else:
        payload = torch.load(source / 'learner.th', weights_only=True)
        if bad == 'learner':
            payload['completed_tasks'] = handoff.TASKS[:1]
        else:
            del payload['optimiser']
        torch.save(payload, source / 'learner.th')
    with pytest.raises(ValueError):
        handoff.validate_checkpoint(source, config, 1)


def b_manifest(root, status='running'):
    state = dict(tasks=handoff.TASKS, task_intervals=handoff.BUDGETS,
                 seeds={str(i): dict(status=status, pid=9000 + i) for i in range(3)})
    root.mkdir(exist_ok=True)
    handoff.write_json(root / 'manifest.json', state)
    return state


def finish_b(root, manifest, seed):
    manifest['seeds'][str(seed)].update(status='completed', exit_code=0)
    folder = root / ('seed_' + str(seed))
    folder.mkdir(exist_ok=True)
    handoff.write_json(folder / 'completion.json', dict(status='completed', seed=seed,
                       seen_tasks=handoff.TASKS, step=sum(handoff.BUDGETS)))
    handoff.write_json(root / 'manifest.json', manifest)


def test_b_requires_three_successful_completions_and_no_live_workers(tmp_path, monkeypatch):
    state = b_manifest(tmp_path)
    monkeypatch.setattr(handoff, 'identity', lambda pid: None)
    monkeypatch.setattr(handoff, 'group_members', lambda pid: {})
    for i in range(2):
        finish_b(tmp_path, state, i)
        assert not handoff.b_completed(tmp_path, state)
    state['seeds']['2']['status'] = 'failed'
    with pytest.raises(RuntimeError, match='failed'):
        handoff.b_completed(tmp_path, state)
    finish_b(tmp_path, state, 2)
    assert handoff.b_completed(tmp_path, state)
    monkeypatch.setattr(handoff, 'group_members', lambda pid: {9010: {}})
    assert not handoff.b_completed(tmp_path, state)


def test_never_signals_reused_pid(monkeypatch):
    expected = dict(start_ticks=10, pgid=100, command='authorized')
    monkeypatch.setattr(handoff, 'identity', lambda pid: dict(expected, start_ticks=11))
    monkeypatch.setattr(handoff.os, 'killpg', lambda *args: pytest.fail('Must not signal'), raising=False)
    with pytest.raises(RuntimeError, match='identity'):
        handoff.stop_group(100, expected)


@pytest.mark.skipif(os.name != 'posix', reason='Linux server process groups')
def test_stop_releases_real_worker_and_simulator_group():
    command = [sys.executable, '-c',
               'import subprocess,sys,time; subprocess.Popen([sys.executable,"-c","import time; time.sleep(60)"]); time.sleep(60)']
    child = subprocess.Popen(command, start_new_session=True)
    try:
        deadline = time.monotonic() + 5
        while len(handoff.group_members(child.pid)) < 2 and time.monotonic() < deadline:
            time.sleep(.05)
        assert len(handoff.group_members(child.pid)) == 2
        handoff.stop_group(child.pid, handoff.identity(child.pid))
        assert not handoff.group_members(child.pid)
        child.wait(timeout=5)
    finally:
        if child.poll() is None:
            os.killpg(child.pid, 9)
            child.wait()


def test_handoff_waits_each_boundary_and_all_b_before_resuming(boundary, tmp_path, monkeypatch):
    source, config, _ = boundary
    b_root = tmp_path / 'b'
    b = b_manifest(b_root)
    live = {100 + i: dict(start_ticks=i, pgid=100 + i, command='A') for i in range(3)}
    live[800] = dict(command='B launcher')
    state = dict(seeds={}, b_output=str(b_root), b_launcher_pid=800, b_launcher_identity=live[800])
    for seed in range(3):
        folder = tmp_path / ('source_' + str(seed))
        folder.mkdir()
        state['seeds'][str(seed)] = dict(status='waiting_for_boundary', config=config, target_stage=1,
            checkpoint=str(folder), saved_checkpoint=str(tmp_path / ('saved_' + str(seed))),
            source_pid=100 + seed, source_identity=live[100 + seed], cwd=str(tmp_path),
            source_sha256={}, environment={}, resume_output=str(tmp_path / ('resume_' + str(seed))),
            original_command=['python', 'src/main.py', '--checkpoint=old', '--output=old'])
    monkeypatch.setattr(handoff, 'identity', lambda pid: live.get(pid))
    monkeypatch.setattr(handoff, 'group_members', lambda pid: {})
    monkeypatch.setattr(handoff, 'source_hashes', lambda path: {})
    stopped, commands, children, handles = [], [], {}, {}
    def stop(pid, expected):
        assert live[pid] == expected
        stopped.append(pid)
        del live[pid]
    monkeypatch.setattr(handoff, 'stop_group', stop)
    class Child:
        def __init__(self, command, **kwargs):
            assert len(stopped) == 3
            assert all(r['status'] == 'completed' for r in b['seeds'].values())
            commands.append(command)
            self.pid = 200 + len(commands)
        def poll(self):
            return None
    monkeypatch.setattr(handoff.subprocess, 'Popen', Child)
    def tick():
        handoff.tick(state, lambda: None, children, handles)
    tick()
    assert not stopped and not commands
    for seed in range(3):
        dest = Path(state['seeds'][str(seed)]['checkpoint'])
        for name in ('driver.json', 'learner.th'):
            (dest / name).write_bytes((source / name).read_bytes())
        tick()
        assert len(stopped) == seed + 1
        assert not commands
    for seed in range(2):
        finish_b(b_root, b, seed)
        tick()
        assert not commands
    finish_b(b_root, b, 2)
    tick()
    assert len(commands) == 3 and state['status'] == 'a_resumed'
    assert all('--checkpoint=old' not in cmd and '--output=old' not in cmd for cmd in commands)
    tick()
    assert len(commands) == 3  # no duplicate launch on subsequent polling
    for handle in handles.values():
        handle.close()
