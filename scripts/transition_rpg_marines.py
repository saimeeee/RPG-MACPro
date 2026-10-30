#!/usr/bin/env python3
"""At each seed's first saved boundary, continue with the requested Marine maps."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from migrate_rpg_curriculum import BUDGETS, NEW_TASKS, OLD_TASKS, migrate

ROOT = Path(__file__).resolve().parents[1]


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_json(path, payload):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(payload, indent=2), encoding='utf-8')
    temporary.replace(path)


def identity(pid):
    """A PID alone is not sufficient authorization to stop a process."""
    try:
        proc = Path('/proc') / str(pid)
        fields = (proc / 'stat').read_text().rsplit(')', 1)[1].split()
        if fields[0] == 'Z':
            return None
        return {'start_ticks': int(fields[19]),
                'command': (proc / 'cmdline').read_bytes().replace(b'\0', b' ').decode(),
                'pgid': os.getpgid(pid)}
    except (OSError, ValueError):
        return None


def stop_source(pid, expected):
    actual = identity(pid)
    if actual is None:
        return
    if actual != expected or actual['pgid'] != pid:
        raise RuntimeError('Process identity/group changed; refusing to signal it')
    os.killpg(pid, signal.SIGTERM)
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        try:
            os.killpg(pid, 0)
        except ProcessLookupError:
            return
        time.sleep(.2)
    # This is still the process group whose identity was validated above.
    try:
        os.killpg(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-output', required=True, type=Path)
    parser.add_argument('--control', required=True, type=Path)
    args = parser.parse_args()
    args.control.mkdir(parents=True, exist_ok=True)
    lock = (args.control / '.lock').open('a+')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    status_path = args.control / 'manifest.json'
    if status_path.exists():
        raise RuntimeError('Existing transition; inspect it instead of starting a duplicate')
    original = read_json(args.source_output / 'manifest.json')
    if original['settings']['tasks'] != OLD_TASKS or original['settings']['task_steps'] != BUDGETS:
        raise ValueError('Unexpected source experiment protocol')
    write_json(args.control / 'original_manifest.json', original)
    state = {'status': 'waiting_for_first_task', 'pid': os.getpid(), 'created': time.time(),
             'source_output': str(args.source_output), 'tasks': NEW_TASKS, 'task_steps': BUDGETS,
             'checkpoint_policy': 'preserve shared first task; replace only untrained map identities',
             'seeds': {}}
    for seed in (0, 1, 2):
        attempt = original['seeds'][str(seed)]['attempts'][-1]
        source = Path(attempt['path'])
        info = identity(attempt['pid'])
        if info is None or str(source) not in info['command'] or 'src/main.py' not in info['command']:
            raise RuntimeError('Source training process did not match the recorded attempt')
        state['seeds'][str(seed)] = {
            'status': 'waiting_for_first_task', 'source_attempt': str(source),
            'source_pid': attempt['pid'], 'source_identity': info,
            'checkpoint': str(source / 'checkpoints/task_00'),
            'converted': str(args.control / ('seed_' + str(seed)) / 'checkpoint'),
            # Preserve the recognizable output prefix used by the other run's resource scheduler.
            'destination': str(args.source_output / ('seed_' + str(seed)) / 'marines/attempt_000'),
        }
    write_json(status_path, state)
    write_json(args.source_output / 'marines_transition.json', {
        'manifest': str(status_path), 'tasks': NEW_TASKS, 'task_steps': BUDGETS,
        'note': 'The original manifest describes superseded source attempts; this transition manifest tracks continuation.'})
    children, logs = {}, {}
    print('Installed boundary transition for seeds 0, 1, 2', flush=True)
    while True:
        for key, item in state['seeds'].items():
            seed = int(key)
            if item['status'] == 'waiting_for_first_task':
                checkpoint = Path(item['checkpoint'])
                if (checkpoint / 'driver.json').exists() and (checkpoint / 'learner.th').exists():
                    try:
                        # driver.json is written after the atomic learner.th; wait for valid JSON.
                        read_json(checkpoint / 'driver.json')
                    except json.JSONDecodeError:
                        continue
                    try:
                        provenance = migrate(checkpoint, item['converted'], seed)
                        item.update(status='checkpoint_verified', migration=provenance)
                        write_json(status_path, state)
                        stop_source(item['source_pid'], item['source_identity'])
                        destination = Path(item['destination'])
                        if destination.exists():
                            raise FileExistsError('Continuation output already exists')
                        destination.parent.mkdir(parents=True, exist_ok=True)
                        command = [sys.executable, '-u', str(ROOT / 'src/main.py'),
                                   '--config=qmix_rpg', '--task-config=rpg_marines_macpro', '--device=cuda',
                                   '--checkpoint=' + item['converted'], '--output=' + str(destination)]
                        log_path = destination.with_suffix('.log')
                        logs[seed] = log_path.open('wb')
                        child = subprocess.Popen(command, cwd=ROOT, stdout=logs[seed],
                                                 stderr=subprocess.STDOUT, start_new_session=True)
                        children[seed] = child
                        item.update(status='running', pid=child.pid, command=command,
                                    started=time.time(), log=str(log_path))
                        print('Seed', seed, 'continued after', provenance['preserved_total_steps'],
                              'steps; pid', child.pid, flush=True)
                    except Exception as exc:
                        item.update(status='failed', error=repr(exc), ended=time.time())
                        print('Seed', seed, 'transition failed:', repr(exc), flush=True)
                elif identity(item['source_pid']) is None:
                    item.update(status='failed', error='Source exited before saving its first boundary')
            elif item['status'] == 'running':
                code = children[seed].poll()
                if code is not None:
                    logs.pop(seed).close()
                    summary = Path(item['destination']) / 'summary.json'
                    verified = False
                    if code == 0 and summary.exists():
                        result = read_json(summary)
                        verified = result.get('completed_tasks') == NEW_TASKS and result.get('environment_steps', 0) >= sum(BUDGETS)
                    item.update(status='completed' if verified else 'failed', exit_code=code, ended=time.time())
        states = [item['status'] for item in state['seeds'].values()]
        state.update(updated=time.time(), status='running' if 'running' in states else 'waiting_for_first_task')
        if all(value in ('completed', 'failed') for value in states):
            state.update(status='completed' if all(value == 'completed' for value in states) else 'failed', ended=time.time())
        write_json(status_path, state)
        if state['status'] in ('completed', 'failed'):
            return 0 if state['status'] == 'completed' else 1
        time.sleep(1)


if __name__ == '__main__':
    raise SystemExit(main())
