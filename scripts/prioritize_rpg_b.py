#!/usr/bin/env python3
"""Pause A at verified task boundaries, wait for all B seeds, then resume A.

Linux server coordinator. Does not modify training code or existing results.
An existing manifest is deliberately rejected: restarting needs an explicit
state review, never an automatic duplicate launch or restart from zero.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time

TASKS = ['5m_vs_6m', '13m', '4m', '8m_vs_9m']
BUDGETS = [1000000, 500000, 500000, 500000]
ENV_KEYS = ('PATH', 'PYTHONPATH', 'SC2PATH', 'VIRTUAL_ENV', 'LD_LIBRARY_PATH',
            'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
            'SDL_VIDEODRIVER', 'MPLBACKEND', 'CUDA_VISIBLE_DEVICES',
            'PYTHONUNBUFFERED', 'PYTHONHASHSEED')


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_json(path, data):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w', encoding='utf-8') as handle:
        json.dump(data, handle, indent=2)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def identity(pid):
    try:
        proc = Path('/proc') / str(pid)
        fields = (proc / 'stat').read_text().rsplit(')', 1)[1].split()
        if fields[0] == 'Z':
            return None
        return {'start_ticks': int(fields[19]), 'pgid': int(fields[2]),
                'command': (proc / 'cmdline').read_bytes().replace(b'\0', b' ').decode()}
    except (OSError, ValueError):
        return None


def group_members(pgid):
    members = {}
    for path in Path('/proc').glob('[0-9]*'):
        pid = int(path.name)
        info = identity(pid)
        if info and info['pgid'] == pgid:
            members[pid] = info
    return members


def stop_group(pid, expected):
    actual = identity(pid)
    if actual != expected or actual is None or actual['pgid'] != pid:
        raise RuntimeError('Process identity/group changed; refusing to signal')
    members = group_members(pid)
    os.killpg(pid, signal.SIGTERM)
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        if not group_members(pid):
            return
        time.sleep(.2)
    remaining = group_members(pid)
    if any(members.get(p) != info for p, info in remaining.items()):
        raise RuntimeError('Unexpected process group member; refusing SIGKILL')
    if remaining:
        os.killpg(pid, signal.SIGKILL)
    deadline = time.monotonic() + 5
    while group_members(pid) and time.monotonic() < deadline:
        time.sleep(.1)
    if group_members(pid):
        raise RuntimeError('Training process group did not exit')


def source_hashes(root):
    root = Path(root)
    files = list((root / 'src').rglob('*.py')) + list((root / 'src/config').rglob('*.yaml'))
    return {str(p.relative_to(root)): sha256(p) for p in sorted(files)}


def validate_checkpoint(folder, config, stage):
    import torch
    folder = Path(folder)
    driver = read_json(folder / 'driver.json')
    if driver['config'] != config:
        raise ValueError('Checkpoint config differs from the running experiment')
    progress = driver['progress']
    if progress['next_stage'] != stage + 1 or progress['total_steps'] < sum(config['task_steps'][:stage + 1]):
        raise ValueError('Checkpoint is not the requested complete task boundary')
    payload = torch.load(folder / 'learner.th', map_location='cpu', weights_only=True)
    required = {'agent', 'target_agent', 'mixer', 'target_mixer', 'optimiser',
                'relation_anchors', 'relation_importance', 'hyper_targets',
                'torch_rng', 'cuda_rng', 'python_rng', 'numpy_rng', 'train_steps',
                'return_samples', 'gamma_p', 'last_target_update_episode', 'log_stats_t'}
    if payload.get('version') != 1 or not required <= payload.keys():
        raise ValueError('Incomplete training/optimizer/continual-learning/RNG state')
    if payload['progress'] != progress or payload['completed_tasks'] != config['tasks'][:stage + 1]:
        raise ValueError('Learner and driver completion metadata disagree')
    if payload['current_task'] != config['tasks'][stage]:
        raise ValueError('Wrong current task in checkpoint')
    return {'next_stage': stage + 1, 'total_steps': progress['total_steps'],
            'optimizer_updates': payload['train_steps'],
            'sha256': {name: sha256(folder / name) for name in ('driver.json', 'learner.th')}}


def preserve_checkpoint(item):
    source, destination = Path(item['checkpoint']), Path(item['saved_checkpoint'])
    proof = validate_checkpoint(source, item['config'], item['target_stage'])
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, destination)  # refuses an existing destination
    if validate_checkpoint(destination, item['config'], item['target_stage']) != proof:
        raise RuntimeError('Checkpoint copy verification failed')
    return proof


def b_completed(root, manifest):
    if set(manifest['seeds']) != {'0', '1', '2'}:
        raise ValueError('Unexpected B seed set')
    if manifest['tasks'] != TASKS or manifest['task_intervals'] != BUDGETS:
        raise ValueError('Unexpected B protocol')
    for seed, row in manifest['seeds'].items():
        if row['status'] == 'failed':
            raise RuntimeError('B seed ' + seed + ' failed; A remains safely paused')
        if row['status'] != 'completed':
            return False
        result = read_json(Path(root) / ('seed_' + seed) / 'completion.json')
        if (row.get('exit_code') != 0 or result.get('status') != 'completed'
                or result.get('seed') != int(seed) or result.get('seen_tasks') != TASKS
                or result.get('step', 0) < sum(BUDGETS)):
            raise ValueError('B completion proof invalid for seed ' + seed)
        if identity(row['pid']) is not None or group_members(row['pid']):
            return False
    return True


def resume_command(item):
    command = [arg for arg in item['original_command']
               if not arg.startswith(('--checkpoint=', '--output='))]
    return command + ['--checkpoint=' + item['saved_checkpoint'], '--output=' + item['resume_output']]


def initialise(args):
    transition = read_json(args.transition / 'manifest.json')
    b = read_json(args.b_output / 'manifest.json')
    if transition['tasks'] != TASKS or transition['task_steps'] != BUDGETS:
        raise ValueError('Unexpected A protocol')
    if set(transition['seeds']) != {'0', '1', '2'}:
        raise ValueError('Unexpected A seed set')
    if b_completed(args.b_output, b):
        raise ValueError('B already completed; no priority handoff is needed')
    launcher = identity(b['launcher_pid'])
    if not launcher or 'run_comparison.py' not in launcher['command'] or str(args.b_output) not in launcher['command']:
        raise RuntimeError('B launcher is not active')
    state = {'status': 'waiting_for_a_boundaries', 'pid': os.getpid(), 'created': time.time(),
             'transition_manifest': str(args.transition / 'manifest.json'),
             'b_output': str(args.b_output), 'b_launcher_pid': b['launcher_pid'],
             'b_launcher_identity': launcher, 'seeds': {}}
    for seed, row in transition['seeds'].items():
        source = Path(row['destination'])
        proc = Path('/proc') / str(row['pid'])
        info = identity(row['pid'])
        command = (proc / 'cmdline').read_bytes().decode().rstrip('\0').split('\0')
        if (row['status'] != 'running' or info is None or info['pgid'] != row['pid']
                or command != row['command'] or '--output=' + str(source) not in command):
            raise RuntimeError('A source process does not match its manifest')
        progress, config = read_json(source / 'progress.json'), read_json(source / 'config.json')
        if (progress['stage'] != args.target_stage or progress['task'] != TASKS[args.target_stage]
                or config['seed'] != int(seed) or config['tasks'] != TASKS or config['task_steps'] != BUDGETS):
            raise ValueError('A has changed stage/protocol; inspect before installing a new target')
        checkpoint = source / 'checkpoints' / ('task_%02d' % args.target_stage)
        if (checkpoint / 'driver.json').exists():
            raise ValueError('Target boundary already saved; inspect current map before installing')
        env = dict(s.split('=', 1) for s in (proc / 'environ').read_text().split('\0') if '=' in s)
        cwd = os.readlink(proc / 'cwd')
        resume_output = source.parent / 'after_b' / 'attempt_000'
        if resume_output.exists():
            raise FileExistsError(resume_output)
        state['seeds'][seed] = {
            'status': 'waiting_for_boundary', 'source_pid': row['pid'], 'source_identity': info,
            'source_output': str(source), 'original_command': command, 'cwd': cwd,
            'environment': {k: env[k] for k in ENV_KEYS if k in env}, 'config': config,
            'target_stage': args.target_stage, 'target_task': TASKS[args.target_stage],
            'checkpoint': str(checkpoint),
            'saved_checkpoint': str(args.control / ('seed_' + seed) / 'checkpoint'),
            'resume_output': str(resume_output), 'source_sha256': source_hashes(cwd),
        }
    write_json(args.control / 'initial_a_manifest.json', transition)
    write_json(args.control / 'initial_b_manifest.json', b)
    pointer = {'manifest': str(args.control / 'manifest.json'),
               'policy': 'A current-map checkpoint -> pause A -> complete B seeds 0/1/2 -> resume A',
               'note': 'Old transition may mark deliberately stopped attempts failed. Use this manifest for continuation.'}
    write_json(args.transition / 'priority_handoff.json', pointer)
    write_json(Path(transition['source_output']) / 'priority_handoff.json', pointer)
    return state


def tick(state, persist, children, handles):
    for seed, item in state['seeds'].items():
        if item['status'] != 'waiting_for_boundary':
            continue
        checkpoint = Path(item['checkpoint'])
        if (checkpoint / 'driver.json').exists() and (checkpoint / 'learner.th').exists():
            try:
                read_json(checkpoint / 'driver.json')  # writer may still be finishing JSON
            except json.JSONDecodeError:
                continue
            item['proof'] = preserve_checkpoint(item)
            item['status'] = 'checkpoint_verified'
            persist()
            stop_group(item['source_pid'], item['source_identity'])
            item.update(status='paused_for_b', paused=time.time())
            print('A seed', seed, 'paused at', item['proof']['total_steps'], flush=True)
            persist()
        elif identity(item['source_pid']) != item['source_identity']:
            raise RuntimeError('A seed ' + seed + ' exited/changed before its boundary')
    if any(x['status'] == 'waiting_for_boundary' for x in state['seeds'].values()):
        state['status'] = 'waiting_for_a_boundaries'
        return
    if any(x['status'] == 'paused_for_b' for x in state['seeds'].values()):
        state['status'] = 'waiting_for_b_completion'
        b = read_json(Path(state['b_output']) / 'manifest.json')
        state['b_seed_states'] = {k: v['status'] for k, v in b['seeds'].items()}
        if not b_completed(state['b_output'], b):
            if identity(state['b_launcher_pid']) != state['b_launcher_identity']:
                raise RuntimeError('B launcher exited before all B seeds completed')
            return
        # Validate ALL source files and snapshots before launching any A child.
        for item in state['seeds'].values():
            if source_hashes(item['cwd']) != item['source_sha256']:
                raise RuntimeError('A source files changed while paused; refusing mixed-code resume')
            if validate_checkpoint(item['saved_checkpoint'], item['config'], item['target_stage']) != item['proof']:
                raise RuntimeError('Preserved A checkpoint changed')
            if Path(item['resume_output']).exists():
                raise FileExistsError(item['resume_output'])
            if identity(item['source_pid']) is not None or group_members(item['source_pid']):
                raise RuntimeError('Original A process group is still present')
        state.update(status='resuming_a', b_completed_at=time.time())
        persist()
        for seed, item in state['seeds'].items():
            output = Path(item['resume_output'])
            output.parent.mkdir(parents=True, exist_ok=True)
            log = output.with_suffix('.log')
            handles[seed] = log.open('xb')
            item.update(status='launching_a', resume_command=resume_command(item), log=str(log))
            persist()  # an interrupted launch is explicit and never retried blindly
            children[seed] = subprocess.Popen(item['resume_command'], cwd=item['cwd'],
                env={**os.environ, **item['environment']}, stdin=subprocess.DEVNULL,
                stdout=handles[seed], stderr=subprocess.STDOUT, start_new_session=True)
            item.update(status='resumed', pid=children[seed].pid,
                        identity=identity(children[seed].pid), resumed=time.time())
            print('A seed', seed, 'resumed; pid', children[seed].pid, flush=True)
            persist()
    for seed, child in children.items():
        item = state['seeds'][seed]
        if item['status'] != 'resumed':
            continue
        code = child.poll()
        if code is not None:
            handles.pop(seed).close()
            summary = Path(item['resume_output']) / 'summary.json'
            result = read_json(summary) if summary.exists() else {}
            ok = (code == 0 and result.get('seed') == int(seed)
                  and result.get('completed_tasks') == TASKS
                  and result.get('environment_steps', 0) >= sum(BUDGETS))
            item.update(status='completed' if ok else 'failed', exit_code=code, ended=time.time())
    statuses = [x['status'] for x in state['seeds'].values()]
    state['status'] = ('completed' if all(s == 'completed' for s in statuses)
                       else 'needs_attention' if 'failed' in statuses else 'a_resumed')


def main():
    import fcntl
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--transition', required=True, type=Path)
    parser.add_argument('--b-output', required=True, type=Path)
    parser.add_argument('--control', required=True, type=Path)
    parser.add_argument('--target-stage', required=True, type=int, choices=(0, 1, 2))
    args = parser.parse_args()
    args.control.mkdir(parents=True, exist_ok=True)
    lock = (args.control / '.lock').open('a+')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    status_path = args.control / 'manifest.json'
    if status_path.exists():
        raise FileExistsError('Existing handoff: inspect manifest; do not launch a duplicate')
    state = initialise(args)
    def persist():
        state['updated'] = time.time()
        write_json(status_path, state)
    persist()
    print('Installed A boundary -> B priority -> A resume handoff', flush=True)
    children, handles = {}, {}
    while True:
        try:
            tick(state, persist, children, handles)
        except Exception as exc:
            state.update(status='needs_attention', error=repr(exc), ended=time.time())
            print('Handoff needs attention:', repr(exc), flush=True)
        persist()
        if state['status'] in ('completed', 'needs_attention'):
            return int(state['status'] != 'completed')
        time.sleep(1 if state['status'] == 'waiting_for_a_boundaries' else 10)


if __name__ == '__main__':
    raise SystemExit(main())
