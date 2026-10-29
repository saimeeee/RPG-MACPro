"""Run the four-task RPG SMAC protocol with durable per-seed attempts.

This launcher uses only the standard library. --dry-run does not import torch,
launch StarCraft, or create files. A resumed attempt discards an unfinished task
and resumes at its latest completed task boundary; its old files are preserved.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
TASKS = ['5m_vs_6m', '12m', '5m', '8m_vs_9m']
BUDGETS = [1000000, 500000, 500000, 500000]


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')
    temporary.replace(path)


def validate_config(config, seed):
    names = [x if isinstance(x, str) else x['name'] for x in config['tasks']]
    if config.get('env') != 'sc2' or names != TASKS or config.get('task_steps') != BUDGETS:
        raise ValueError('Checkpoint does not use the four-task, 2.5M-step SMAC protocol')
    if config.get('seed') != seed:
        raise ValueError('Checkpoint seed does not match the seed being resumed')


def latest_checkpoint(seed_dir, seed):
    """Ignore partial writes; require both learner and driver at a task boundary."""
    best = None
    for path in Path(seed_dir).glob('attempts/attempt_*/checkpoints/task_*/driver.json'):
        if not (path.parent / 'learner.th').is_file():
            continue
        try:
            data = read_json(path)
            validate_config(data['config'], seed)
            progress = data['progress']
            stage = int(progress['next_stage'])
            step = int(progress['total_steps'])
            if not 1 <= stage <= len(TASKS) or step < sum(BUDGETS[:stage]):
                continue
            candidate = {'path': str(path.parent.resolve()), 'next_stage': stage,
                         'total_steps': step}
            if best is None or (stage, step, str(path)) > (best['next_stage'], best['total_steps'], best['path']):
                best = candidate
        except (KeyError, ValueError, TypeError, json.JSONDecodeError):
            # An incompatible or incomplete file cannot become a resume point.
            continue
    return best


def build_command(python, destination, seed, device, checkpoint=None, overrides=()):
    command = [str(python), '-u', str(ROOT / 'src/main.py'), '--config=qmix_rpg',
               '--task-config=rpg_marines', '--device=' + device,
               '--output=' + str(destination)]
    if checkpoint:
        command += ['--checkpoint=' + str(checkpoint)]
    else:
        command += ['--seed=' + str(seed)]
        command += ['--set=' + value for value in overrides]
    return command


def process_alive(pid):
    try:
        os.kill(int(pid), 0)
        return True
    except (OSError, ValueError):
        return False


@contextmanager
def launcher_lock(output, resume):
    lock = output / '.launcher.lock'
    if lock.exists():
        state = read_json(lock)
        if state.get('hostname') != socket.gethostname() or process_alive(state.get('pid', 0)):
            raise RuntimeError('Another launcher may own this experiment directory; inspect .launcher.lock')
        if not resume:
            raise RuntimeError('A stale launcher lock exists; use --resume to recover it')
        lock.unlink()
    fd = os.open(lock, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as f:
            json.dump({'pid': os.getpid(), 'hostname': socket.gethostname(), 'created': utc_now()}, f)
        yield
    finally:
        if lock.exists():
            lock.unlink()


def stop_process_tree(process):
    """Each experiment owns a separate process group, including its SC2 children."""
    if os.name == 'nt':
        subprocess.run(['taskkill', '/PID', str(process.pid), '/T', '/F'],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
    else:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            return
    try:
        process.wait(timeout=8)
    except subprocess.TimeoutExpired:
        if os.name == 'nt':
            process.kill()
        else:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        process.wait()
    if os.name != 'nt':
        # The Python parent may exit before a stuck game child. Check the group
        # itself rather than assuming parent exit means every child has exited.
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            try:
                os.killpg(process.pid, 0)
            except ProcessLookupError:
                break
            time.sleep(.1)
        else:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass


def next_attempt(seed_dir):
    parent = Path(seed_dir) / 'attempts'
    ids = [int(p.stem.split('_')[-1]) for p in parent.glob('attempt_*')
           if p.stem.split('_')[-1].isdigit()]
    return parent / ('attempt_%03d' % (max(ids, default=-1) + 1))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='results/rpg_smac_full')
    parser.add_argument('--python', default=sys.executable)
    parser.add_argument('--device', choices=['cpu', 'cuda', 'auto'], default='cuda')
    parser.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
    parser.add_argument('--max-parallel', type=int, default=1,
                        help='Independent seed processes; each also launches multiple SC2 engines')
    parser.add_argument('--set', action='append', default=[], metavar='KEY=VALUE')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args(argv)
    if len(set(args.seeds)) != len(args.seeds) or args.max_parallel < 1:
        parser.error('Seeds must be unique and max-parallel must be positive')
    for override in args.set:
        if '=' not in override or override.split('=', 1)[0] in ('seed', 'env', 'tasks', 'task_steps'):
            parser.error('--set must preserve the published task order and budgets')
    return args


def main(argv=None):
    args = parse_args(argv)
    output = Path(args.output).resolve()
    settings = {'seeds': args.seeds, 'device': args.device, 'overrides': args.set,
                'tasks': TASKS, 'task_steps': BUDGETS}
    plan = []
    for seed in args.seeds:
        seed_dir = output / f'seed_{seed}'
        checkpoint = latest_checkpoint(seed_dir, seed) if args.resume else None
        destination = next_attempt(seed_dir)
        plan.append({'seed': seed, 'checkpoint': checkpoint,
                     'already_complete': bool(checkpoint and checkpoint['next_stage'] == len(TASKS)),
                     'destination': str(destination),
                     'command': build_command(args.python, destination, seed, args.device,
                                              checkpoint['path'] if checkpoint else None, args.set)})
    if args.dry_run:
        print(json.dumps({'settings': settings, 'plan': plan,
                          'note': 'No processes launched. No result or success is inferred.'}, indent=2))
        return 0
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / 'manifest.json'
    with launcher_lock(output, args.resume):
        if manifest_path.exists():
            if not args.resume:
                raise ValueError('Existing experiment manifest requires --resume')
            manifest = read_json(manifest_path)
            if manifest['settings'] != settings:
                raise ValueError('Resuming preserves seeds, device and overrides from manifest.json')
        else:
            if any((output / f'seed_{s}').exists() for s in args.seeds):
                raise ValueError('Existing seed files have no manifest; use a fresh output directory')
            manifest = {'schema_version': 1, 'created': utc_now(), 'settings': settings,
                        'seeds': {str(seed): {'status': 'pending', 'attempts': []} for seed in args.seeds}}
        manifest.update(status='running', launcher_pid=os.getpid(), updated=utc_now())
        write_json(manifest_path, manifest)
        pending = list(plan)
        running = {}
        exit_code = 0
        def interrupted(signum, frame):
            raise KeyboardInterrupt('Launcher received signal %s' % signum)
        old_term = signal.signal(signal.SIGTERM, interrupted)
        try:
            while pending or running:
                while pending and len(running) < args.max_parallel:
                    item = pending.pop(0)
                    record = manifest['seeds'][str(item['seed'])]
                    if item['already_complete']:
                        record.update(status='completed', completed_checkpoint=item['checkpoint'])
                        write_json(manifest_path, manifest)
                        continue
                    destination = Path(item['destination'])
                    # The driver insists its output directory be empty; logs and
                    # attempt metadata are siblings, never inside that directory.
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    log_path = destination.with_suffix('.log')
                    meta_path = destination.with_suffix('.json')
                    log = log_path.open('w', encoding='utf-8', buffering=1)
                    attempt = {'path': str(destination), 'log': str(log_path),
                               'started': utc_now(), 'status': 'starting',
                               'resumed_from': item['checkpoint']['path'] if item['checkpoint'] else None,
                               'resume_boundary': item['checkpoint'],
                               'command': item['command']}
                    record['attempts'].append(attempt)
                    record['status'] = 'running'
                    options = {'start_new_session': True} if os.name != 'nt' else {
                        'creationflags': subprocess.CREATE_NEW_PROCESS_GROUP}
                    try:
                        process = subprocess.Popen(item['command'], cwd=ROOT, stdout=log,
                                                   stderr=subprocess.STDOUT, **options)
                    except Exception:
                        log.close()
                        attempt.update(status='failed_to_start', ended=utc_now())
                        record['status'] = 'failed'
                        write_json(meta_path, attempt)
                        write_json(manifest_path, manifest)
                        raise
                    attempt.update(status='running', pid=process.pid)
                    write_json(meta_path, attempt)
                    running[process.pid] = (process, log, attempt, record, meta_path, item)
                    write_json(manifest_path, manifest)
                    print('Started seed %s; log %s' % (item['seed'], log_path), flush=True)
                for pid, entry in list(running.items()):
                    process, log, attempt, record, meta_path, item = entry
                    returncode = process.poll()
                    if returncode is None:
                        continue
                    log.close()
                    completed = latest_checkpoint(output / f'seed_{item["seed"]}', item['seed'])
                    summary_path = Path(attempt['path']) / 'summary.json'
                    verified = (returncode == 0 and completed is not None and
                                completed['next_stage'] == len(TASKS) and summary_path.is_file())
                    if verified:
                        summary = read_json(summary_path)
                        verified = (summary.get('mode') == 'training' and
                                    summary.get('completed_tasks') == TASKS and
                                    summary.get('environment_steps', 0) >= sum(BUDGETS))
                    attempt.update(status='completed' if verified else 'failed',
                                   returncode=returncode, ended=utc_now())
                    record['status'] = attempt['status']
                    if completed:
                        record['completed_checkpoint'] = completed
                    if not verified:
                        stop_process_tree(process)
                        exit_code = 1
                    write_json(meta_path, attempt)
                    manifest['updated'] = utc_now()
                    write_json(manifest_path, manifest)
                    del running[pid]
                    print('Seed %s: %s (exit %s)' % (item['seed'], attempt['status'], returncode), flush=True)
                if running:
                    time.sleep(1)
        except BaseException:
            for process, log, attempt, record, meta_path, item in running.values():
                stop_process_tree(process)
                log.close()
                attempt.update(status='interrupted', ended=utc_now(), returncode=process.returncode)
                record['status'] = 'interrupted'
                write_json(meta_path, attempt)
            manifest.update(status='interrupted', updated=utc_now())
            write_json(manifest_path, manifest)
            raise
        finally:
            signal.signal(signal.SIGTERM, old_term)
        statuses = [record['status'] for record in manifest['seeds'].values()]
        manifest.update(status='completed' if all(x == 'completed' for x in statuses) else 'failed',
                        updated=utc_now())
        write_json(manifest_path, manifest)
        return exit_code


if __name__ == '__main__':
    raise SystemExit(main())
