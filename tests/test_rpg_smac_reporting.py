"""Measured-result aggregation and resume plans must not invent evidence."""
import json
from pathlib import Path
import subprocess
import sys

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / 'scripts'
sys.path.insert(0, str(SCRIPTS))
from run_rpg_smac_experiments import BUDGETS, TASKS, latest_checkpoint, next_attempt
from summarize_rpg_smac import aggregate, load_seed, summarize


def config(seed):
    return {'env': 'sc2', 'tasks': TASKS, 'task_steps': BUDGETS, 'seed': seed,
            'device': 'cpu', 'test_interval': 10000, 'env_args': {'game_version': '4.10'}}


def row(stage, step, task, value):
    return {'stage': stage, 'step': step, 'task': task, 'success_rate': value,
            'embedding_task': task, 'return_mean': value * 20, 'return_std': 0,
            'episodes': 32}


def write_attempt(base, seed, index, rows, boundaries=(), next_stage=1, resumed=None):
    attempt = base / f'seed_{seed}' / 'attempts' / f'attempt_{index:03d}'
    attempt.mkdir(parents=True)
    (attempt / 'config.json').write_text(json.dumps(config(seed)))
    (attempt / 'evaluations.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
    if boundaries:
        checkpoint = attempt / 'checkpoints' / f'task_{next_stage-1:02d}'
        checkpoint.mkdir(parents=True)
        (checkpoint / 'learner.th').write_bytes(b'not read by aggregation')
        progress = {'next_stage': next_stage, 'total_steps': max(r['step'] for r in boundaries),
                    'boundary_rows': list(boundaries)}
        (checkpoint / 'driver.json').write_text(json.dumps({'config': config(seed), 'progress': progress}))
    if resumed:
        attempt.with_suffix('.json').write_text(json.dumps({'resumed_from': str(resumed)}))
    return attempt


def test_three_seed_mean_std_and_acquisition_are_measured(tmp_path):
    for seed in range(3):
        boundary = row(0, 1000000 + seed * 10, TASKS[0], .8 + seed * .1)
        write_attempt(tmp_path, seed, 0,
                      [row(0, 10000 + seed * 7, TASKS[0], .4 + seed * .1), boundary], [boundary])
    report = summarize(tmp_path, tmp_path / 'aggregate', [0, 1, 2])
    assert report['paper_results_reproduced'] is None
    assert not report['all_requested_seeds_completed_all_stages']
    target = report['paper_table1_comparison'][0]
    assert target['measured_mean_win_rate'] == pytest.approx(.9)
    assert target['measured_sample_std'] == pytest.approx(.1)
    assert target['gap_percentage_points'] == pytest.approx(2.5)
    assert report['paper_table1_comparison'][1]['measured_mean_win_rate'] is None
    curves, _, _ = aggregate([load_seed(tmp_path / f'seed_{s}', s) for s in range(3)], [0, 1, 2])
    interval = next(r for r in curves if r['milestone'] == 'interval')
    assert interval['mean_win_rate'] == pytest.approx(.5)
    assert interval['actual_step_min'] == 10000
    assert interval['actual_step_max'] == 10014
    assert interval['complete_seed_coverage']


def test_missing_seed_is_disclosed_and_never_zero_filled(tmp_path):
    for seed in (0, 1):
        boundary = row(0, 1000000 + seed, TASKS[0], .8 + seed * .1)
        write_attempt(tmp_path, seed, 0, [boundary], [boundary])
    report = summarize(tmp_path, tmp_path / 'report', [0, 1, 2])
    value = report['paper_table1_comparison'][0]
    assert report['missing_seeds'] == [2]
    assert value['n_seeds'] == 2
    assert value['measured_mean_win_rate'] == pytest.approx(.85)
    assert not value['complete_seed_coverage']


def test_resumed_attempt_discards_abandoned_post_checkpoint_rows(tmp_path):
    boundary = row(0, 1000010, TASKS[0], .8)
    first = write_attempt(tmp_path, 0, 0,
                          [boundary, row(1, 1010010, TASKS[1], .99),
                           row(1, 1020010, TASKS[1], .98)], [boundary])
    checkpoint = first / 'checkpoints/task_00'
    write_attempt(tmp_path, 0, 1, [row(1, 1010010, TASKS[1], .1)], resumed=checkpoint)
    run = load_seed(tmp_path / 'seed_0', 0)
    rates = [r['success_rate'] for r in run['records'] if r['stage'] == 1]
    assert rates == [.1]
    assert run['boundary_rows'] == [boundary]


def test_incompatible_seed_configs_are_not_averaged(tmp_path):
    for seed in (0, 1):
        boundary = row(0, 1000000, TASKS[0], .5)
        attempt = write_attempt(tmp_path, seed, 0, [boundary], [boundary])
        if seed == 1:
            changed = config(seed)
            changed['test_interval'] = 20000
            (attempt / 'config.json').write_text(json.dumps(changed))
    with pytest.raises(ValueError, match='configurations differ'):
        aggregate([load_seed(tmp_path / f'seed_{s}', s) for s in (0, 1)], [0, 1])


def test_resume_chooses_latest_complete_checkpoint_and_preserves_failed_logs(tmp_path):
    first_boundary = row(0, 1000010, TASKS[0], .4)
    first = write_attempt(tmp_path, 0, 0, [first_boundary], [first_boundary])
    partial = first / 'checkpoints/task_01'
    partial.mkdir()
    (partial / 'driver.json').write_text('{}')  # learner.th absent: incomplete write.
    seed_dir = tmp_path / 'seed_0'
    chosen = latest_checkpoint(seed_dir, 0)
    assert chosen['next_stage'] == 1
    assert chosen['path'] == str((first / 'checkpoints/task_00').resolve())
    (seed_dir / 'attempts/attempt_001.log').write_text('failed before driver created directory')
    assert next_attempt(seed_dir).name == 'attempt_002'


def test_dry_run_needs_no_torch_or_game_and_creates_no_files(tmp_path):
    output = tmp_path / 'absent'
    result = subprocess.run([sys.executable, str(SCRIPTS / 'run_rpg_smac_experiments.py'),
                             '--dry-run', '--output', str(output)],
                            capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stderr
    plan = json.loads(result.stdout)
    assert plan['settings']['task_steps'] == BUDGETS
    assert [r['seed'] for r in plan['plan']] == [0, 1, 2]
    assert not output.exists()
    assert all('--checkpoint' not in ' '.join(r['command']) for r in plan['plan'])
