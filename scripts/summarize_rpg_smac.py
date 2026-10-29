"""Aggregate measured RPG SMAC records without interpolating missing seeds.

Periodic evaluations are grouped by their stage-relative test-interval bucket;
actual step ranges are retained. Task endpoints come only from committed task
boundary checkpoints. Paper Table 1 is an acquisition comparison, not the final
retention result after all later tasks.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics

from run_rpg_smac_experiments import BUDGETS, TASKS, read_json, validate_config, write_json

PAPER_ACQUISITION = dict(zip(TASKS, [0.875, 1.0, 1.0, 0.91]))
PAPER_URL = 'https://www.ijcai.org/proceedings/2025/759'


def configuration_signature(config):
    comparable = {k: v for k, v in config.items() if k not in ('seed', 'device')}
    return hashlib.sha256(json.dumps(comparable, sort_keys=True).encode()).hexdigest()


def row_key(row):
    return (int(row['stage']), int(row['step']), row['task'], row.get('embedding_task', row['task']))


def read_rows(path):
    if not Path(path).exists():
        return []
    rows = []
    with Path(path).open(encoding='utf-8') as f:
        for number, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f'Invalid JSON in {path}:{number}; do not aggregate a partially written file') from exc
            if isinstance(row.get('stage'), int):
                if row['task'] not in TASKS or not 0 <= row['stage'] < len(TASKS):
                    raise ValueError(f'Unexpected task/stage in {path}:{number}')
                value = float(row['success_rate'])
                if not math.isfinite(value) or not 0 <= value <= 1:
                    raise ValueError(f'Invalid win rate in {path}:{number}')
                rows.append(row)
    return rows


def committed_boundary(attempt):
    best = None
    for driver in Path(attempt).glob('checkpoints/task_*/driver.json'):
        if not (driver.parent / 'learner.th').is_file():
            continue
        data = read_json(driver)
        progress = data.get('progress', {})
        if best is None or progress.get('next_stage', 0) > best['progress'].get('next_stage', 0):
            best = data
    return best


def load_seed(seed_dir, seed):
    """Later attempts replace abandoned work after their resume checkpoint."""
    records, boundary_rows, runtimes = {}, {}, []
    config = None
    source_hashes = set()
    attempts = sorted(Path(seed_dir).glob('attempts/attempt_*'))
    attempts = [path for path in attempts if path.is_dir() and (path / 'config.json').exists()]
    if not attempts:
        return {'seed': seed, 'missing': True, 'records': [], 'boundary_rows': [], 'runtimes': []}
    for index, attempt in enumerate(attempts):
        current_config = read_json(attempt / 'config.json')
        validate_config(current_config, seed)
        if config is not None and configuration_signature(current_config) != configuration_signature(config):
            raise ValueError(f'Configuration changed across attempts for seed {seed}')
        config = current_config
        attempt_meta = attempt.with_suffix('.json')
        metadata = read_json(attempt_meta) if attempt_meta.exists() else {}
        resumed_from = metadata.get('resumed_from')
        if index:
            if resumed_from:
                resume_boundary = metadata.get('resume_boundary')
                if resume_boundary:
                    cutoff = int(resume_boundary['total_steps'])
                else:
                    original = Path(resumed_from)
                    if not original.exists() and Path(seed_dir).name in original.parts:
                        offset = original.parts.index(Path(seed_dir).name)
                        original = Path(seed_dir).joinpath(*original.parts[offset + 1:])
                    restored = read_json(original / 'driver.json')
                    validate_config(restored['config'], seed)
                    cutoff = int(restored['progress']['total_steps'])
                records = {key: row for key, row in records.items() if row['step'] <= cutoff}
                boundary_rows = {key: row for key, row in boundary_rows.items() if row['step'] <= cutoff}
            else:
                # No checkpoint means a full restart. Old partial curves must
                # not be spliced into this different run of the seed.
                records, boundary_rows = {}, {}
        for row in read_rows(attempt / 'evaluations.jsonl'):
            records[row_key(row)] = row
        committed = committed_boundary(attempt)
        if committed:
            validate_config(committed['config'], seed)
            for row in committed['progress'].get('boundary_rows', []):
                boundary_rows[row_key(row)] = row
                records[row_key(row)] = row
        runtime_path = attempt / 'runtime.json'
        if runtime_path.exists():
            runtime = read_json(runtime_path)
            runtimes.append({'attempt': str(attempt), 'runtime': runtime})
            for key in ('source_sha256', 'source_hash', 'source_fingerprint'):
                if runtime.get(key):
                    source_hashes.add(hashlib.sha256(json.dumps(runtime[key], sort_keys=True).encode()).hexdigest())
    return {'seed': seed, 'missing': False, 'config': config,
            'configuration_signature': configuration_signature(config),
            'records': list(records.values()), 'boundary_rows': list(boundary_rows.values()),
            'runtimes': runtimes, 'source_hashes': sorted(source_hashes),
            'attempts': [str(path) for path in attempts]}


def measured_summary(values):
    values = [float(v) for v in values]
    return {'mean': statistics.mean(values),
            'sample_std': statistics.stdev(values) if len(values) > 1 else None,
            'n_seeds': len(values)}


def aggregate(seed_runs, expected_seeds):
    present = [run for run in seed_runs if not run['missing']]
    signatures = {run['configuration_signature'] for run in present}
    if len(signatures) > 1:
        raise ValueError('Seed configurations differ; do not average incompatible experiments')
    grouped, boundary_groups = {}, {}
    for run in present:
        seed, config = run['seed'], run['config']
        endpoints = {(int(row['stage']), row['task']): row for row in run['boundary_rows']}
        stage_start = {0: 0}
        for stage in range(1, len(TASKS)):
            previous = [row['step'] for row in run['boundary_rows'] if row['stage'] == stage - 1]
            if previous:
                stage_start[stage] = max(previous)
        boundary_keys = {row_key(row) for row in run['boundary_rows']}
        per_seed_bins = {}
        for row in sorted(run['records'], key=lambda x: (x['stage'], x['step'])):
            stage = int(row['stage'])
            if stage not in stage_start:
                continue  # Cannot align an uncommitted preceding task boundary.
            relative = int(row.get('stage_step', row['step'] - stage_start[stage]))
            boundary = row_key(row) in boundary_keys
            bucket = 'boundary' if boundary else relative // int(config['test_interval'])
            key = (stage, bucket, row['task'])
            # A task endpoint and an interval evaluation are intentionally
            # different milestones; repeated interval observations use the last.
            per_seed_bins[key] = {**row, 'stage_step': relative, 'seed': seed}
        for key, row in per_seed_bins.items():
            grouped.setdefault(key, {})[seed] = row
        for (stage, task), row in endpoints.items():
            boundary_groups.setdefault((stage, task), {})[seed] = row
    curves = []
    for (stage, bucket, task), seed_rows in grouped.items():
        rows = list(seed_rows.values())
        stat = measured_summary(row['success_rate'] for row in rows)
        curves.append({'stage': stage, 'task': task,
                       'milestone': 'boundary' if bucket == 'boundary' else 'interval',
                       'interval_index': None if bucket == 'boundary' else bucket,
                       'scheduled_step': sum(BUDGETS[:stage]) +
                           (BUDGETS[stage] if bucket == 'boundary' else bucket * present[0]['config']['test_interval']),
                       'actual_step_min': min(row['step'] for row in rows),
                       'actual_step_max': max(row['step'] for row in rows),
                       'stage_step_min': min(row['stage_step'] for row in rows),
                       'stage_step_max': max(row['stage_step'] for row in rows),
                       'mean_win_rate': stat['mean'], 'sample_std': stat['sample_std'],
                       'n_seeds': stat['n_seeds'], 'seeds': sorted(seed_rows),
                       'complete_seed_coverage': set(seed_rows) == set(expected_seeds)})
    curves.sort(key=lambda row: (row['task'], row['scheduled_step'], row['milestone']))
    matrix = []
    for (stage, task), seed_rows in sorted(boundary_groups.items()):
        stat = measured_summary(row['success_rate'] for row in seed_rows.values())
        matrix.append({'stage': stage, 'task': task, 'mean_win_rate': stat['mean'],
                       'sample_std': stat['sample_std'], 'n_seeds': stat['n_seeds'],
                       'seeds': sorted(seed_rows),
                       'complete_seed_coverage': set(seed_rows) == set(expected_seeds)})
    acquisition = []
    for stage, task in enumerate(TASKS):
        row = next((row for row in matrix if row['stage'] == stage and row['task'] == task), None)
        acquisition.append({'task': task, 'paper_table_1_win_rate': PAPER_ACQUISITION[task],
                            'measured_mean_win_rate': row['mean_win_rate'] if row else None,
                            'measured_sample_std': row['sample_std'] if row else None,
                            'n_seeds': row['n_seeds'] if row else 0,
                            'gap_percentage_points': 100 * (row['mean_win_rate'] - PAPER_ACQUISITION[task]) if row else None,
                            'complete_seed_coverage': row['complete_seed_coverage'] if row else False})
    return curves, matrix, acquisition


def write_csv(path, rows):
    if not rows:
        Path(path).write_text('', encoding='utf-8')
        return
    with Path(path).open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value) if isinstance(value, list) else value
                             for key, value in row.items()})


def plot_curves(path, curves):
    try:
        import matplotlib
        matplotlib.use('Agg')
        from matplotlib import pyplot as plt
    except ImportError as exc:
        raise RuntimeError('--plot requires matplotlib; CSV/JSON work without it') from exc
    fig, axes = plt.subplots(4, 1, figsize=(9, 10), sharex=True)
    for index, (axis, task) in enumerate(zip(axes, TASKS)):
        rows = [row for row in curves if row['task'] == task and row['complete_seed_coverage']]
        rows.sort(key=lambda row: row['scheduled_step'])
        x = [row['scheduled_step'] / 1e6 for row in rows]
        y = [100 * row['mean_win_rate'] for row in rows]
        std = [100 * (row['sample_std'] or 0) for row in rows]
        axis.plot(x, y, color='#1864ab', linewidth=1.5, label='RPG reproduction')
        axis.fill_between(x, [max(0, a-b) for a, b in zip(y, std)],
                          [min(100, a+b) for a, b in zip(y, std)], color='#1864ab', alpha=.17)
        axis.axvspan(sum(BUDGETS[:index])/1e6, sum(BUDGETS[:index+1])/1e6, color='gray', alpha=.12)
        axis.set(title=task, ylim=(0, 100), ylabel='Win rate (%)')
        axis.grid(alpha=.15)
    axes[-1].set(xlabel='Scheduled training environment steps (millions)', xlim=(0, 2.5))
    fig.suptitle('Measured mean ± sample SD; complete seed coverage only')
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def summarize(input_dir, output_dir, expected_seeds, plot=False):
    input_dir, output_dir = Path(input_dir), Path(output_dir)
    runs = [load_seed(input_dir / f'seed_{seed}', seed) for seed in expected_seeds]
    curves, matrix, acquisition = aggregate(runs, expected_seeds)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / 'learning_curves.csv', curves)
    write_csv(output_dir / 'continual_matrix.csv', matrix)
    write_csv(output_dir / 'paper_table1_comparison.csv', acquisition)
    complete = all(any(row['stage'] == 3 and row['task'] == TASKS[-1]
                       for row in run.get('boundary_rows', [])) for run in runs)
    present = [run for run in runs if not run['missing']]
    fingerprints = {value for run in present for value in run.get('source_hashes', [])}
    game_versions_per_seed = {}
    for run in present:
        versions = {game['game_version'] for item in run['runtimes']
                    for game in item['runtime'].get('sc2_games', {}).values()
                    if game.get('game_version')}
        game_versions_per_seed[str(run['seed'])] = sorted(versions)
    observed_versions = {value for values in game_versions_per_seed.values() for value in values}
    report = {'schema_version': 1, 'protocol_tasks': TASKS, 'task_steps': BUDGETS,
              'expected_seeds': expected_seeds, 'missing_seeds': [run['seed'] for run in runs if run['missing']],
              'all_requested_seeds_completed_all_stages': complete,
              'paper_results_reproduced': None,
              'comparison_scope': 'Table 1 acquisition rates; final retention is in continual_matrix.csv',
              'paper_reference': PAPER_URL, 'paper_table1_comparison': acquisition,
              'alignment': 'stage-relative test-interval buckets; actual step ranges retained; no interpolation',
              'uncertainty': 'Sample standard deviation across measured seeds (ddof=1); null for one seed',
              'missing_data_policy': 'Missing seeds are disclosed, never replaced with zeros or paper values',
              'runtime_metadata_recorded_for_all_present_runs': any(not run['missing'] for run in runs) and
                  all(run.get('runtimes') for run in runs if not run['missing']),
              'observed_sc2_game_versions_by_seed': game_versions_per_seed,
              'actual_game_version_recorded_and_consistent': bool(present) and
                  all(game_versions_per_seed.values()) and len(observed_versions) == 1,
              'source_fingerprint_recorded_and_consistent': bool(present) and
                  all(run.get('source_hashes') for run in present) and len(fingerprints) == 1,
              'implementation_assumptions': 'Supplied paper lacks appendix; current hyperparameters and unseen-task embedding policy are reconstruction choices',
              'runs': [{key: value for key, value in run.items() if key not in ('records', 'boundary_rows')}
                       for run in runs]}
    write_json(output_dir / 'report.json', report)
    lines = ['# RPG SMAC measured results', '',
             'Protocol: 5m_vs_6m → 12m → 5m → 8m_vs_9m; budgets 1M / 0.5M / 0.5M / 0.5M.',
             f'Expected seeds: {expected_seeds}. All stages complete: {complete}.', '',
             '| Task acquisition endpoint | Paper Table 1 | Measured mean | Sample SD | Seeds | Gap (percentage points) |',
             '| --- | ---: | ---: | ---: | ---: | ---: |']
    for row in acquisition:
        def fmt(value, percent=False):
            return 'unavailable' if value is None else ('%.2f' % (value * 100 if percent else value))
        lines.append('| %s | %s%% | %s | %s | %s/%s | %s |' % (
            row['task'], fmt(row['paper_table_1_win_rate'], True),
            fmt(row['measured_mean_win_rate'], True), fmt(row['measured_sample_std'], True),
            row['n_seeds'], len(expected_seeds), fmt(row['gap_percentage_points'])))
    lines += ['', f'Paper targets: [IJCAI 2025, Table 1]({PAPER_URL}).', '',
              'Table 1 measures performance after acquiring each task. It must not be compared to the final retention score after later tasks.',
              'Curves group nearby observations by scheduled stage intervals and retain actual step ranges in CSV. Missing seeds are not imputed.',
              'A matching average alone does not establish reproduction: inspect SC2 version, source/config metadata, uncertainty, full curves, and forgetting.',
              'The supplied paper omits appendix settings. Exact author hyperparameters and zero-shot embedding selection remain unconfirmed.']
    (output_dir / 'REPORT.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    if plot:
        plot_curves(output_dir / 'learning_curves.png', curves)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', default='results/rpg_smac_full')
    parser.add_argument('--output', default=None)
    parser.add_argument('--seeds', type=int, nargs='+')
    parser.add_argument('--plot', action='store_true')
    args = parser.parse_args(argv)
    manifest = Path(args.input) / 'manifest.json'
    seeds = args.seeds or (read_json(manifest)['settings']['seeds'] if manifest.exists() else [0, 1, 2])
    if len(seeds) != len(set(seeds)):
        parser.error('Seeds must be unique')
    output = args.output or str(Path(args.input) / 'aggregate')
    report = summarize(args.input, output, seeds, args.plot)
    print(json.dumps({'report': str(Path(output).resolve()),
                      'all_stages_complete': report['all_requested_seeds_completed_all_stages'],
                      'missing_seeds': report['missing_seeds']}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
