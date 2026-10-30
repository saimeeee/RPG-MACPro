"""Paired comparison of the 256/5000 replay runs; never fill missing seeds."""
import argparse
import json
import math
from pathlib import Path

from run_rpg_smac_experiments import TASKS, write_json
from summarize_rpg_smac import load_seed, measured_summary, write_csv


def normalized_config(config):
    result = dict(config)
    result.setdefault('reset_epsilon', True)
    result.setdefault('rpg_return_calibration', 'greedy_pretrain')
    return result


def validate_pair(before, after):
    before, after = normalized_config(before), normalized_config(after)
    differences = {k: [before.get(k), after.get(k)] for k in before.keys() | after.keys()
                   if before.get(k) != after.get(k)}
    if differences != {'buffer_size': [256, 5000]}:
        raise ValueError('Expected only buffer_size 256 -> 5000; actual differences: ' + str(differences))
    return differences


def boundary_metrics(rows):
    endpoints = {(int(r['stage']), r['task']): float(r['success_rate'])
                 for r in sorted(rows, key=lambda r: r['step'])}
    required = {(stage, task) for stage in range(len(TASKS)) for task in TASKS[:stage + 1]}
    if not required <= endpoints.keys():
        return None
    if any(not math.isfinite(v) or not 0 <= v <= 1 for v in endpoints.values()):
        raise ValueError('Invalid boundary win rate')
    result = {}
    final_stage = len(TASKS) - 1
    for i, task in enumerate(TASKS):
        acquisition = endpoints[i, task]
        retention = endpoints[final_stage, task]
        # Signed forgetting over stage boundaries, excluding the final stage
        # from the historical maximum. Negative means improvement, not an error.
        forgetting = (max(endpoints[s, task] for s in range(i, final_stage)) - retention
                      if i < final_stage else None)
        result[task] = {'acquisition_pp': 100 * acquisition, 'retention_pp': 100 * retention,
                        'acquisition_drop_pp': 100 * (acquisition - retention),
                        'forgetting_pp': None if forgetting is None else 100 * forgetting}
    return result


def compare(baseline, treatment, output, seeds):
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError('Seeds must be nonempty and unique')
    output = Path(output)
    pairs, pending, identity = [], [], []
    for seed in seeds:
        old = load_seed(Path(baseline) / f'seed_{seed}', seed)
        new = load_seed(Path(treatment) / f'seed_{seed}', seed)
        if old['missing'] or new['missing']:
            pending.append(seed)
            continue
        diff = validate_pair(old['config'], new['config'])
        identity.append({'seed': seed, 'config_differences': diff,
                         'baseline_source_fingerprints': old['source_hashes'],
                         'treatment_source_fingerprints': new['source_hashes']})
        a, b = boundary_metrics(old['boundary_rows']), boundary_metrics(new['boundary_rows'])
        if a is None or b is None:
            pending.append(seed)
            continue
        for task in TASKS:
            pairs.append({'seed': seed, 'task': task,
                          **{'baseline_' + k: v for k, v in a[task].items()},
                          **{'treatment_' + k: v for k, v in b[task].items()},
                          'acquisition_gain_pp': b[task]['acquisition_pp'] - a[task]['acquisition_pp'],
                          'retention_gain_pp': b[task]['retention_pp'] - a[task]['retention_pp'],
                          'forgetting_reduction_pp': (a[task]['forgetting_pp'] - b[task]['forgetting_pp']
                                                      if a[task]['forgetting_pp'] is not None else None)})
    aggregated = []
    for task in TASKS:
        selected = [r for r in pairs if r['task'] == task]
        if not selected:
            continue
        row = {'task': task, 'n_seeds': len(selected), 'complete': len(selected) == len(seeds)}
        for key in selected[0]:
            if key in ('task', 'seed') or selected[0][key] is None:
                continue
            stat = measured_summary(r[key] for r in selected)
            row[key + '_mean'] = stat['mean']
            row[key + '_sample_std'] = stat['sample_std']
        aggregated.append(row)
    report = {'complete': not pending, 'expected_seeds': seeds, 'pending_seeds': pending,
              'identity': identity, 'paired_metrics': pairs, 'aggregate': aggregated,
              'units': 'percentage points',
              'forgetting_definition': 'max previous task-boundary win rate - final rate; signed; excludes newest task',
              'source_note': 'Source fingerprints may differ due to audited diagnostics. Same config is not proof of identical code.',
              'interpretation': 'Positive acquisition/retention gain or forgetting reduction favors 5000; 3 seeds do not establish significance.'}
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / 'comparison.json', report)
    write_csv(output / 'paired_metrics.csv', pairs)
    write_csv(output / 'aggregate.csv', aggregated)
    lines = ['# Replay 256 → 5000 paired comparison', '',
             f'Complete: {not pending}; pending seeds: {pending}. Units: percentage points.', '',
             '| Task | Seeds | Acquisition gain | Retention gain | Forgetting reduction |',
             '| --- | ---: | ---: | ---: | ---: |']
    for row in aggregated:
        def fmt(key):
            value = row.get(key + '_mean')
            if value is None:
                return 'not applicable'
            sd = row.get(key + '_sample_std')
            return f'{value:+.2f}' + (f' ± {sd:.2f}' if sd is not None else ' (SD unavailable)')
        lines.append(f"| {row['task']} | {row['n_seeds']}/{len(seeds)} | {fmt('acquisition_gain_pp')} | "
                     f"{fmt('retention_gain_pp')} | {fmt('forgetting_reduction_pp')} |")
    lines += ['', 'Values are paired per-seed differences, with sample SD (ddof=1).',
              report['forgetting_definition'], report['interpretation'], report['source_note']]
    (output / 'REPORT.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', required=True)
    parser.add_argument('--treatment', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
    args = parser.parse_args()
    report = compare(args.baseline, args.treatment, args.output, args.seeds)
    print(json.dumps({'complete': report['complete'], 'pending_seeds': report['pending_seeds']}))
    raise SystemExit(0 if report['complete'] else 2)
