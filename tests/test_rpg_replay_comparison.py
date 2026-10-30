from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from compare_rpg_replay import boundary_metrics, compare, validate_pair
from run_rpg_smac_experiments import TASKS


def test_pair_allows_only_replay_change_and_legacy_explicit_defaults():
    base = {'buffer_size': 256, 'seed': 0, 'lr': .0005}
    treatment = {**base, 'buffer_size': 5000, 'reset_epsilon': True,
                 'rpg_return_calibration': 'greedy_pretrain'}
    assert validate_pair(base, treatment) == {'buffer_size': [256, 5000]}
    for key, value in [('seed', 1), ('lr', .001), ('reset_epsilon', False),
                       ('rpg_return_calibration', 'training_episodes')]:
        with pytest.raises(ValueError, match='only buffer_size'):
            validate_pair(base, {**treatment, key: value})


def test_forgetting_uses_history_peak_and_keeps_improvement_signed():
    rows = [{'stage': s, 'step': s, 'task': t, 'success_rate': .5}
            for s in range(4) for t in TASKS[:s + 1]]
    for row in rows:
        if row['task'] == TASKS[0]:
            row['success_rate'] = [.4, .8, .7, .3][row['stage']]
        if row['task'] == TASKS[1] and row['stage'] == 3:
            row['success_rate'] = .9
    metrics = boundary_metrics(rows)
    assert metrics[TASKS[0]]['acquisition_drop_pp'] == pytest.approx(10)
    assert metrics[TASKS[0]]['forgetting_pp'] == pytest.approx(50)
    assert metrics[TASKS[1]]['forgetting_pp'] == pytest.approx(-40)
    assert metrics[TASKS[3]]['forgetting_pp'] is None
    assert boundary_metrics(rows[:-1]) is None


def test_missing_runs_produce_pending_report_not_fabricated_improvement(tmp_path):
    result = compare(tmp_path / 'old', tmp_path / 'new', tmp_path / 'out', [0, 1, 2])
    assert result['complete'] is False
    assert result['pending_seeds'] == [0, 1, 2]
    assert result['paired_metrics'] == result['aggregate'] == []
