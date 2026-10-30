"""Copy a first-task RPG checkpoint, changing only never-trained task identities.

This is a one-time, explicitly requested curriculum change, not ordinary resume.
All tensors, optimizer/target state, RNG states and completed-task evidence remain
unchanged. The original checkpoint is never modified.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path

import torch

OLD_TASKS = ['5m_vs_6m', '12m', '5m', '8m_vs_9m']
NEW_TASKS = ['5m_vs_6m', '13m', '4m', '8m_vs_9m']
BUDGETS = [1000000, 500000, 500000, 500000]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def assert_equal(left, right):
    if isinstance(left, torch.Tensor):
        assert isinstance(right, torch.Tensor)
        assert left.dtype == right.dtype and left.shape == right.shape
        assert torch.equal(left.cpu(), right.cpu())
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            assert_equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert type(left) is type(right) and len(left) == len(right)
        for a, b in zip(left, right):
            assert_equal(a, b)
    else:
        assert left == right


def transform(driver, payload, seed):
    config, progress = driver['config'], driver['progress']
    if (config.get('env') != 'sc2' or config.get('tasks') != OLD_TASKS or
            config.get('task_steps') != BUDGETS or config.get('seed') != seed or
            config.get('buffer_size') != 5000):
        raise ValueError('Expected the original seed-specific replay-5000 SMAC configuration')
    if progress.get('next_stage') != 1 or progress.get('total_steps', 0) < BUDGETS[0]:
        raise ValueError('Only a completed first-task boundary may change future maps')
    if payload.get('completed_tasks') != [OLD_TASKS[0]] or payload.get('current_task') != OLD_TASKS[0]:
        raise ValueError('Checkpoint has trained tasks beyond the shared first task')
    assert_equal(progress, payload['progress'])
    if any(row['task'] != OLD_TASKS[0] for row in progress.get('boundary_rows', [])):
        raise ValueError('Unexpected completed-task evaluation history')
    if progress.get('evaluation_env_states'):
        raise ValueError('This converter requires the SMAC checkpoint layout without serialized simulator states')
    for key in ('relation_anchors', 'relation_importance', 'hyper_targets'):
        if set(payload[key]) != {OLD_TASKS[0]}:
            raise ValueError('Unexpected protected task in ' + key)
    expected = {task: 't' + str(i) for i, task in enumerate(OLD_TASKS)}
    for key in ('agent', 'target_agent'):
        extra = payload[key]['_extra_state']
        if (extra['task_keys'] != expected or extra['begun_tasks'] != [OLD_TASKS[0]] or
                extra['active_task'] != OLD_TASKS[0]):
            raise ValueError('Agent/target task identities are not at the first boundary')

    converted_driver, converted = copy.deepcopy(driver), copy.deepcopy(payload)
    converted_driver['config']['tasks'] = list(NEW_TASKS)
    for key in ('agent', 'target_agent'):
        converted[key]['_extra_state']['task_keys'] = {task: 't' + str(i) for i, task in enumerate(NEW_TASKS)}

    # Prove that undoing exactly the three metadata edits restores the full source.
    check_driver, check = copy.deepcopy(converted_driver), copy.deepcopy(converted)
    check_driver['config']['tasks'] = list(OLD_TASKS)
    for key in ('agent', 'target_agent'):
        check[key]['_extra_state']['task_keys'] = expected
    assert_equal(driver, check_driver)
    assert_equal(payload, check)
    return converted_driver, converted


def migrate(source, destination, seed):
    source, destination = Path(source), Path(destination)
    if destination.exists() or destination.with_name(destination.name + '.partial').exists():
        raise FileExistsError('Destination or partial migration already exists')
    driver = json.loads((source / 'driver.json').read_text(encoding='utf-8'))
    payload = torch.load(source / 'learner.th', map_location='cpu', weights_only=True)
    converted_driver, converted = transform(driver, payload, seed)
    temporary = destination.with_name(destination.name + '.partial')
    temporary.mkdir(parents=True)
    torch.save(converted, temporary / 'learner.th')
    (temporary / 'driver.json').write_text(json.dumps(converted_driver, indent=2), encoding='utf-8')
    restored = torch.load(temporary / 'learner.th', map_location='cpu', weights_only=True)
    assert_equal(converted, restored)
    provenance = {
        'source': str(source.resolve()), 'source_driver_sha256': digest(source / 'driver.json'),
        'source_learner_sha256': digest(source / 'learner.th'),
        'converted_learner_sha256': digest(temporary / 'learner.th'),
        'seed': seed, 'old_tasks': OLD_TASKS, 'new_tasks': NEW_TASKS,
        'task_steps': BUDGETS, 'preserved_total_steps': driver['progress']['total_steps'],
        'preserved_optimizer_updates': payload['train_steps'],
        'tensor_optimizer_target_rng_equality_verified': True,
        'note': 'Only config.tasks and agent/target _extra_state.task_keys changed. Simulator RNG is not checkpointed by SMAC.',
    }
    (temporary / 'migration.json').write_text(json.dumps(provenance, indent=2), encoding='utf-8')
    temporary.replace(destination)
    return provenance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True, type=Path)
    parser.add_argument('--destination', required=True, type=Path)
    parser.add_argument('--seed', required=True, type=int)
    args = parser.parse_args()
    print(json.dumps(migrate(args.source, args.destination, args.seed), indent=2))


if __name__ == '__main__':
    main()
