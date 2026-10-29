"""Continual RPG experiment driver reusing MACPro EpisodeBatch and ReplayBuffer.

Task-boundary checkpoints resume at the next task. No old experience is replayed.
SMAC is optional and imported only when selected. See README_RPG.md for scope.
"""
import argparse
import copy
import csv
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import random
import time
from types import SimpleNamespace

import numpy as np
import torch
import yaml

from components.episode_buffer import EpisodeBatch, ReplayBuffer
from components.transforms import OneHot
from controllers import REGISTRY as MAC_REGISTRY
from learners import REGISTRY as LEARNER_REGISTRY
from rpg_envs import build_env
from rpg_runtime import collect_runtime, record_observed_game, save_runtime

ROOT = Path(__file__).resolve().parents[1]


def task_name(spec):
    return spec if isinstance(spec, str) else spec['name']


class ExperimentLogger:
    def __init__(self, output):
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        self.console_logger = logging.getLogger('rpg')

    def write(self, filename, record):
        with (self.output / filename).open('a', encoding='utf-8') as f:
            f.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + '\n')

    def log_stat(self, name, value, t_env):
        self.write('metrics.jsonl', {'step': int(t_env), 'name': name, 'value': float(value)})


def make_scheme(info):
    scheme = {
        'state': {'vshape': info['state_shape']},
        'obs': {'vshape': info['obs_shape'], 'group': 'agents'},
        'actions': {'vshape': (1,), 'group': 'agents', 'dtype': torch.long},
        'avail_actions': {'vshape': (info['n_actions'],), 'group': 'agents', 'dtype': torch.int},
        'reward': {'vshape': (1,)},
        'terminated': {'vshape': (1,), 'dtype': torch.uint8},
    }
    return scheme, {'agents': info['n_agents']}, {'actions': ('actions_onehot', [OneHot(info['n_actions'])])}


class RPGRunner:
    def __init__(self, env, task, scheme, groups, preprocess, device):
        self.env, self.task, self.device = env, task, device
        self.scheme, self.groups, self.preprocess = scheme, groups, preprocess

    @torch.no_grad()
    def run(self, mac, t_env=0, test_mode=False, embedding_task=None):
        batch = EpisodeBatch(self.scheme, self.groups, 1, self.env.episode_limit + 1,
                             preprocess=self.preprocess, device=self.device)
        self.env.reset()
        mac.init_hidden(1, self.task)
        total_return, info = 0.0, {}
        for t in range(self.env.episode_limit):
            batch.update({'obs': np.asarray(self.env.get_obs())[None],
                          'state': np.asarray(self.env.get_state())[None],
                          'avail_actions': np.asarray(self.env.get_avail_actions())[None]}, ts=t)
            actions = mac.select_actions(batch, t, t_env, self.task, test_mode=test_mode,
                                         embedding_task=embedding_task)
            reward, done, info = self.env.step(actions[0].cpu().numpy())
            total_return += reward
            # A timeout retains bootstrapping; an actual terminal state does not.
            terminal = bool(done and not info.get('episode_limit', False))
            batch.update({'actions': actions.unsqueeze(-1), 'reward': [[float(reward)]],
                          'terminated': [[terminal]]}, ts=t)
            if done:
                break
        length = t + 1
        batch.update({'obs': np.asarray(self.env.get_obs())[None],
                      'state': np.asarray(self.env.get_state())[None],
                      'avail_actions': np.asarray(self.env.get_avail_actions())[None]}, ts=length)
        # MACPro filled marks stored states. The learner requires both t and
        # t+1 filled, so the final bootstrap row cannot become a transition.
        batch.update({'actions': torch.zeros(1, self.groups['agents'], 1, dtype=torch.long)},
                     ts=length, mark_filled=False)
        return batch, {'return': float(total_return), 'length': length,
                       'success': float(info.get('battle_won', info.get('success', total_return >= 0.999)))}


def evaluate(mac, runners, tasks, n_episodes, logger, step, stage, embedding_task=None):
    # MACPro's greedy selector still draws random numbers. Evaluation must not
    # alter the training RNG stream; environment RNGs are already separate.
    rng = (random.getstate(), np.random.get_state(), torch.get_rng_state(),
           torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)
    rows = []
    try:
        for task in tasks:
            episodes = [runners[task].run(mac, test_mode=True, embedding_task=embedding_task)[1]
                        for _ in range(n_episodes)]
            row = {'step': step, 'stage': stage, 'task': task, 'episodes': n_episodes,
                   'embedding_task': embedding_task or task,
                   'return_mean': float(np.mean([x['return'] for x in episodes])),
                   'return_std': float(np.std([x['return'] for x in episodes])),
                   'success_rate': float(np.mean([x['success'] for x in episodes]))}
            logger.write('evaluations.jsonl', row)
            rows.append(row)
        return rows
    finally:
        random.setstate(rng[0]); np.random.set_state(rng[1]); torch.set_rng_state(rng[2])
        if rng[3] is not None:
            torch.cuda.set_rng_state_all(rng[3])


def load_config(cli):
    if cli.checkpoint:
        config = json.loads((Path(cli.checkpoint) / 'driver.json').read_text(encoding='utf-8'))['config']
    else:
        with (ROOT / 'src/config/algs/qmix_rpg.yaml').open(encoding='utf-8') as f:
            config = yaml.safe_load(f)
        path = ROOT / 'src/config/tasks' / (cli.task_config + '.yaml')
        with path.open(encoding='utf-8') as f:
            config.update(yaml.safe_load(f))
        if config['env'] == 'sc2':
            with (ROOT / 'src/config/envs/sc2.yaml').open(encoding='utf-8') as f:
                defaults = yaml.safe_load(f)['env_args']
            defaults.update(config.get('env_args', {}))
            config['env_args'] = defaults
    for key in ('seed', 'device', 'test_nepisode'):
        if getattr(cli, key) is not None:
            if cli.checkpoint and key == 'seed' and cli.seed != config['seed']:
                raise ValueError('Checkpoint resumption preserves the original seed')
            config[key] = getattr(cli, key)
    if getattr(cli, 'game_version', None) is not None:
        old = config.get('env_args', {}).get('game_version')
        if cli.checkpoint and cli.game_version != old:
            raise ValueError('Checkpoint resumption preserves the game version')
        if config['env'] != 'sc2':
            raise ValueError('--game-version requires a SMAC task configuration')
        config.setdefault('env_args', {})['game_version'] = cli.game_version
    if cli.steps_per_task is not None:
        if cli.checkpoint:
            raise ValueError('Do not change step budgets when resuming a checkpoint')
        config['task_steps'] = [cli.steps_per_task] * len(config['tasks'])
    for option in cli.set:
        key, value = option.split('=', 1)
        if key not in config or isinstance(config[key], (dict, list)):
            raise ValueError('--set only accepts existing scalar configuration keys')
        if cli.checkpoint and key not in ('test_nepisode', 'torch_threads'):
            raise ValueError('Checkpoint resumption preserves algorithm hyperparameters')
        config[key] = yaml.safe_load(value)
    names = [task_name(s) for s in config['tasks']]
    if len(set(names)) != len(names) or len(names) != len(config['task_steps']):
        raise ValueError('Task names must be unique and match task_steps')
    for key in ('batch_size', 'buffer_size', 'test_nepisode', 'test_interval', 'updates_per_episode',
                'rpg_return_probe_episodes', 'rpg_importance_samples', 'target_update_interval'):
        if config[key] < 1:
            raise ValueError(key + ' must be positive')
    if config['buffer_size'] < config['batch_size'] or any(x < 1 for x in config['task_steps']):
        raise ValueError('Invalid replay capacity or task step budget')
    if config['return_max'] <= config['return_min']:
        raise ValueError('return_max must exceed return_min')
    if cli.zero_shot_embedding and not cli.evaluate:
        raise ValueError('Zero-shot evaluation requires --evaluate and --checkpoint')
    return config


def run_experiment(config, cli):
    device = config['device']
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but this PyTorch installation has no available CUDA device')
    config = copy.deepcopy(config)
    config['device'] = device
    torch.set_num_threads(config['torch_threads'])
    seed = config['seed']
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    output = Path(cli.output) if cli.output else ROOT / 'results/rpg' / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    if output.exists() and any(output.iterdir()):
        raise ValueError(f'Output directory is not empty: {output}; use a new directory')
    logger = ExperimentLogger(output)
    (output / 'config.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
    runtime = collect_runtime(ROOT, config)
    save_runtime(output, runtime)
    started_at = time.monotonic()
    args = SimpleNamespace(**config)
    train_tasks = [task_name(s) for s in config['tasks']]
    specs = list(config['tasks'])
    if cli.zero_shot_embedding:
        specs += config.get('generalization_tasks', [])
    names = [task_name(s) for s in specs]
    if len(names) != len(set(names)):
        raise ValueError('Training and generalization task names must be distinct')
    envs, runners, eval_runners, schemes, task_args, decomposers, layouts = {}, {}, {}, {}, {}, {}, {}
    all_envs = []
    try:
        for i, spec in enumerate(specs):
            name = task_name(spec)
            env, decomposer = build_env(spec, config['env'], config.get('env_args', {}), seed + i * 100)
            all_envs.append(env)
            eval_env, _ = build_env(spec, config['env'], config.get('env_args', {}), seed + 100000 + i * 100)
            all_envs.append(eval_env)
            info = env.get_env_info()
            scheme, groups, preprocess = make_scheme(info)
            # One small EpisodeBatch adds the derived one-hot field to the scheme.
            schemes[name] = EpisodeBatch(scheme, groups, 1, 1, preprocess=preprocess).scheme
            layouts[name] = (scheme, groups, preprocess)
            envs[name] = env
            decomposers[name] = decomposer
            ta = copy.deepcopy(config)
            ta.update(n_agents=info['n_agents'], n_actions=info['n_actions'], state_shape=info['state_shape'])
            task_args[name] = SimpleNamespace(**ta)
            runners[name] = RPGRunner(env, name, scheme, groups, preprocess, device)
            eval_runners[name] = RPGRunner(eval_env, name, scheme, groups, preprocess, device)
        mac = MAC_REGISTRY[config['mac']](train_tasks, schemes, task_args, args,
                                       task2decomposer={t: decomposers[t] for t in train_tasks}).to(device)
        learner = LEARNER_REGISTRY[config['learner']](mac, logger, args)
        total_steps, episode_num, start_stage, boundary_rows = 0, 0, 0, []
        if cli.checkpoint:
            learner.load_models(cli.checkpoint)
            progress = learner.progress
            total_steps, episode_num = progress['total_steps'], progress['episode_num']
            start_stage = progress['next_stage']
            boundary_rows = progress.get('boundary_rows', [])
            for task, state in progress.get('evaluation_env_states', {}).items():
                if hasattr(eval_runners[task].env, 'load_checkpoint_state'):
                    eval_runners[task].env.load_checkpoint_state(state)
        if cli.zero_shot_embedding:
            # Unseen task observations use a trained embedding explicitly;
            # do not allocate extra embedding parameters or alter the checkpoint.
            for task in names:
                if task not in train_tasks:
                    base, dec = mac.surrogate_decomposer, decomposers[task]
                    signature = ('own_obs_dim', 'obs_nf_al', 'obs_nf_en', 'n_actions_no_attack')
                    if any(getattr(base, k) != getattr(dec, k) for k in signature):
                        raise ValueError('Unseen task entity features do not match trained features')
                    mac.task2decomposer[task] = dec
                    mac.agent.task2decomposer[task] = dec
                    mac.task2n_agents[task] = task_args[task].n_agents
        if cli.evaluate:
            if not cli.checkpoint:
                raise ValueError('--evaluate requires --checkpoint')
            tasks = learner.completed_tasks
            embedding = cli.zero_shot_embedding
            if embedding:
                if embedding not in tasks:
                    raise ValueError('Zero-shot embedding must name a completed training task')
                tasks = [task_name(s) for s in config.get('generalization_tasks', [])]
                if not tasks:
                    raise ValueError('No generalization_tasks configured')
            evaluate(mac, eval_runners, tasks, config['test_nepisode'], logger, total_steps,
                     'zero_shot' if embedding else 'checkpoint', embedding)
            summary = {'mode': 'evaluation', 'tasks': tasks, 'checkpoint': str(Path(cli.checkpoint).resolve())}
        else:
            for stage in range(start_stage, len(train_tasks)):
                task = train_tasks[stage]
                learner.begin_task(task, train_tasks[stage - 1] if stage else None)
                scheme, groups, preprocess = layouts[task]
                buffer = ReplayBuffer(scheme, groups, config['buffer_size'], envs[task].episode_limit + 1,
                                      preprocess=preprocess, device='cpu')
                stage_steps, last_test = 0, 0
                last_console_step = total_steps
                # First N greedy episodes calibrate transfer ability before any
                # new-task gradient update. Their steps count toward the budget.
                for _ in range(config['rpg_return_probe_episodes']):
                    batch, result = runners[task].run(mac, test_mode=True)
                    learner.observe_return(float(np.clip((result['return'] - config['return_min']) /
                                                          (config['return_max'] - config['return_min']), 0, 1)))
                    batch.to('cpu'); buffer.insert_episode_batch(batch)
                    stage_steps += result['length']; total_steps += result['length']; episode_num += 1
                if stage_steps >= config['task_steps'][stage]:
                    raise ValueError('Task step budget exhausted by return probes; increase budget or reduce probes')
                record_observed_game(runtime, task, envs[task])
                save_runtime(output, runtime)
                logger.console_logger.info('Task %s (%s/%s), gamma_p=%.4f', task, stage+1, len(train_tasks), learner.gamma_p)
                updates_before = learner.train_steps
                while stage_steps < config['task_steps'][stage]:
                    batch, result = runners[task].run(mac, t_env=stage_steps)
                    batch.to('cpu'); buffer.insert_episode_batch(batch)
                    stage_steps += result['length']; total_steps += result['length']; episode_num += 1
                    logger.write('episodes.jsonl', {'task': task, 'step': total_steps, **result})
                    elapsed = time.monotonic() - started_at
                    if total_steps - last_console_step >= config['learner_log_interval']:
                        logger.console_logger.info('Task %s: %s/%s steps, total=%s, elapsed=%.0fs',
                                                   task, stage_steps, config['task_steps'][stage], total_steps, elapsed)
                        (output / 'progress.json').write_text(json.dumps({
                            'task': task, 'stage': stage, 'stage_steps': stage_steps,
                            'stage_budget': config['task_steps'][stage], 'total_steps': total_steps,
                            'elapsed_seconds_this_attempt': elapsed}), encoding='utf-8')
                        last_console_step = total_steps
                    if buffer.can_sample(config['batch_size']):
                        for _ in range(config['updates_per_episode']):
                            sample = buffer.sample(config['batch_size'])
                            # filled includes the final bootstrap state.
                            end = int(sample.max_t_filled().item())
                            sample = sample[:, :end]
                            sample.to(device)
                            metrics = learner.train(sample, total_steps, episode_num, task)
                            logger.write('updates.jsonl', {'task': task, 'step': total_steps, **metrics})
                    if stage_steps - last_test >= config['test_interval']:
                        evaluate(mac, eval_runners, train_tasks[:stage+1], config['test_nepisode'], logger, total_steps, stage)
                        last_test = stage_steps
                if learner.train_steps == updates_before:
                    raise RuntimeError('No optimizer update occurred; increase task budget or reduce batch_size')
                importance = learner.end_task(task, buffer)
                rows = evaluate(mac, eval_runners, train_tasks[:stage+1], config['test_nepisode'], logger, total_steps, stage)
                for seen in train_tasks[:stage+1]:
                    record_observed_game(runtime, 'evaluation/' + seen, eval_runners[seen].env)
                save_runtime(output, runtime)
                boundary_rows.extend(rows)
                learner.progress = {'total_steps': total_steps, 'episode_num': episode_num,
                                    'next_stage': stage+1, 'boundary_rows': boundary_rows,
                                    'evaluation_env_states': {
                                        name: runner.env.get_checkpoint_state()
                                        for name, runner in eval_runners.items()
                                        if hasattr(runner.env, 'get_checkpoint_state')}}
                checkpoint = output / 'checkpoints' / f'task_{stage:02d}'
                learner.save_models(checkpoint)
                (checkpoint / 'driver.json').write_text(json.dumps({'config': config, 'progress': learner.progress}, indent=2), encoding='utf-8')
                logger.console_logger.info('Completed %s: steps=%s, updates=%s, checkpoint=%s',
                                           task, stage_steps, learner.train_steps-updates_before, checkpoint)
                logger.write('boundaries.jsonl', {'task': task, 'step': total_steps, 'importance': importance})
            if boundary_rows:
                with (output / 'continual_matrix.csv').open('w', newline='', encoding='utf-8') as f:
                    writer = csv.DictWriter(f, fieldnames=boundary_rows[0].keys())
                    writer.writeheader(); writer.writerows(boundary_rows)
            summary = {'mode': 'training', 'tasks': train_tasks, 'completed_tasks': learner.completed_tasks,
                       'environment_steps': total_steps, 'episodes': episode_num, 'optimizer_updates': learner.train_steps,
                       'device': device, 'seed': seed, 'full_paper_results_reproduced': False,
                       'checkpoint_policy': 'task_boundaries_only',
                       'elapsed_seconds_this_attempt': time.monotonic() - started_at}
        (output / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
        logger.console_logger.info('Results: %s', output.resolve())
        return output
    finally:
        for env in all_envs:
            env.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='qmix_rpg', choices=['qmix_rpg'])
    parser.add_argument('--task-config', default='rpg_lbf_smoke')
    parser.add_argument('--seed', type=int)
    parser.add_argument('--device', choices=['cpu', 'cuda', 'auto'])
    parser.add_argument('--game-version', help='Explicit SC2 version, e.g. 4.10 (preserved when resuming)')
    parser.add_argument('--steps-per-task', type=int)
    parser.add_argument('--test-nepisode', type=int)
    parser.add_argument('--output')
    parser.add_argument('--checkpoint', help='Trusted local task boundary checkpoint directory')
    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('--zero-shot-embedding', help='Explicit trained task embedding used on unseen tasks')
    parser.add_argument('--set', action='append', default=[], metavar='KEY=VALUE')
    cli = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format='%(levelname)s %(message)s')
    run_experiment(load_config(cli), cli)


if __name__ == '__main__':
    main()
