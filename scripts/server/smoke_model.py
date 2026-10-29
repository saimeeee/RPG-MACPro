"""Small real-SMAC optimizer check using one game process in CPU-mode images.

This tests environment -> RPG -> replay -> TD update -> importance -> checkpoint.
It is an installation test, with small networks; it measures no paper performance.
"""
import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
import numpy as np
import torch
import yaml
from components.episode_buffer import EpisodeBatch, ReplayBuffer
from controllers import REGISTRY as MACS
from learners import REGISTRY as LEARNERS
from rpg_envs import build_env
from rpg_run import RPGRunner, ExperimentLogger, make_scheme
from rpg_runtime import collect_runtime, record_observed_game, save_runtime


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--game-version', default='4.10')
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        parser.error('Use a fresh output directory')
    torch.set_num_threads(1)
    torch.manual_seed(0)
    np.random.seed(0)
    config = yaml.safe_load((ROOT / 'src/config/algs/qmix_rpg.yaml').read_text())
    options = yaml.safe_load((ROOT / 'src/config/envs/sc2.yaml').read_text())['env_args']
    options['game_version'] = args.game_version
    task = '3m'
    config.update(env='sc2', tasks=[task], device='cpu', env_args=options,
                  rnn_hidden_dim=16, entity_embed_dim=16, attn_embed_dim=8,
                  rpg_head_hidden_dim=16, rpg_hyper_hidden_dim=16, rpg_task_embed_dim=8,
                  mixing_embed_dim=8, batch_size=2, buffer_size=4,
                  rpg_return_probe_episodes=2, rpg_importance_samples=2)
    logger = ExperimentLogger(args.output)
    (args.output / 'config.json').write_text(json.dumps(config, indent=2))
    env, dec = build_env(task, 'sc2', options, seed=0)
    try:
        info = env.get_env_info()
        scheme, groups, preprocess = make_scheme(info)
        derived = EpisodeBatch(scheme, groups, 1, 1, preprocess=preprocess).scheme
        task_args = SimpleNamespace(**dict(config, **{k: info[k] for k in ('n_agents', 'n_actions', 'state_shape')}))
        main_args = SimpleNamespace(**config)
        mac = MACS['rpg_mac']([task], {task: derived}, {task: task_args}, main_args, {task: dec})
        learner = LEARNERS['rpg_learner'](mac, logger, main_args)
        learner.begin_task(task)
        runner = RPGRunner(env, task, scheme, groups, preprocess, 'cpu')
        replay = ReplayBuffer(scheme, groups, 4, env.episode_limit + 1, preprocess=preprocess)
        runtime, steps = collect_runtime(ROOT, config), 0
        for episode in range(3):
            batch, result = runner.run(mac, steps, test_mode=episode < 2)
            record_observed_game(runtime, task, env)
            replay.insert_episode_batch(batch)
            steps += result['length']
            learner.observe_return(float(np.clip(result['return'] / 20, 0, 1)))
            if replay.can_sample(2):
                sample = replay.sample(2)
                sample = sample[:, :int(sample.max_t_filled().item())]
                metrics = learner.train(sample, steps, episode + 1, task)
                logger.write('updates.jsonl', dict(step=steps, **metrics))
        learner.end_task(task, replay)
        learner.save_models(args.output / 'checkpoint')
        save_runtime(args.output, runtime)
        summary = {'mode': 'training', 'validation_only': True, 'completed_tasks': [task],
                   'environment_steps': steps, 'optimizer_updates': learner.train_steps,
                   'device': 'cpu', 'full_paper_experiment_completed': False}
        (args.output / 'summary.json').write_text(json.dumps(summary, indent=2))
        print(json.dumps(summary))
    finally:
        env.close()


if __name__ == '__main__':
    main()
