"""Read-only SMAC assets preflight; does not launch StarCraft or train."""
import ast
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from rpg_envs import _require_starcraft_installation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--launch', action='store_true', help='Launch real games and play random episodes')
    parser.add_argument('--episodes', type=int, default=1)
    parser.add_argument('--game-version', default='4.10')
    parser.add_argument('--output', help='Write the actual preflight/rollout report as JSON')
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error('--episodes must be positive')
    base = ROOT / 'src/envs/smac/smac/env/starcraft2/maps'
    tree = ast.parse((base / 'smac_maps.py').read_text(encoding='utf-8'))
    registry = next(ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign)
                    and any(isinstance(t, ast.Name) and t.id == 'map_param_registry' for t in n.targets))
    train = ['5m_vs_6m', '12m', '5m', '8m_vs_9m']
    unseen = ['3m', '8m', '10m', '10m_vs_11m', '15m']
    maps = {name: {'registered': name in registry,
                   'bundled_map': (base / 'SMAC_Maps' / (name + '.SC2Map')).is_file()}
            for name in train + unseen}
    try:
        game = str(_require_starcraft_installation())
        game_error = None
    except FileNotFoundError as exc:
        game, game_error = None, str(exc)
    installed = {}
    for name in train + unseen:
        installed[name] = bool(game and (Path(game) / 'Maps/SMAC_Maps' / (name + '.SC2Map')).is_file())
    report = {'training_tasks': train, 'zero_shot_tasks': unseen,
                      'maps': maps, 'starcraft_path': game, 'error': game_error,
                      'installed_maps': installed,
                      'note': '15m is absent from upstream MACPro; supply the actual map and its registration to reproduce all of Table 2.'}
    ready = bool(game and all(installed[n] and maps[n]['registered'] for n in train))
    if args.launch and ready:
        import numpy as np
        import yaml
        from rpg_envs import build_env
        from rpg_runtime import record_observed_game
        with (ROOT / 'src/config/envs/sc2.yaml').open(encoding='utf-8') as f:
            options = yaml.safe_load(f)['env_args']
        options['game_version'] = args.game_version
        report['sc2_games'], report['rollouts'] = {}, []
        for name in train:
            env = None
            try:
                env, dec = build_env(name, 'sc2', options, seed=0)
                info = env.get_env_info()
                for episode in range(args.episodes):
                    env.reset()
                    record_observed_game(report, name, env)
                    if np.asarray(env.get_obs()).shape != (info['n_agents'], dec.obs_dim):
                        raise RuntimeError('Observation layout differs from decomposer')
                    reward_sum = 0.0
                    rng = np.random.default_rng(episode)
                    for step in range(env.episode_limit):
                        available = env.get_avail_actions()
                        actions = [int(rng.choice(np.flatnonzero(a))) for a in available]
                        reward, done, final_info = env.step(actions)
                        reward_sum += reward
                        if done:
                            break
                    report['rollouts'].append({'map': name, 'episode': episode, 'steps': step+1,
                                                'return': reward_sum, 'battle_won': final_info.get('battle_won'),
                                                'env_info': info})
            except Exception as exc:
                report['launch_error'] = type(exc).__name__ + ': ' + str(exc)
                ready = False
                break
            finally:
                if env is not None:
                    env.close()
    report['real_game_validation_passed'] = bool(args.launch and ready)
    if args.output:
        destination = Path(args.output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))
    return 0 if ready else 1


if __name__ == '__main__':
    raise SystemExit(main())
