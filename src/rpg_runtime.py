"""Record package/source identity and observed StarCraft engine versions."""
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import sys

import torch


def collect_runtime(root, config):
    packages = {}
    for name in ('torch', 'numpy', 'PyYAML', 'PySC2', 's2clientprotocol', 'protobuf', 'pygame', 'lbforaging'):
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            packages[name] = None
    files = [*Path(root).glob('src/rpg*.py'), *Path(root).glob('src/config/algs/qmix_rpg.yaml'),
             *Path(root).glob('src/config/tasks/rpg_marines.yaml'),
             *Path(root).glob('src/modules/agents/rpg_agent.py'),
             *Path(root).glob('src/modules/mixers/rpg_mixer.py'),
             *Path(root).glob('src/learners/rpg_learner.py'),
             *Path(root).glob('src/envs/smac/smac/env/starcraft2/starcraft2.py')]
    hashes = {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in sorted(files)}
    return {'schema_version': 1, 'python': sys.version, 'platform': platform.platform(),
            'packages': packages, 'device': config['device'], 'torch_cuda': torch.version.cuda,
            'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            'sc2_path': os.environ.get('SC2PATH'),
            'requested_sc2_version': config.get('env_args', {}).get('game_version'),
            'source_sha256': hashes, 'sc2_games': {},
            'paper_hyperparameters_confirmed': False,
            'protocol': 'RPG main-text equations; undocumented parameters are explicit implementation choices'}


def record_observed_game(runtime, task, env):
    """Read ping only after a real engine launch; never infer from a path name."""
    if task in runtime['sc2_games'] or getattr(env, '_controller', None) is None:
        return
    try:
        ping = env._controller.ping()
        runtime['sc2_games'][task] = {key: getattr(ping, key) for key in
                                    ('game_version', 'data_version', 'data_build', 'base_build')
                                    if hasattr(ping, key)}
    except Exception as exc:
        runtime['sc2_games'][task] = {'version_query_error': type(exc).__name__ + ': ' + str(exc)}


def save_runtime(output, runtime):
    path = Path(output) / 'runtime.json'
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(runtime, indent=2), encoding='utf-8')
    temporary.replace(path)
