"""Sequential independent experiments; full SMAC uses three seeds as in Fig.3."""
import argparse
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--task-config', default='rpg_marines')
parser.add_argument('--seeds', nargs='+', type=int, default=[0, 1, 2])
parser.add_argument('--device', default='auto', choices=['cpu', 'cuda', 'auto'])
parser.add_argument('--output', default='results/rpg_three_seeds')
args = parser.parse_args()
for seed in args.seeds:
    subprocess.run([sys.executable, str(ROOT / 'src/main.py'), '--config=qmix_rpg',
                    '--task-config=' + args.task_config, '--seed=' + str(seed),
                    '--device=' + args.device, '--output=' + str(Path(args.output) / f'seed_{seed}')],
                   cwd=ROOT, check=True)
