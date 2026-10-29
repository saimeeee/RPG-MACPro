# RPG-MACPro

RPG continual multi-agent reinforcement learning implementation based on [MACPro](https://github.com/lilh76/MACPro).

This repository implements **General Relation Patterns-Guided Task-specific Decision-Maker (RPG)** from the IJCAI 2025 paper [From General Relation Patterns to Task-Specific Decision-Making in Continual Multi-Agent Coordination](https://www.ijcai.org/proceedings/2025/759).
It reuses MACPro's episode batches, replay buffer, action selectors and SMAC interfaces while retaining the original project structure and Apache-2.0 license.

This is an **independent reproduction**, with explicit implementation choices where the supplied main paper omits settings. Full paper scores have not yet been reproduced. See [algorithm notes](README_RPG.md).

## Environment Installation

Use Python 3.10 for the Linux SMAC environment:

```bash
git clone https://github.com/saimeeee/RPG-MACPro.git
cd RPG-MACPro
bash scripts/server/setup_smac.sh --system-deps --accept-sc2-license
source results/server_setup/activate.sh
python scripts/check_rpg_smac.py --launch --output results/server_setup/game-check.json
```

The script creates `.venv-smac`, installs PyTorch 2.5.1 CUDA 12.1, downloads SC2 4.10 and copies the bundled maps. SC2 4.10 is an explicit reproduction choice; the supplied paper does not identify its exact build. See [server setup](SERVER_SETUP.md) for options, license information and reusable cloud images.

For CPU-only LBF experiments and tests, StarCraft II is optional:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-rpg.txt
```

RPG uses the official `lbforaging==2.0.0` package. The original `requirements.txt` and modified LBF environment are retained as upstream references; use `requirements-rpg*.txt` for the new algorithm.

## Run an Experiment

The entry point follows MACPro's configuration convention:

```bash
python src/main.py --config=qmix_rpg --task-config=rpg_marines --device=cuda
```

Configurations are under `src/config/algs` and `src/config/tasks`:

| Task config | Environment | Purpose |
| --- | --- | --- |
| `rpg_marines` | SMAC | Four continual tasks from Figure 3 |
| `rpg_lbf` | LBF | Explicit three-task research protocol |
| `rpg_lbf_smoke` | LBF | Short two-task integration check |

SMAC sequence: `5m_vs_6m → 12m → 5m → 8m_vs_9m`. Budgets: 1,000,000 / 500,000 / 500,000 / 500,000 environment steps per seed.

Run three seeds, recover at task boundaries and aggregate measured results:

```bash
python scripts/run_rpg_smac_experiments.py --output results/rpg_smac_full --device cuda
# Resume an interrupted experiment:
python scripts/run_rpg_smac_experiments.py --output results/rpg_smac_full --device cuda --resume
# Win-rate curves, sample standard deviations and continual-task matrices:
python scripts/summarize_rpg_smac.py --input results/rpg_smac_full --plot
```

Results and checkpoints are stored under the selected output directory. Installed environments, game archives and training outputs are excluded from this source repository.

Quick verification:

```bash
python src/main.py --config=qmix_rpg --task-config=rpg_lbf_smoke --device=cpu
python -m pytest -q --basetemp .pytest-tmp
```

## Implementation

| Component | Location |
| --- | --- |
| Dual relation attention, GRU and conditional hypernetwork | `src/modules/agents/rpg_agent.py` |
| Multi-agent controller | `src/controllers/rpg_controller.py` |
| Variable-entity multi-head monotonic QMIX | `src/modules/mixers/rpg_mixer.py` |
| TD learning and continual regularization, Eq. 3–12 | `src/learners/rpg_learner.py` |
| Continual training and checkpoint driver | `src/rpg_run.py` |
| Official LBF / bundled SMAC adapters | `src/rpg_envs.py` |
| Server setup, experiment launcher and result aggregation | `scripts/` |

## Validation and Reproduction Status

- **50 automated tests passed** on Windows and Linux/Python 3.10.
- Real SC2 **4.10.0.75689** episodes passed on all four training maps.
- A real-SMAC RPG installation test completed 151 steps and two optimizer updates, including checkpoint saving.
- The Linux image was verified in CPU mode. GPU hardware validation and the full three-seed experiment remain outstanding.
- The main paper omits appendix settings; adopted hyperparameters are documented. The `15m` zero-shot map is absent from upstream MACPro, and the unseen-task embedding policy must be specified explicitly.

Details: [algorithm notes](README_RPG.md), [validation record](VALIDATION_RPG.md), [SMAC image validation](SMAC_IMAGE_VALIDATION.md).
Original MACPro installation, examples and publication information: [README_MACPro.md](README_MACPro.md).

## Publication

Please cite the RPG paper and MACPro framework when using this reproduction:

```bibtex
@inproceedings{yao2025rpg,
  title = {From General Relation Patterns to Task-Specific Decision-Making in Continual Multi-Agent Coordination},
  author = {Chang Yao and Youfang Lin and Shoucheng Song and Hao Wu and Yuqing Ma and Sheng Han and Kai Lv},
  booktitle = {Proceedings of the Thirty-Fourth International Joint Conference on Artificial Intelligence},
  pages = {6821--6829},
  year = {2025},
  doi = {10.24963/ijcai.2025/759}
}

@article{macpro,
  title = {Multi-agent Continual Coordination via Progressive Task Contextualization},
  author = {Lei Yuan and Lihe Li and Ziqian Zhang and Fuxiang Zhang and Cong Guan and Yang Yu},
  journal = {IEEE Transactions on Neural Networks and Learning Systems},
  year = {2024}
}
```

## License

[Apache License 2.0](LICENSE), retaining upstream attribution and license files.
