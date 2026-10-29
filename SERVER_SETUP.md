# SMAC 服务器配置与完整实验

已在用户指定的 Ubuntu 20.04 服务器完成安装，四图真实对局、模型冒烟更新和 50 项测试
均通过，详见 [SMAC_IMAGE_VALIDATION.md](SMAC_IMAGE_VALIDATION.md)。该服务器为 CPU 模式，
新 GPU 实例仍需执行硬件检查。以下脚本可用于其他机器从头安装，完整训练结果尚未获得。

## 租用镜像和资源

AutoDL、矩池云等平台优先选择 **Ubuntu 22.04 + Python 3.10 + CUDA 12.1**。
镜像预装的 PyTorch 版本可以不同，安装脚本会建立独立 `.venv-smac` 并安装
PyTorch 2.5.1 CUDA 12.1 wheel。若平台只显示“PyTorch 2.1 / Python 3.10 /
CUDA 12.1 / Ubuntu 22.04”这样的组合，也可以选择，最终运行版本由独立环境决定。
这里是软件组合要求，不是已核实的某个平台实时镜像名称。

| 用法 | GPU | CPU | 内存 | 可用数据盘 |
| --- | --- | --- | --- | --- |
| 建议先租一台，三个种子依次训练 | RTX 3090 24GB 或 RTX 4090 24GB | 至少 16 vCPU | 至少 32GB | 至少 100GB |
| 后续并行三个种子 | 一张上述 GPU 起，先测实际显存占用 | 至少 24 vCPU | 建议 64GB | 至少 100GB |

这是针对当前实现的容量建议，不是实测最低要求。SMAC 的游戏进程和序列回放会占用
CPU、内存；需要先用真实 rollout 测每秒步数再估计总时间。不要根据 GPU 型号直接
推算多少小时能复现。CUDA 12.x 的驱动兼容要求见
[NVIDIA 官方表格](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html)，
PyTorch 2.5.1 的 cu121 安装命令见
[PyTorch 官方历史版本页面](https://pytorch.org/get-started/previous-versions/#v251)。

## 上传项目

将**当前改过的整个 MACPro 项目**上传服务器。只重新 `git clone` 原公开仓库
不会包含本地新增的 RPG 文件。上传时排除 `.venv`、所有 `.pytest-*`、`__pycache__`、
临时文件和旧 `results`；保留 `src/envs/smac/smac/env/starcraft2/maps/SMAC_Maps`。
普通训练时项目可以放在数据盘。**要保存为 AutoDL 私有镜像时，应将 Python 环境、
StarCraft II 和项目放在系统盘**，建议 `/root/rpg-smac-image/MACPro`。数据盘内容
不应当作镜像的一部分；安装前检查系统盘剩余容量。训练日志和检查点可以写入数据盘。
也可直接克隆本仓库：`git clone https://github.com/saimeeee/RPG-MACPro.git`。

## 安装

登录 Linux 服务器并进入已上传项目：

```bash
cd /root/rpg-smac-image/MACPro
bash scripts/server/setup_smac.sh --system-deps --accept-sc2-license
source results/server_setup/activate.sh
python scripts/check_rpg_smac.py --launch
```

`--system-deps` 安装 Ubuntu 系统库，使用 root 或 sudo。若镜像已包含系统库，可省略。
若 Python 3.10 位于 conda 环境，使用 `--python /实际路径/bin/python`。
脚本不要求在系统环境中替换原镜像的 PyTorch。

Blizzard 的 Linux 包有公开的 AI/ML 许可条款，解压密码表示接受条款；
`--accept-sc2-license` 表示你接受该条款并允许脚本解压。条款与下载来源见
[Blizzard 官方文档](https://github.com/Blizzard/s2client-proto#downloads)。
脚本仅使用其官方 HTTPS 下载地址，支持断点续传并保留缓存。
已安装游戏可用 `--sc2-path /实际路径/StarCraftII`，有效的现有安装无需重新解压。

默认安装 **StarCraft II 4.10**，路径为
`3rdparty/sc2-4.10/StarCraftII`。选择该版本的依据是 MACPro 安装脚本实际下载
4.10，以及[官方 PyMARL 安装脚本](https://github.com/oxwhirl/pymarl/blob/master/install_sc2.sh)
的设置。原 MACPro 脚本下载 4.10 却解压旧版本文件名，新增脚本已解决这个问题。
**RPG 正文没有确认精确游戏版本**；4.10 是当前实验设定，不能当作作者设置。
[SMAC 官方说明](https://github.com/oxwhirl/smac#readme)指出不同游戏版本的表现可能
不可直接比较。需要版本对照时，可在另一目录安装
`--sc2-version 4.6.2.69232`，并同步更改训练配置的 `env_args.game_version`；
不要混合不同版本的实验曲线。

安装将复制仓库里的真实地图。若目标目录已有同名不同内容的地图，会先保存带原
SHA256 前缀的备份。四个连续训练地图均有注册及文件；论文零样本任务 `15m`
仍缺真实地图，不能通过修改地图名称代替。

## 实验顺序

实际启动检查通过后，先做短训练和吞吐估计，再执行完整配置。完整实验使用四个
连续任务 `5m_vs_6m → 12m → 5m → 8m_vs_9m`，环境步预算分别为
100 万、50 万、50 万、50 万；三个种子总训练预算 750 万步，评估额外消耗时间。
这些预算来自论文图中的阶段边界。

推荐用 `tmux`，避免断开 SSH 导致训练中断：

```bash
tmux new -s rpg-smac
cd /root/rpg-smac-image/MACPro
bash scripts/server/run_full_smac.sh --output /root/autodl-tmp/rpg_smac_full
```

启动脚本先做真实游戏检查，再调用 `scripts/run_rpg_smac_experiments.py`。三个种子默认依次运行，
结果保留每个种子的日志和任务边界检查点，续训时加 `--resume`。后续会根据真实训练
吞吐选择是否用 `--max-parallel 3` 并行。`tmux` 中按 Ctrl-B 再按 D
可脱离，之后 `tmux attach -t rpg-smac` 恢复查看。

汇总已实际测得的曲线、三种子平均值和标准差：

```bash
python scripts/summarize_rpg_smac.py --input /root/autodl-tmp/rpg_smac_full --plot
```

任务完成时的胜率会与论文 Table 1 的 87.5%、100%、100%、91% 对照；所有后续任务
完成后的旧任务保持率另列，避免混淆。缺少种子的记录会明确标记，不能拿单个种子
补成三种子结果。每次训练另存 `runtime.json`，其中游戏版本来自实际引擎 ping。

安装清单保存在 `results/server_setup/`：

| 文件 | 用途 |
| --- | --- |
| `activate.sh` | 每次登录激活 Python 环境与 SC2PATH |
| `installation.json` | Python、包版本、GPU/驱动、二进制目录、Git 状态 |
| `pip-freeze.txt` | 依赖精确快照 |
| `sc2_archive.sha256` | 下载包哈希（重新使用既有安装时可能没有此项） |
| `sc2_binaries.sha256` | 实际游戏二进制哈希 |
| `maps.sha256` | 安装地图哈希 |

`installation.json` 仅表示安装清单；其中 `game_launch_validated=false` 不会因文件
下载完成就被误置为通过。真实启动记录由检查脚本另行输出。

论文缺少的正则系数等超参数仍使用 `README_RPG.md` 公开的实现设定。只有完整训练
和三种子统计完成后，才能判断是否接近论文结果；若失败，应先报告现有配置与差异，
再考虑有记录的参数实验。
