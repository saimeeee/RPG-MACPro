# SMAC 私有镜像环境验证

用户指定服务器的环境已安装完成。所有运行所需文件均位于系统盘
`/root/rpg-smac-image/`，镜像审计未发现运行时数据盘依赖。

| 组件 | 实际版本 |
| --- | --- |
| 系统 | Ubuntu 20.04.4 LTS，x86_64 |
| Python | 3.10.22 |
| PyTorch | 2.5.1+cu121 |
| CUDA 编译版本 | 12.1 |
| PySC2 | 4.0.0 |
| StarCraft II 引擎实际报告 | 4.10.0.75689，Base75689 |
| NumPy / protobuf | 1.26.4 / 3.20.3 |

精确 Python 包快照保存为 `requirements-rpg-smac-tested-linux.lock.txt`。
游戏来自 Blizzard 官方 4.10 Linux 包，地图来自当前 MACPro 仓库。

实际执行并通过：

- 四个连续训练地图的真实随机策略对局：`5m_vs_6m`、`12m`、`5m`、`8m_vs_9m`。
- RPG 单游戏进程冒烟训练：`3m`，151 个环境步、2 次优化更新，完成参数重要性估计和模型保存。
- Linux/Python 3.10 目标环境中的完整自动测试：50 passed，4 条原 EpisodeBatch 警告，35.97 秒。
- 镜像审计：`software_ready=true`、`real_game_validated=true`、`errors=[]`。

当前实例处于 CPU 模式，分配 0.5 核、2GB 内存，没有暴露 GPU。
CUDA 版软件已完整安装，`gpu_validated=false`；须在新 GPU 实例上运行硬件检查。
上述冒烟训练验证安装和训练流程，不代表论文收敛结果；完整三种子实验尚未运行。

## 保存与复用

环境已经准备好，用户可在 AutoDL **关机 → 更多操作 → 保存镜像**。
建议命名：`RPG-SMAC410-Py310-Torch251-cu121`。新实例选择“我的镜像”中的这份环境。
[AutoDL 官方镜像说明](https://www.autodl.com/docs/image/)说明保存范围为系统盘。

镜像中的快捷入口：

```bash
# 激活环境
source /root/rpg-smac-image/activate.sh

# 新 GPU 实例上的实际游戏与 CUDA 检查
bash /root/rpg-smac-image/check-gpu.sh

# 完整三个种子实验，建议放在 tmux 会话中
tmux new -s rpg-smac
bash /root/rpg-smac-image/train.sh
```

完整训练结果写入数据盘 `/root/autodl-tmp/rpg_smac_full`，需要另行备份。
中断后使用 `bash /root/rpg-smac-image/train.sh --resume` 从最近完成的任务边界恢复。
镜像中的 `/root/rpg-smac-image/README_IMAGE.txt` 也保留了以上说明。

## 验证证据

服务器原件：`/root/rpg-smac-image/MACPro/results/server_setup/`。
本地副本：`results/server-image/server_setup/`，包含 `image-ready.json`、
`game-check.json`、`model-smoke/summary.json`、`tests.log`、依赖快照及地图/二进制哈希。

论文附录参数和 `15m` 自定义地图仍未获得；当前镜像支持四个连续训练任务及原仓库
已有的其他地图。后续论文比较需明确记录这些实现假设，不能仅凭环境通过即宣称成绩复现。
