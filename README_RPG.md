# 在 MACPro 上复现 RPG（IJCAI 2025）

更新：已完成可保存为 AutoDL 私有镜像的 Linux 环境，四图真实对局、RPG 冒烟训练及
50 项测试通过。保存与复用见 [SMAC_IMAGE_VALIDATION.md](SMAC_IMAGE_VALIDATION.md)。
下文“本机未安装”的描述指 Windows 工作站；服务器已验证环境，完整论文训练仍待 GPU 实例。

本目录基于 [lilh76/MACPro](https://github.com/lilh76/MACPro)，基准提交为
`950b7a259eb5fe171dcc8f3e24f5fd1448b92466`。
新增 RPG 算法和持续训练入口，复用 MACPro 的 `EpisodeBatch`、`ReplayBuffer`、
动作选择器、控制器/学习器注册机制，以及 SMAC 环境和实体分解接口。

对应论文：Chang Yao 等，*From General Relation Patterns to Task-Specific
Decision-Making in Continual Multi-Agent Coordination*，IJCAI 2025，6821–6829。
[论文页面](https://www.ijcai.org/proceedings/2025/759)，
[arXiv](https://arxiv.org/abs/2507.06004)。实现依据用户提供的正文 PDF；该 PDF
没有正文多次提到的 Appendix。因此本项目是**公式级算法复现与可运行实现**，
不是作者原代码，也尚未完成论文数值结果的复现。

## 已实现的算法

| 论文部分 | 实现 | 代码 |
| --- | --- | --- |
| Eq. 1–2 | 自身作 Query，对队友和其他实体分别做 cross-attention；自身编码与两路关系拼接后进入 GRU | `src/modules/agents/rpg_agent.py` |
| Eq. 3 | 注意力熵加入训练损失，排除填充时间步；空集合和全部不可见集合数值稳定 | 同上及 learner |
| §4.2 | 学习任务 embedding，超网络生成 ego MLP 和逐实体 interaction MLP 的全部权重及偏置 | 同上 |
| Eq. 7 | 对旧任务各采样 episode 分别求 TD 梯度，平均 `(theta * gradient)^2`，再按整个 capturer 的全局最大值归一化 | `src/learners/rpg_learner.py` |
| Eq. 8 | 保存所有旧任务的参数 anchor 和 importance，求加权参数漂移平方和 | 同上 |
| Eq. 9 | 新任务更新前先执行 N 个 greedy episode，取归一化累计回报均值并冻结 `gamma_p` | `src/rpg_run.py` |
| Eq. 10 | 每个任务边界记录当前超网络在所有旧 embedding 上生成的参数；下阶段约束生成参数的漂移 | learner |
| Eq. 11 | `e_new = alpha_init * e_previous + (1-alpha_init) * Gaussian`，旧任务 embedding 冻结 | agent |
| Eq. 12 | `TD + alpha_att * entropy + gamma_p * L_rc + L_hy`；首任务两项防遗忘损失为零 | learner |
| §4.3 | 多头状态注意力适应不同实体数，非负 mixing weights 保持 QMIX 单调性；Double-Q 和目标网络 | `src/modules/mixers/rpg_mixer.py` |

旧任务不保留回放数据；每个新阶段重新分配当前任务 replay buffer。评估使用同一套
当前共享网络和各任务 embedding，不回滚到历史模型。interaction 输出按环境目标
实体顺序排列。LBF 的 LOAD 是一个固定动作，因此 LBF 始终输出六个 ego Q 值；
食品仍参与关系注意力。SMAC 输出六个 ego 动作和每个敌人的 attack Q 值。

## 在本机直接运行

先按下文安装步骤创建 `.venv` 并安装依赖。已验证 CPU PyTorch 2.5.1、NumPy 1.26.4、官方 LBF 2.0.0。
在 PowerShell 中：

```powershell
cd RPG-MACPro
.\.venv\Scripts\python.exe src/main.py --config=qmix_rpg --task-config=rpg_lbf_smoke
.\.venv\Scripts\python.exe -m pytest -q --basetemp .pytest-tmp
```

第一条执行两个任务的 192 步集成验证，默认会创建新的 `results/rpg/<时间>/`。
它用于快速验证安装、任务切换、训练、评估和保存，不用于判断算法性能。

三任务 LBF 训练（该序列是明确给出的实验协议，不是缺失附录中的作者序列）：

```powershell
.\.venv\Scripts\python.exe src/main.py --config=qmix_rpg --task-config=rpg_lbf --device=cpu
```

默认各训练 40 万环境步。可用 `--steps-per-task=2000` 做短验证；
用 `--set=batch_size=4` 等覆盖已有标量配置；所有最终配置写入结果目录。
每个阶段按完整 episode 结束，因此实际步数可能略超预算。最初 N 个探测 episode
计入预算并进入当前任务回放池，随后才开始优化；测试 episode 不计入训练步数。

在其他机器创建环境：

```text
python -m venv .venv
.venv\Scripts\python.exe -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
.venv\Scripts\python.exe -m pip install -r requirements-rpg.txt
```

GPU 训练需安装与驱动匹配的 CUDA PyTorch wheel，再使用 `--device=cuda`。
`--device=auto` 会自动选择可用设备；本次实际验证使用 CPU，没有 GPU 加速结果。
不要安装原仓库的旧 `requirements.txt` 来运行新增 RPG 路径，也不要将原仓库
修改过的 `src/envs/lb-foraging` 安装进这个环境。

## SMAC 论文任务

`src/config/tasks/rpg_marines.yaml` 使用论文 Figure 3 的任务顺序：

| 顺序 | 地图 | 环境步预算 |
| --- | --- | ---: |
| 1 | 5m_vs_6m | 1,000,000 |
| 2 | 12m | 500,000 |
| 3 | 5m | 500,000 |
| 4 | 8m_vs_9m | 500,000 |

阶段预算从 Figure 3/4 的训练背景边界读取。论文注明三组随机种子，具体种子值
未公布；脚本采用 0、1、2。原 MACPro 的 13m、4m 配置未被误用为本文设置。

本机未检测到 StarCraft II 可执行文件。先安装游戏，按照
[SMAC 官方说明](https://github.com/oxwhirl/smac)配置 `SC2PATH`，将仓库中的地图
复制到 `<SC2PATH>/Maps/SMAC_Maps/`。建议另建 Python 3.10 环境运行旧 SC2 软件栈，
安装 PyTorch 和 `requirements-rpg-smac.txt`。这些可选依赖及实际游戏训练尚未在本机
完成端到端验证。RPG 工厂会加载本仓库的 SMAC，不要求把整个原项目安装成包。

```text
python scripts/check_rpg_smac.py
python src/main.py --config=qmix_rpg --task-config=rpg_marines --device=auto
python scripts/run_rpg_seeds.py --task-config=rpg_marines --device=auto
```

第一条只检查地图和游戏文件，不启动游戏。四个训练地图的注册及文件都存在。
修正了 bundled SMAC 的两处 `np.bool`，以兼容 NumPy 1.26；修正了分解器在禁用
state last-action 时的 timestep 偏移；没有改动战斗规则。

## 保存、恢复和评估

每个任务结束生成 `checkpoints/task_XX/learner.th` 和 `driver.json`。检查点包含
当前与目标网络、优化器、任务 embedding、所有关系参数 anchor/importance、最近边界
生成参数 targets、回报校准、随机数状态和训练阶段。**CLI 仅在任务边界恢复**，不承诺
从某个任务中间恢复 replay 或游戏状态。

```powershell
.\.venv\Scripts\python.exe src/main.py --config=qmix_rpg --checkpoint results/rpg_smoke/checkpoints/task_00 --output results/my_resume
.\.venv\Scripts\python.exe src/main.py --config=qmix_rpg --checkpoint results/rpg_smoke/checkpoints/task_01 --evaluate --output results/my_eval
```

结果目录必须为空，避免覆盖已有实验。恢复自动读取原配置，禁止改变 seed 或算法
超参数。LBF 的评估 episode 序号也会恢复；SMAC 引擎内部随机状态未保存，所以
其恢复后的评估轨迹不保证逐条一致。测试验证了 LBF 任务边界恢复后，下一阶段的
网络、目标网络、优化器及持续学习状态与不中断训练逐位一致。

零样本评估必须明确指定使用哪个已训练任务的 embedding：

```text
python src/main.py --config=qmix_rpg --checkpoint <SMAC检查点目录> --evaluate --zero-shot-embedding=8m_vs_9m
```

这里不学习新 embedding、不反向传播、不按测试回报挑选 embedding。论文没有给出
未见任务的 embedding 选择方法，所以该选择属于实现假设。可运行配置列出
`3m, 8m, 10m, 10m_vs_11m`；论文 Table 2 还包含 `15m`，但 MACPro 没有该地图
文件和注册。需取得真实地图并注册后才能完整复现 Table 2；没有用其他地图代替。

## 输出文件

| 文件 | 内容 |
| --- | --- |
| `config.json` | 实际使用的全部配置 |
| `episodes.jsonl` | 训练 episode 的累计团队回报和长度 |
| `updates.jsonl` | 每次更新的 TD、attention、relation、hyper 损失及 gamma_p |
| `metrics.jsonl` | 按间隔记录的训练统计 |
| `evaluations.jsonl` | 当前模型在各已见任务的回报与成功率 |
| `continual_matrix.csv` | 每个任务边界对已见任务的评估表 |
| `boundaries.jsonl` | 重要性估计样本数与归一化前最大值 |
| `summary.json` | 步数、任务完成情况及验证范围 |
| `checkpoints/` | 可用于下一任务训练或评估的完整状态 |

已有验证结果：`results/rpg_smoke/`、`results/rpg_smoke_resume/`、
`results/rpg_smoke_eval/`、`results/rpg_lbf_validation/`。另用 seed 0、1、2 分别通过
192 步短验证，位于 `results/rpg_smoke_three_seeds/`；这也不等于论文三种子实验。
6000 步 LBF 验证的三任务均完成训练流程，共 111 次更新；各任务边界的 4 episode
greedy 评估成功率为 **0**，未收敛。其新任务初始回报均为零，因此 `gamma_p=0`，
按论文公式该次短跑的关系正则贡献为零；非零 gamma 的关系正则通过解析单元测试
验证。不能将这些短跑结果作为论文抗遗忘性能的证据。

## 未公开细节与本实现选择

正文没有公布以下数值或定义；全部在 YAML 和代码中公开，便于拿到附录后替换：

- 关系/隐藏维度 64、attention 维度 32；mixer 四头、mixing 维度 32。
- task embedding 维度 32，标准高斯初始化，`alpha_init=0.5`，旧 embedding 冻结。
- 超网络隐藏维度 64，生成两层 ReLU 决策 MLP；其他实体投影计入 capturer。
- `alpha_att=0.001`、`lambda_rc=100`、`lambda_hy=0.01`；生成参数损失按向量
  平方和计算，按旧任务数平均，没有擅自按参数数目再平均。
- 重要性用 32 个单独 episode 的 TD 梯度估计；episode 是采样单位，而非逐 transition。
- 回报探测 16 个 greedy episode；`clip((G-min)/(max-min),0,1)`；LBF 范围 [0,1]，
  SMAC reward scaling 下默认 [0,20]。SMAC 范围只是配置假设，并非从正文确认。
- RMSprop、学习率 0.0005、gamma 0.99、batch32、buffer256、target200episodes。
- LBF 使用官方 2.0.0；坐标按场地边长归一化，等级按队伍最大总等级归一化。
- 默认只覆盖实体特征类型相同、实体数量可变的任务族（如 Marines）。混合不同
  unit-type feature widths 会明确报错，需另做统一特征布局；不能无条件混合任意地图。
- 未实现论文全部对照算法、消融曲线和 t-SNE 图。本次不声称复现 Figure 3 或 Table 2 的数值。

## 检查范围

测试覆盖实体排列不变性/等变性、动态实体和动作数量、空/不可见实体、超网络梯度、
QMIX 单调性、Eq.7 全局归一化及逐样本平方、所有历史 anchor、Eq.10 最新边界目标、
冻结 embedding、Double-Q 动作掩码、终止与超时、padding、真实 LBF LOAD、断点恢复和 CLI 评估。
最终为 44 项测试通过，详细命令、结果及限制见 [VALIDATION_RPG.md](VALIDATION_RPG.md)。
可重复运行 `python -m pytest -q --basetemp .pytest-tmp`。

原仓库许可证保留在 `LICENSE`。原 MACPro 算法路径保留；其旧依赖与训练流程没有
经过本次全面迁移，新增 RPG 路径在 `src/main.py` 导入 Sacred 之前分流。
