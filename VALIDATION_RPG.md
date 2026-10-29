# RPG 验证记录

后续 Linux 服务器验证已完成：四个训练地图真实对局、151 步模型冒烟训练、2 次更新、
50 项自动测试通过。镜像路径审计通过，详见 [SMAC_IMAGE_VALIDATION.md](SMAC_IMAGE_VALIDATION.md)。
本文下面保留此前 Windows 本机验证记录；其中未安装 StarCraft II 的限制现仅适用于该工作站。

验证环境：Windows、Python 3.12.14、PyTorch 2.5.1+cpu、NumPy 1.26.4、
官方 lbforaging 2.0.0。依赖完整快照在 `requirements-rpg-tested.lock.txt`。
基准仓库提交：`950b7a259eb5fe171dcc8f3e24f5fd1448b92466`。

## 最终自动检查

```text
.venv\Scripts\python.exe -m pytest -q --basetemp .pytest-tmp-release
44 passed, 4 warnings in 10.53s

.venv\Scripts\python.exe -m pip check
No broken requirements found.

git diff --check
通过，无输出
```

四条 warning 来自原 `EpisodeBatch.update()` 使用 `torch.tensor(existing_tensor)`，
不是测试失败。测试内容为：

| 测试文件 | 数量 | 关键覆盖 |
| --- | ---: | --- |
| `tests/test_rpg_model.py` | 14 | 关系注意力排列性质、动态实体、梯度、embedding、单调 mixer |
| `tests/test_rpg_learner.py` | 8 | Eq.7/8/10、目标网络、动作掩码、padding、检查点状态 |
| `tests/test_rpg_envs.py` | 6 | 真实 LBF LOAD、团队回报、可见性、种子、超时 |
| `tests/test_rpg_driver.py` | 6 | 最后奖励、超时 bootstrap、逐位一致续训、CLI 评估、seed 保护、零样本入口 |
| `tests/test_rpg_smac_layout.py` | 10 | 原 SC2Decomposer 与八个原地图定义的张量接口，及可选时间特征 |

SMAC 测试读取真实地图注册表和分解器的 AST，并执行原分解器类；不安装、不启动
StarCraft，不模拟战斗结果。它验证模型输入输出维度与梯度，不验证游戏训练或胜率。

## 实际执行的训练

| 结果目录 | 任务数 | 训练环境步 | 优化次数 | 结论 |
| --- | ---: | ---: | ---: | --- |
| `results/rpg_smoke/` | 2 | 192 | 12 | 训练、切换、保存、评估闭环通过 |
| `results/rpg_smoke_resume/` | 恢复后剩余 1 | 累计 192 | 累计 12 | 从 task_00 继续到 task_01 通过 |
| `results/rpg_smoke_eval/` | 2 | 不训练 | 0 | 已保存模型独立评估通过 |
| `results/rpg_lbf_validation/` | 3 | 6000 | 111 | 2→3→4 智能体、1→2 食品的持续训练通过 |
| `results/rpg_smoke_three_seeds/seed_0` | 2 | 192 | 12 | seed 0 通过 |
| `results/rpg_smoke_three_seeds/seed_1` | 2 | 192 | 12 | seed 1 通过 |
| `results/rpg_smoke_three_seeds/seed_2` | 2 | 192 | 12 | seed 2 通过 |

6000 步验证的完整命令：

```text
.venv\Scripts\python.exe src/main.py --config=qmix_rpg --task-config=rpg_lbf --steps-per-task=2000 --set=batch_size=4 --set=buffer_size=32 --set=rpg_importance_samples=4 --set=rpg_return_probe_episodes=2 --test-nepisode=4 --output results/rpg_lbf_validation
```

这次三任务边界的 greedy 成功率均为 **0%**，短训练未收敛。最初探测回报为零，
故各阶段 `gamma_p=0`；非零 gamma 对关系正则的作用由解析测试验证。
不同种子的 192 步验证同样只用于集成检查。没有运行论文总计 250 万步 × 3 seeds
的 SMAC 实验，也没有生成看似来自论文的训练曲线或胜率表。

## 未完成的数值复现条件

- 未检测到 StarCraft II 游戏安装，实际 SMAC rollout 未运行。
- 原仓库缺少论文零样本任务 `15m` 的地图与注册；可运行零样本列表仅包含已有四图。
- 用户 PDF 不含附录：正则强度、网络尺寸、采样 N、LBF 任务序列和零样本 embedding
  规则等目前使用公开记录的工程设定。
- optional SMAC Python 依赖文件提供了安装路径，但当前主机没有验证整个 SC2 软件栈。
- 未开展完整三种子收敛训练、基线对比及消融实验。

因此当前交付是**实现与流程验证通过**，不应称为“论文实验结果已复现”。
