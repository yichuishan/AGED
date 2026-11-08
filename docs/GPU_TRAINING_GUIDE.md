# GPU 训练：超参数、实验优先级与最快验证方法

适用于整合提交 `51d23d1` 及保留相同接口的后续版本。在仓库根目录、现有 GPU Python 环境中执行命令，无需为本指南重新安装 PyTorch。训练接口、short/long YAML 和旧版整合 v1 检查点保持兼容；默认设置也保持原样，**更新代码不会自动将 Adam 换成 SGD**。

## 依据与当前结论

组员提供的 `integration-checks.zip` 记录了旧提交 `224b948` 在 RTX 4070、真实 Human3.6M 文本导出上的实验。22 个流程检查成功，其中自动测试为 14 项；这与最新版 CPU 环境的 22 项自动测试通过不是同一计数。

- 训练、续训、验证、TensorBoard 和 S5 评估流程已在旧版 GPU 环境跑通。
- 短期零速度表及四类动作的长期基线吻合参考值。短期表对 sitting/160 ms 使用注明的推定勘误 0.63。
- 相同学习率 0.005 下，Adam 训练明显恶化，SGD 稳定得多。2000 步时，S11 平均误差：plain/Adam 10.155、plain/SGD 0.808、full/Adam 10.472、full/SGD 0.718；相同 S11 窗口的零速度为 0.704。
- 这支持优先验证优化器与学习率组合；不能证明作者使用了 SGD，也不能宣称当前 SGD 已复现论文。
- 旧记录没有第 0 步模型评估，不能据此断言训练结果比随机初始化还差。

原始 ZIP、日志和数据指纹由小组保留。本指南引用这份外部记录，本地 CPU 检查不构成重新执行 GPU 实验的证明。

## 哪些值固定，哪些需要验证

先固定数据、种子和训练预算，再逐项改变缺失的实现细节。所有候选在 S11 上选择，S5 留作最终报告。

| 参数 | 当前值 | 论文依据 / 验证原则 |
| --- | --- | --- |
| predictor 隐藏维度、batch、梯度裁剪 | 1024、16、5 | 论文明确；首轮保持不变。critic hidden 1024 也先固定，后者属于实现选择 |
| 学习率、λ、正式轮数 | 0.005、0.6、50 | 论文明确；优先用 0.005 验证 SGD。更小 Adam LR 属单独候选 |
| 观测 / 预测长度 | 50 / 10；长期 50 / 25 | 论文明确；先验证短期，再训练独立长期模型 |
| geodesic loss | 对时间、关节求和，batch 平均 | 公式（4）没有平方。不要为了降低数值改成平方或 mean |
| 对抗目标 | `minimax` | 论文目标；`non_saturating` 是需单独标注的工程变体 |
| `optimizer` | `adam` | 原文未命名；**P0：首先试 `sgd`** |
| `learning_rate_decay` / `every` | 1.0 / 10000 | 原文未给；稳定后试 0.95 / 10000 |
| `critic_optimizer` / `critic_learning_rate` | 跟随 G | 可分别设置；先保持跟随，失衡时单独调 D |
| critic 结构 / 优化器 | `projected` / 联合 | 对照团队 `direct` / 独立优化器，每次只改一项 |
| 抽样 / 最早起点 | `windows` / 0 | 原文未完整确定；可分别试 `recordings`、16 |
| `steps_per_epoch` | 2000 | 工程选择；比较总更新数，不只比较 epoch 数 |
| S11 / 验证窗口 | 可选 / 256，固定 seed 999 | 开发阶段启用；不参与训练或统计量拟合 |

学习率会同时影响 G 和 D，除非显式指定 `--critic-learning-rate`。若改用 Adam，建议先试 0.0005，必要时再试 0.0001；这些是待验证候选，不是论文给出的值。

## 实验优先级与组合

除表中显式修改项外，以下短跑都固定：all 15 actions、hidden 1024、batch 16、clip 5、λ 0.6、seed 0、S11 验证、minimax、geodesic sum、无衰减。诊断预算为 **2 epochs × 1000 steps = 2000 次更新**，每 1000 步得到一次验证和存档。

| 优先级 / 编号 | 组合 | 相对比较对象只改什么 | 何时执行 |
| --- | --- | --- | --- |
| P0 / A | full + SGD，LR 0.005 | 相对旧默认只换优化器 | 最先跑，检查主模型稳定性 |
| P0 / B | plain + SGD，LR 0.005 | 相对 A 去掉判别器，改为 MSE | 验证普通 predictor 能否学习；它是基准，不是单因素损失消融 |
| P1 / C | geodesic + SGD | 相对 A 只去掉两个判别器 | A 不稳定或需判断对抗训练是否拖累时 |
| P1 / D | full + Adam，LR 0.0005 | 相对旧 Adam 0.005 降低 LR | SGD 不理想或需要优化器对照时；不必重跑已知差的 Adam 0.005 |
| P1 / E | A + critic LR 0.001 | 只降低 D 的 LR，G 保持 0.005 | 判别器接近满分且 G 验证不改善时 |
| P2 / F | A + `critic_architecture=direct` | 只改 critic 结构 | 稳定基准建立后，验证成员实现 |
| P2 / G | A + 独立 critic 优化器 | 只加 `separate_critic_optimizers` | 同上；这也改变各 critic 的裁剪方式 |
| P2 / H | A + `sampling_strategy=recordings` | 只改记录抽样分布 | 同上；之后再单独试 `minimum_start=16` |
| P2 / I | 稳定候选 + 0.95/10000 衰减 | 只改 schedule | 必须训练超过 10000 步才有衰减对照意义 |
| P3 / J | A + `non_saturating` | 只改生成器对抗目标 | 稳定性扩展；单独标明偏离论文 minimax |

不要全排列。先完成 A，必要时 B、C；根据诊断选择后续项。表中 F/G/H/J 各自与 A 比较，不能把前一行的额外开关累积到下一行。`column` 只是 critic 输出形状选项，本轮没有必要作为效果搜索项。

D/E 的额外学习率设置和 J 的目标函数属于工程对照，不能直接标为采用论文全部原始超参数的结果。原文未规定的 schedule、结构细节和抽样选择也要随实验报告。

`configs/team-reference.yaml` 同时改变 critic、优化器分组、抽样、schedule、对抗目标和早停。先完成单项验证，再把它作为明确标注的团队组合实验；它不是完整确定的“论文原始配置”。

## 最快验证流程

以下命令为单行，可用于 Bash、Git Bash 或 PowerShell。保持数据根目录相同，每次使用新的 `--out`；已有有效 CUDA 环境无需运行安装脚本。

### 第一步：环境、版本与基线

```bash
git rev-parse HEAD
python -c "import torch; assert torch.cuda.is_available(); print(torch.__version__, torch.cuda.get_device_name(0))"
python -m pytest -q
python -m aged.evaluate --root data/h36m --baseline --check-baseline --output experiments/current-baseline.json
```

这是部署和评估检查；还不能证明学习效果。基线命令主要是 CPU 计算。若这个安装和数据已经通过，不必在每个候选前重复。测试套件包含需要可用 CUDA 的条件测试；实际执行数和跳过原因以日志为准。

### 第二步：只跑 A，约 2000 次更新

```bash
python -m aged.train --root data/h36m --config configs/short-term.yaml --variant full --optimizer sgd --val-subject S11 --epochs 2 --steps-per-epoch 1000 --save-every 1 --seed 0 --device cuda --out experiments/p0-A-full-sgd
```

历史 RTX 4070 的 full 短期训练约 40–45 ms/步，2000 步纯训练约 80–90 秒；还需加上读数据、验证和保存时间。此估计来自旧版，不能保证当前硬件或新增组合的耗时。预算最紧时，先完成这一项；只跑几十或 200 步可排查报错，不能判断收敛。

2×1000 是诊断预算，不是论文正式 epoch 定义；改变每轮步数也会改变 epoch 相关的抽样序列。不要要求它与旧脚本 1×2000 的逐步数值完全一致。新候选统一用 2×1000，保证彼此公平。

### 第三步：需要 predictor 基准时跑 B

```bash
python -m aged.train --root data/h36m --config configs/short-term.yaml --variant plain --optimizer sgd --val-subject S11 --epochs 2 --steps-per-epoch 1000 --save-every 1 --seed 0 --device cuda --out experiments/p0-B-plain-sgd
```

从 `results.json` 比较两轮的 `validation_angle_error`、`reconstruction` 和梯度；不要把 plain 的 MSE 数值与 full 的 geodesic 求和值直接比较。

### 后续候选的精确修改方式

以第二步 A 的完整命令为模板，替换 `--out` 并做以下单项修改；不要把这些片段单独当作命令运行：

| 编号 | 命令修改 |
| --- | --- |
| C | 将 `--variant full` 换成 `--variant geodesic` |
| D | 将 `--optimizer sgd` 换成 `--optimizer adam`，增加 `--learning-rate 0.0005` |
| E | 增加 `--critic-learning-rate 0.001` |
| F | 增加 `--critic-architecture direct` |
| G | 增加 `--separate-critic-optimizers` |
| H | 增加 `--sampling-strategy recordings`；在另一次独立比较中再加 `--minimum-start 16` |
| I | 增加 `--learning-rate-decay 0.95 --learning-rate-decay-every 10000`；稳定后的衰减/无衰减两组都运行例如 12×1000 步 |
| J | 增加 `--adversarial-objective non_saturating` |

## 如何最快判断结果

先看开发集 S11，不要用反复评估 S5 来选择超参数。

| 观察项 | 继续 / 排查标准 |
| --- | --- |
| 数值健康 | loss、验证误差有限，更新可执行。有限值只是最低门槛 |
| 学习趋势 | 比较第 1000/2000 步验证；小幅波动正常，误差跃升至旧记录约 10 的量级应暂停排查 |
| 相对零速度 | 使用当前配置相同 S11 窗口的基线。旧 0.704 仅供同数据同协议参考；未超过基线不等于程序坏了，也不能宣布复现成功 |
| 对抗状态 | 接近 100% 的 D 准确率 + minimax 项接近零 + G 不改善，才共同支持失衡判断。D 为 50% 也不等于成功 |
| 梯度 | 日志是裁剪前范数；很大需排查，但不能直接认定裁剪失效 |

下面的可选片段读取 A 的保存配置，重建**与训练验证相同**的 S11 窗口，计算零速度基线并汇总两轮模型误差。使用 Bash/Git Bash 粘贴；PowerShell 用户可将 Python 内容存为临时 `.py` 后用现有环境执行。测试另一候选时只改 `run`。

```bash
python - <<'PY'
import json
from pathlib import Path
import numpy as np
from aged.config import Config
from aged import data as D
from aged.evaluate import mean_angle_error

run = Path("experiments/p0-A-full-sgd")
cfg = Config(**json.loads((run / "config.json").read_text()))
assert cfg.val_subject is not None, "This comparison requires a validation run"
norm = D.Normalisation.from_dict(json.loads((run / "normalisation.json").read_text()))
val = D.WindowDataset("data/h36m", norm, [cfg.val_subject], cfg.actions,
                      cfg.observed_length, cfg.prediction_length,
                      cfg.validation_windows, 999,
                      cfg.sampling_strategy, cfg.minimum_start)
windows = [val[i] for i in range(len(val))]
observed = np.stack([w[0].numpy() for w in windows])
target = np.stack([w[1].numpy() for w in windows])
baseline = mean_angle_error(np.repeat(observed[:, -1:], cfg.prediction_length, axis=1), target, norm)
baseline_mean = float(np.mean(list(baseline.values())))
rows = []
for row in json.loads((run / "results.json").read_text())["history"]:
    errors = row["validation_angle_error"]
    average = float(np.mean(list(errors.values())))
    rows.append({"epoch": row["epoch"], "step": row["step"],
                 "per_horizon": errors, "mean": average,
                 "difference_from_zero_velocity": average - baseline_mean})
report = {"zero_velocity_s11": baseline, "zero_velocity_mean": baseline_mean, "model": rows}
(run / "validation-comparison.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
PY
```

差值为负表示模型平均误差低于同窗口零速度。S11 是随机开发窗口，S5 是固定八窗口且使用原始目标；不要把两种协议或不同采样配置的数值直接互比。片段未评估随机初始化模型；若要声称训练优于初始化，需要另外记录同一验证集的 step-0 结果。

2000 步的作用是筛掉明显不稳定候选。表现稳定但还略差于基线时，可延长到 5000–10000 步；不要仅凭两个点的小幅差异下结论。正式比较还需逐动作结果与预测可视化。

## 检查点、续训与正式实验

- 每个 epoch 结束覆盖 `last.pt`。启用 S11 后，验证均值改善时更新 `best.pt`，默认按 Euler 误差选择。
- 上面的 `--save-every 1` 额外保留每轮 `epoch_001.pt` 等，便于比较。当前没有按 step 自动存档。
- 同一实验延长训练时，使用保存的配置，不再传 `--config`：

```bash
python -m aged.train --root data/h36m --out experiments/p0-A-full-sgd --resume experiments/p0-A-full-sgd/last.pt --epochs 10 --device cuda
```

这会保留每轮 1000 步，总预算到 10000 步，且保留 S11 验证与每轮存档。改变优化器、LR、目标、采样、结构或每轮步数必须新开目录，不能通过 `--resume` 切换。`--init-from` 也要求相同 G 优化器类型，不是 Adam→SGD 的转换工具。

选定稳定设置后，再开展以下实验：

1. 在 S11 上用 seed 0/1/2 检查稳定性，报告均值和波动，不只保留最好的一次。
2. 固定最终设置，重新从头训练六个训练受试者；short-term YAML 下移除 `--val-subject S11`。若使用含 S11 的 team-reference YAML，则显式加 `--final`。重新拟合统计量，不能直接续训开发检查点来改变划分。
3. 正式预算可用 50×2000 次更新，并清楚标注 2000 为工程定义；此时无 S11，保存 last 和周期存档，不存在验证 best。严格跑满论文 50 epochs 时不要启用会提前结束的早停。
4. 对 25 帧独立长期模型使用 `configs/long-term.yaml`，重新验证稳定性。增加 `--horizon 100` 只是 rollout 能力，不代表已得到正确长期动作。
5. 在设置冻结后做 S5 定量评估、逐动作分析和可视化。重建统计量后 S11/S5 分数不应与开发阶段混为一个表。

每次保留代码 SHA、有效 `config.json`、命令、`results.json`、归一化、检查点和数据指纹。结论需注明属于“流程通过”“训练稳定”还是“达到论文指标”，三者分别需要证据。

相关文档：[论文设置与解释](PAPER_ALIGNMENT.md)、[成员贡献与兼容范围](TEAM_INTEGRATION.md)、[验证范围](VALIDATION.md)。
