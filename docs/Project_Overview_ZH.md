# LexiFAR 学术英语改写项目总览

**LexiFAR**（Lexically Guided Feature Fusion for Academic Rewriting）把不够正式的英文科研句子改成更准确、自然的学术表达，同时保留原文中的事实、数字、专业术语、否定和不确定性。模型在一次训练中学习四个阶段，把四级特征融合后生成一句改写。

早期实验比较过 BART 和 BART + Mamba。现在的训练主线是自建的六层因果模型，历史实验仍可在 Git 历史中查阅。比较结果时，需要区分模型、分词器、数据和解码设置。

## 模型怎么工作

![LexiFAR 模型，四级特征汇入一个最终输出](figures/LexiFAR_Architecture.svg)

[查看 LexiFAR 模型矢量图](figures/LexiFAR_Architecture.svg)。

| 预测点 | 实现 | 作用 |
| --- | --- | --- |
| P1 | 第一层 LSTM | 定位可能需要修改的单词，生成词级引导 E1。 |
| P2 | 第二层 LSTM | 学习单词替换或保留，生成 E2。例如根据语境替换 `get access` 中的 `get`，保留 `get stuck` 中的 `get`。 |
| P3 | 两层 Transformer | 通过辅助句级预测学习重述，并细化词分数以生成 E3。 |
| P4 | 融合 F1、F2、F3 与两层 Mamba 后的 F4 | 解码为唯一的最终改写句子。 |

第 1、2、4、6 个序列块后各有一个 MidFPred 投影。替代 CDEA 的词级模块由 P1、P2、P3 共享。模块利用原词和整句语境预测 `KEEP` 与各候选词的概率。每次级间引导由两个部分组成。

- **位置**用 `1 − P(KEEP)` 定位可能需要修改的词，并用一维膨胀减腐蚀突出子词边界。
- **替换内容**按概率加权求和候选词向量，再作为残差加到原词的子词位置。`KEEP` 对候选向量的贡献为零。同一个 `get` 可按语境传递 `gain` 或 `obtain` 的信息。

每个 token 位置的 F1–F4 特征拼接后，用一层可学习的 `4d → d` 投影融合成 P4 的输入。训练和缓存解码都从原文计算词级引导，参考答案用于训练监督。详见[结构图与联合训练说明](LexiFAR_Architecture.md#architecture-and-loss)。

句级训练以原文作前缀，只对参考改写的 token 计算交叉熵。常规训练从随机权重开始，字节级 BPE 分词器只使用两个训练语料拟合。推理按顺序贪心生成。Mamba 使用 PyTorch 实现的参考递推。

单句改写命令发现最终候选为空、未正常结束或数字发生变化时，会返回原文并记录原因。单位、论断的强弱和逻辑关系仍需核对。详见[模型结构](LexiFAR_Architecture.md)和[训练文档](LexiFAR_Training.md)。

## 数据从哪里来

| 数据 | 规模 | 用途与状态 |
| --- | --- | --- |
| `academic-pairs-v1` 训练集 | 12,491 对，其中 491 对真实修订、12,000 对合成句对 | 训练 P3 辅助句级目标和 P4 最终句级目标。真实修订来自 IteraTeR，混有风格、清晰度和流畅度等修改。合成样本覆盖 48 种语义模板和 24 个技术领域。 |
| `academic-pairs-v1` 验证集和测试集 | 98 对真实验证样本、87 对真实测试样本 | 按文档组划分。该版本完整保留了已有验证集和测试集的字节内容。两套数据都已用于诊断。 |
| `word-formality-v1` | 76 条合成样本，其中 56 条换词、20 条保留原词，覆盖 12 种输入词形 | 在同一次训练中监督 P1、P2、P3 的词级预测。独立词级评估需要另行准备经过人工复核的样本。 |
| `data/curation/` | IteraTeR 的 100 条初步审核记录，以及 arXivEdits 的冻结分区和来源记录 | 用于追溯数据来源和筛选过程。arXivEdits 记录中独立人工确认的新样本数为 0。 |

合成数据只用于训练。当前句对数据及其生成器哈希已冻结。调整语料或划分时，应建立新版本。新增论文原文和私有盲审映射保存在本地忽略目录。详见[数据卡](Dataset_Card.md)、[句对语料](../data/academic-pairs-v1/README.md)和[词级语料](../data/word-formality-v1/README.md)。

## 怎么训练和评价

每一步取一批句对与一批词级样本，做一次参数更新。默认总损失为 `0.5 × P3 辅助句级 CE + 0.5 × P4 最终句级 CE + 0.25 × 词级 CE`，其中词级 CE 取 P1、P2、P3 三次词级预测的分类损失平均值。训练结束得到一个检查点，推理只生成 P4 的最终句子。

[默认实验配置](../configs/six_layer_experiment.json)的隐藏维度为 192、批量大小为 16、学习率为 `3e-4`，最多训练 2,000 步，每 250 步验证一次，按 P4 的验证交叉熵选择检查点。测试评价最终句子与复制原文的基线，报告 ROUGE-L、可选的 SacreBLEU、复制率、数字保留率与截断诊断。提供独立词级评估样本时，还能单独核对词模块的准确率和误改率。

参照 **Doolittle（EMNLP 2023）** 的思路，计划分别评估最终改写的学术正式度、流畅度与语义保留，并把原样返回作为基线。建议采用两个独立评审者的盲评，同时检查数值、单位、否定、限定条件和结论是否改变。AESW、外部语言模型困惑度、`SIM-input` 和 `SIM-gold` 等指标已写入方案，尚待接入。论文的 MORL 用生成结果的多个指标计算奖励，再使用 PPO 更新模型。它可以作为融合模型的后续对照实验。详见[评价与 Loss 方案](LexiFAR_Architecture.md#evaluation-and-loss-informed-by-doolittle)。

## 在 PowerShell 中运行

从仓库根目录操作，需要 Python 3.10 或以上。首次安装运行下面的命令。已有可用的离线环境可以加上 `-Offline`。详见[环境配置](Environment_Setup.md)。

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/setup.ps1
```

下面先联合训练一次，再用同一个检查点试用与测试。再次训练时请改用新的输出目录。

```powershell
$python = '.\.venv\Scripts\python.exe'
$runDir = 'runs/six_layer_joint_v4'
$sentence = 'The proposed method gets better results than the baseline.'

# 联合训练一个检查点
& $python scripts/train_six_layer.py --output $runDir

# 生成并测试融合后的最终句子
& $python -m academic_rewriter.rewrite --checkpoint $runDir --text $sentence
& $python -m academic_rewriter.test --checkpoint $runDir --split test --limit 0
```

训练终端默认每步打印总损失、近 20 步平均损失、P3 和 P4 的句级交叉熵、词级交叉熵、学习率和耗时，并显示验证的开始与结果。加 `--log-every 10` 可改为每 10 步打印一次。`training_progress.jsonl` 保存逐步记录。

`--limit 0` 测试全部 87 条真实测试样本。`evaluation/test-192/` 中的 `predictions.csv` 和 `predictions.jsonl` 按行对齐原文、人工参考和最终改写。`metrics.json` 记录最终指标、复制基线、是否启用候选词引导和内部预测点 `[1, 2, 4, 6]`。改成 `--split validation` 可诊断全部 98 条验证样本。准备好独立人工复核的词级样本后可加 `--word-test .cache/reviewed_word_pairs.jsonl`。训练权重和预测保存在本地忽略目录。

要比较候选词向量的贡献，可在同样的设置下加 `--word-guidance position --output runs/six_layer_joint_v4_position` 训练位置引导的对照检查点，并用同一测试命令评估。详见[训练和测试命令](LexiFAR_Training.md#evaluate-the-entire-split)。

需要核验项目时，可在根目录手动执行以下命令。训练、推理、测试和其他项目脚本由用户执行，除非用户明确授权代跑。

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/verify_integrity.ps1
```

## 当前进度与维护

新训练默认启用候选词引导和最终特征融合。位置引导的旧检查点可按保存的配置加载。新增模块的训练、测试和位置引导对照待手动运行，独立人工盲评与 MORL 对照仍待实测。

以后每次完成一项项目修改，都要在同一提交里同步这份总览。更新受影响的章节，并在下面记录改动和实际验证情况。规则见[仓库维护说明](../AGENTS.md)。

| 日期 | 改动 | 验证与状态 |
| --- | --- | --- |
| 2026-11-21 | 训练终端默认逐步打印联合损失的各分项和进度，验证开始时提示，并支持 `--log-every` 调整频率。 | 已静态核对训练入口和配置。终端输出回归待手动运行。 |
| 2026-11-20 | 在 P1–P3 的词级引导中加入概率加权的候选词向量，提供位置引导对照选项。 | 已静态核对代码、配置和文档。候选词对齐、梯度、缓存解码、检查点加载和端到端回归待手动验证。 |
| 2026-11-19 | 将模型命名为 LexiFAR，重命名架构文档、训练文档和模型图，同步 README 和中文总览。 | 已核对文档链接、章节锚点和图中标题。 |
| 2026-11-19 | 把 F1–F4 拼接后融合，推理和测试只返回 P4。模型图同步标出四条特征路径和唯一的最终输出。 | 静态核对代码和图。融合梯度、缓存解码和单一输出的回归检查待手动运行。 |
| 2026-11-19 | 绘制四阶段模型矢量图，标出 P1 至 P4、E1 至 E3、词级融合和联合训练损失。 | 图中的预测点、连接和损失已与当前实现核对。 |
| 2026-11-19 | 按参考图补齐 P1 至 P4、四个中间投影和三次词级反馈，保留浅、中、深三个对外输出。 | 静态核对模型路径与数据。新增级间梯度、缓存解码、旧版兼容和统一测试检查，待手动运行。 |
| 2026-11-19 | 参考多阶段分割图，加入两次词级引导。三种输出联合训练成一个检查点，并在同一测试集并排比较。 | 静态核对训练入口、数据清单和输出。新增梯度、缓存解码和联合测试用例，待手动运行。 |
| 2026-11-19 | 建立中文项目总览，增加 README 入口和每次修改同步更新的仓库规则。 | 核对现有代码、数据清单、操作命令和链接。 |
| 2026-11-19 | 增加浅、中、深三种模式和独立词级模块，记录 Doolittle 评价与 Loss 方案。 | 实现已静态检查。训练、推理及外部指标的实测待完成。 |
| 2026-11-17 | 加入六层模型与 `academic-pairs-v1` 训练语料。 | 数据版本、分割与生成器指纹记录在语料清单中。 |
