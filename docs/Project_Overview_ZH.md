# 学术英语改写项目总览

本项目把不够正式的英文科研句子改成更准确、自然的学术表达，同时保留原文中的事实、数字、专业术语、否定和不确定性。当前以单句为单位，提供浅层换词、中层改写和深层改写三种模式。

早期实验比较过 BART 和 BART + Mamba。现在的训练主线是自建的六层因果模型，历史实验仍可在 Git 历史中查阅。比较结果时，需要区分模型、分词器、数据和解码设置。

## 模型怎么工作

| 层级 | 实现 | 适用场景 |
| --- | --- | --- |
| 浅层 `shallow` | 独立的词级模块，根据局部上下文学会替换或保留某个英语单词。每处修改只替换一个词，其余文字、空格和标点保持原样。 | 个别词不够正式，句式本身合适。例如在合适的语境中把 `get access` 改成 `gain access`，在 `get stuck` 中保留 `get`。 |
| 中层 `medium` | 运行前四个序列块，即两层 LSTM 和两层 Transformer。使用单独训练的中层检查点。 | 以局部重述为目标，例如调整短语和句内表达。实际修改幅度需要评估。 |
| 深层 `deep` | 运行完整六个序列块，即两层 LSTM、两层 Transformer 和两层 Mamba。旧的六层检查点默认归入深层。 | 以更大范围的句式调整为目标。当前以单句改写为主。 |

中层和深层共用相同的句对数据与训练目标。训练时把原句放在前缀位置，只对参考改写的 token 计算交叉熵。常规训练从随机权重开始，字节级 BPE 分词器只使用训练文本拟合。推理时按顺序贪心生成。Mamba 使用 PyTorch 实现的参考递推。

中层和深层的单句改写命令发现候选为空、未正常结束或数字发生变化时，会返回原文并记录原因。单位、论断的强弱和逻辑关系仍需核对。详见[三级使用说明](Three_Level_Model.md)和[六层架构](Six_Layer_Model.md)。

## 数据从哪里来

| 数据 | 规模 | 用途与状态 |
| --- | --- | --- |
| `academic-pairs-v1` 训练集 | 12,491 对，其中 491 对真实修订、12,000 对合成句对 | 训练中层和深层。真实修订来自 IteraTeR，混有风格、清晰度和流畅度等修改。合成样本覆盖 48 种语义模板和 24 个技术领域。 |
| `academic-pairs-v1` 验证集和测试集 | 98 对真实验证样本、87 对真实测试样本 | 按文档组划分。该版本完整保留了已有验证集和测试集的字节内容。两套数据都已用于诊断。 |
| `word-formality-v1` | 76 条合成样本，其中 56 条换词、20 条保留原词，覆盖 12 种输入词形 | 只训练浅层词级模块。独立词级评估需要另行准备经过人工复核的样本。 |
| `data/curation/` | IteraTeR 的 100 条初步审核记录，以及 arXivEdits 的冻结分区和来源记录 | 用于追溯数据来源和筛选过程。arXivEdits 记录中独立人工确认的新样本数为 0。 |

合成数据只用于训练。当前句对数据及其生成器哈希已冻结。调整语料或划分时，应建立新版本。新增论文原文和私有盲审映射保存在本地忽略目录。详见[数据卡](Dataset_Card.md)、[句对语料](../data/academic-pairs-v1/README.md)和[词级语料](../data/word-formality-v1/README.md)。

## 怎么训练和评价

浅层从上下文中的换词和保留样本学习计数。评估时分别报告需要改动样本的目标词替换准确率、无需改动样本的误改率，以及整句完全匹配率。

中层和深层用目标 token 交叉熵作监督基线。[默认实验配置](../configs/six_layer_experiment.json)的隐藏维度为 192、批量大小为 16、学习率为 `3e-4`，最多训练 2,000 步，每 250 步验证一次，按验证交叉熵选择检查点。交叉熵衡量模型预测参考改写的能力。生成结果另看 ROUGE-L、可选的 SacreBLEU、复制率、数字保留率与截断诊断。

参照 **Doolittle（EMNLP 2023）** 的思路，中层和深层计划在同一批输入上比较学术正式度、流畅度与语义保留，并把原样返回作为基线。建议采用两个独立评审者的盲评，同时检查数值、单位、否定、限定条件和结论是否改变。AESW、外部语言模型困惑度、`SIM-input` 和 `SIM-gold` 等指标已写入方案，尚待接入。论文的 MORL 用生成结果的多个指标计算奖励，再使用 PPO 更新模型。本项目将它作为深层的后续对照实验。详见[评价与 Loss 方案](Three_Level_Model.md#evaluation-and-loss-informed-by-doolittle)。

## 在 PowerShell 中运行

从仓库根目录操作，需要 Python 3.10 或以上。首次安装运行下面的命令。已有可用的离线环境可以加上 `-Offline`。详见[环境配置](Environment_Setup.md)。

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/setup.ps1
```

以下命令分别训练、试用和评估。`runs/overview_v1` 是示例路径，再次训练时请改用新的输出目录。已有合适的检查点时，可直接把 `--checkpoint` 改成它的路径。

```powershell
$python = '.\.venv\Scripts\python.exe'
$runDir = 'runs/overview_v1'
$sentence = 'The proposed method gets better results than the baseline.'

# 浅层训练和试用
& $python -m academic_rewriter.lexical train --data data/word-formality-v1/train.jsonl --output "$runDir/word_model.json"
& $python -m academic_rewriter.rewrite --depth shallow --lexical-model "$runDir/word_model.json" --text $sentence

# 中层和深层分别训练
& $python scripts/train_six_layer.py --depth medium --output "$runDir/medium"
& $python scripts/train_six_layer.py --depth deep --output "$runDir/deep"

# 深层改写示例与全量验证集诊断
& $python -m academic_rewriter.rewrite --depth deep --checkpoint "$runDir/deep" --text $sentence
& $python -m academic_rewriter.test --checkpoint "$runDir/deep" --data data/academic-pairs-v1 --split validation --limit 0
```

句对评估默认将 `predictions.csv`、`predictions.jsonl` 和 `metrics.json` 写入检查点的 `evaluation/validation-192/`。`--limit 0` 表示使用整个数据分割，改成 `--split test` 可跑测试集。训练权重、预测和临时产物保存在本地 `runs/`、`output/` 或 `.cache/` 中。

需要核验项目时，可在根目录手动执行以下命令。训练、推理、测试和其他项目脚本由用户执行，除非用户明确授权代跑。

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/verify_integrity.ps1
```

## 当前进度与维护

三级入口、独立词级模型、句对训练和批量评估流程均已写入代码。现有真实分割可用于诊断不同模式的问题。新训练指标以实际生成的检查点和评估文件为准。独立人工盲评、三种模式的质量结论与 MORL 对照仍待验证。

以后每次完成一项项目修改，都要在同一提交里同步这份总览。更新受影响的章节，并在下面记录改动和实际验证情况。规则见[仓库维护说明](../AGENTS.md)。

| 日期 | 改动 | 验证与状态 |
| --- | --- | --- |
| 2026-11-19 | 建立中文项目总览，增加 README 入口和每次修改同步更新的仓库规则。 | 核对现有代码、数据清单、操作命令和链接。 |
| 2026-11-19 | 增加浅、中、深三种模式和独立词级模块，记录 Doolittle 评价与 Loss 方案。 | 实现已静态检查。训练、推理及外部指标的实测待完成。 |
| 2026-11-17 | 加入六层模型与 `academic-pairs-v1` 训练语料。 | 数据版本、分割与生成器指纹记录在语料清单中。 |
