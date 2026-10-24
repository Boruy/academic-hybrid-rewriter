# 学术文本风格增强项目

Transformer 与 Mamba 混合模型里程碑报告

Borui Ning、Licheng Wang、Haoxiang Ning、Lingfeng Lu

第 9 周  2026 年 10 月 25 日至 31 日

我们已建立英文句子级学术修订流程，取得真实数据和初步 GPU 运行证据。本阶段验证实现进展；预训练模型微调与独立质量评价将在下一阶段开展。

## 1 数据集

下载作者发布的 IteraTeR-HUMAN 句子级数据 [1]，固定版本 e22e0371dac444239b944f9293f5b491d62b73f0 并记录校验值。原始 4,018 条记录经规范化、去重、长度和数字一致性过滤后，保留 676 对 arXiv 文本，意图为 clarity、fluency 或 style。按文档连接组和种子 42 划分，训练、验证、测试为 491、98、87 对，分别来自 76、9、10 篇文档；跨集合文档与完全相同文本重叠均为 0。真实人工编辑示例将“in a dialogue system”改为“in dialogue systems”。

## 2 模型

改写器组合 BART Transformer 编码器、Mamba 参考适配模块、门控或残差融合，以及 Transformer 解码器；另实现替换融合，用于跳连消融。目标为忽略填充位置的 token 交叉熵。PyTorch 顺序状态递推遵循 Mamba-1 基本结构 [2]，与官方优化内核的数值一致性尚未验证。检测器目前使用启发式规则，监督检测器与 Mamba-only 基线仍待开发。

## 3 当前进展

已实现清洗、文档划分、仅在训练集学习的分词器、批量加载、训练、生成和检查点重载，四项完整性测试通过。在 RTX 4060 Laptop GPU 上，两个随机初始化小型 BART 模型分别用 64 对真实训练样本运行 30 步，并以 16 对样本验证。固定训练批次交叉熵由 7.778 降至 Transformer 的 6.571、Hybrid 的 6.570；验证交叉熵分别为 6.841、6.840。这些结果仅证明流程连通，不能证明改写质量提升。测试集尚未用于模型评价。

## 4 当前问题

仅 24 对保留样本具有 style 标签，直接学术风格监督不足。数字一致不能保证语义保留。人工审核、预训练微调和官方 Mamba 集成尚未完成。保存的小模型验证输出为空，不能用于实际写作；参考扫描也不能证明效率优势。

## 5 后续计划

里程碑周审核 50 对文本并建立独立检测标签。建议组内目标为：11 月 1–7 日完成预训练 BART 基线，11 月 8–14 日开展 Hybrid 集成与融合消融，11 月 15–21 日进行盲评和效率分析。统一划分与解码设置，按验证结果选择检查点，冻结方案后再测试；评价参考重合度、语义、学术语气和资源消耗。本地源码、README 与日志已整理，远程 GitHub 发布待完成。以上为组内目标，应核对课程最终日程。

## 参考来源

[1] IteraTeR 作者仓库 github.com/vipulraheja/iterater；数据 huggingface.co/datasets/wanyu/IteraTeR_human_sent

[2] Gu 与 Dao，Mamba: Linear-Time Sequence Modeling with Selective State Spaces，arxiv.org/abs/2312.00752
