# 数据统计与运行文件字段说明

本文件解释机器可读文件的英文键。JSON 键、命令和英文数据保持原样，便于直接对照程序。

## 下载清单

文件为 data/raw/manifest.json，版本管理副本为 docs/data_manifest.json。

| 字段 | 中文含义 |
|---|---|
| dataset | 数据集标识 wanyu/IteraTeR_human_sent |
| revision | 固定的上游提交版本 |
| files | 每个下载文件的信息 |
| sha256 | 文件内容 SHA256 校验值 |
| bytes | 字节数 |

## 清洗与划分统计

文件为 data/processed/stats.json，副本为 docs/dataset_statistics.json。

| 字段 | 中文含义 |
|---|---|
| seed | 划分随机种子，本次为 42 |
| raw_counts / raw_total | 原始各集合记录数 / 总数 |
| kept_total | 保留文本对数，本次 676 |
| rejections | 按首次触发原因统计的丢弃数量 |
| components | 文档连接组数，本次 95 |
| split_policy | 按文档连接组比例划分的规则 |
| splits | 各新集合的对数、文档数、意图分布 |
| pairs / documents / intents | 文本对数 / 文档数 / 意图计数 |
| leakage | 跨集合文档与完全相同文本重叠检查 |
| shared_docs / shared_exact_texts | 共享文档 / 完全相同文本数量 |
| upstream_shared_documents | 原始下载划分之间的共享文档数 |
| limitation | 检查不能证明的事项，如语义近似重复和标签可靠性 |

## 处理后的样本

data/processed/train.jsonl、validation.jsonl、test.jsonl 每行是一条 JSON。

| 字段 | 中文含义 |
|---|---|
| id | 由规范化文本对生成的样本标识 |
| source / target | 修改前英文 / 上游人工修订后的英文 |
| intent | clarity 清晰度、fluency 流畅度、style 风格 |
| doc_id | 原文档编号，同篇文档的修订必须同集合 |
| upstream_split / upstream_line | 原始集合与文件行号，供追溯 |
| group_id | 防止文档或相同文本跨集合的连接组标识 |

examples.json 存放展示用训练样本，target 是人工参考，不是模型输出。

## 训练结果与日志

两个实验目录为 runs/transformer_smoke 和 runs/hybrid_smoke。

| 文件 | 中文说明 |
|---|---|
| result.json | 配置、规模、损失、设备和限制 |
| training_log.json | 各步骤训练批次的损失，批次可能不同 |
| validation_examples.json | 验证样本及生成结果，当前保存输出为空 |
| weights.pt | 供重新加载的权重，尚非可用改写模型 |
| tokenizer/ | 仅在训练文本学习的分词器 |
| backbone_config/ | 小型 BART 主干配置 |

result.json 主要字段如下。

| 字段 | 中文含义 |
|---|---|
| model / initialization | 模型类型 / 初始化方式，本次为随机小型 BART |
| backend / fusion | Mamba 后端 / 融合方式，Transformer 对应值为空 |
| steps / train_pairs / validation_pairs | 步数 / 实际运行训练对数 / 验证对数 |
| max_length / batch_size / learning_rate | token 长度上限 / 批量 / 学习率 |
| fixed_train_batch_loss_before / after | 同一固定训练批次在优化前后的交叉熵 |
| validation_cross_entropy | 验证子集按有效目标 token 加权的交叉熵 |
| seconds_training | 训练秒数，不含准备和评价 |
| peak_cuda_allocated_mib | 进程 CUDA 峰值已分配显存，单位 MiB |
| parameters / trainable_parameters | 总参数量 / 可训练参数量 |
| train_truncated_sources / targets | 因长度限制截断的源句 / 目标句数 |
| device / gpu / torch | 设备 / GPU 名称 / PyTorch 版本 |
| experiment_type | 实验用途，本次为流程验证 |
| test_set_used | 是否用测试集进行模型运行，本次 false |
| args | 完整命令行参数 |

runs/tests.txt 为四项测试结果，runs/demo_reload.txt 为检查点重载演示，runs/environment.json 为环境版本。configs/experiments.json 区分已运行的 smoke 与未运行的 planned_main 配置；不能把后续配置当成已完成实验。
