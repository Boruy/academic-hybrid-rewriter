# 第三方数据和依赖说明

IteraTeR-HUMAN 句子级数据来自 Wanyu Du、Vipul Raheja、Dhruv Kumar、Zae Myung Kim、Melissa Lopez 和 Dongyeop Kang 的论文 Understanding Iterative Revision from Human-Written Text，发表于 ACL 2022。

- 数据集：https://huggingface.co/datasets/wanyu/IteraTeR_human_sent
- 作者仓库：https://github.com/vipulraheja/iterater
- 固定版本：e22e0371dac444239b944f9293f5b491d62b73f0

数据集页面标注 Apache-2.0。上游英文许可证保存在 data/IteraTeR_LICENSE.txt，另附 data/IteraTeR_LICENSE_zh_reference.txt 作为非官方中文参考译文。

处理数据的修改包括规范化、按 arXiv 与意图筛选、长度和数字过滤、去重及按文档连接组重新划分。来源、统计与校验值见 docs/Dataset_Card_zh.md 和 docs/data_manifest.json。使用文本时应保留原始来源与署名。

交付包中的小型 BART 格式检查点为本地流程验证产物，不是预训练权重。PyTorch、Transformers、Tokenizers 是另行安装的依赖，适用各自上游许可证。src 中 Mamba 参考适配模块为本阶段新编写实现；可选官方 mamba_ssm 未随包提供或完成验证。

ZIP 用于本地交接，目前没有公开发布数据或推送远程 GitHub。原始数据、处理数据、权重与交接 ZIP 继续由 .gitignore 排除。
