# 学术文本风格增强项目

这是基于 Transformer + Mamba 的课程项目里程碑原型。材料整理于 2026 年 10 月 24 日，用于第 9 周汇报；根据提供的课程要求，该周为 10 月 25 日至 31 日。

## 中文材料入口

- [中文材料总目录](docs/Chinese_Materials_Index_zh.md)
- [七页中文汇报 PPT](output/Milestone_Presentation_zh.pptx)和 [PDF 备份](output/Milestone_Presentation_zh.pdf)
- [一页中文报告 PDF](output/Milestone_Report_zh.pdf)、[可编辑 Word](output/Milestone_Report_zh.docx)和 [Markdown 原稿](docs/Milestone_Report_zh.md)
- [中文讲稿与常见问答](docs/Presentation_Guide_zh.md)
- [实际进展与运行证据](docs/Progress_Evidence_zh.md)
- [数据集来源、筛选与局限](docs/Dataset_Card_zh.md)
- [模型架构与实验方案](docs/Architecture_and_Protocol_zh.md)
- [数据和运行文件字段说明](docs/Data_and_Run_Files_zh.md)
- [提交检查清单](docs/Submission_Checklist_zh.md)
- [第三方数据和依赖说明](THIRD_PARTY_NOTICES_zh.md)
- [许可证中文参考译文](data/IteraTeR_LICENSE_zh_reference.txt)

## 安装报错与本机已修复环境

如果 pip 下载超时，或重复创建 .venv 后找不到 torch，请查看[环境故障排查](docs/Environment_Troubleshooting_zh.md)。本机项目 .venv 已基于 D:\conda\envs\tee 重新创建，共享其中已安装的依赖，项目包以 editable 方式安装在 .venv。可直接使用 .\.venv\Scripts\python.exe -m ... 指定解释器。该方式依赖现有 tee 环境，不是独立复制所有依赖。

## 当前已经完成什么

已下载作者发布的 IteraTeR-HUMAN 句子级修订数据，固定版本并记录 SHA256 校验值。清洗后得到来自 95 篇文档的 676 对 arXiv 修订文本，训练、验证、测试分别为 491、98、87 对。不同集合之间的文档和完全相同文本重叠均为 0。

已实现仅使用训练集学习的分词器、动态填充、目标填充掩码、小型 BART 训练、Mamba 参考适配模块、三种特征融合方式、文本生成和本地检查点重新加载。两个小模型均在 64 对真实训练样本上完成 30 步 GPU 优化，使用 16 对验证样本；保留的测试集尚未用于模型评估。

**当前成果属于开发里程碑。** 小模型从随机权重开始训练，尚不能生成可用的学术改写。Mamba 采用普通 PyTorch 顺序扫描参考实现，尚未验证与官方内核的数值一致性。当前结果不能证明学术语气、语义保留或运行速度有所提升。检测器仍是启发式规则，尚未训练为分类器；Mamba-only 模型和人工风格评价属于后续工作。

## 本机直接运行

当前环境已准备好，不需要再创建 .venv。直接执行：

~~~powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -m academic_rewriter.run --tiny --model hybrid --output runs/hybrid_local --steps 30
~~~

如需离线重新注册本项目，使用下面的命令。脚本会复用已有环境，不会重置共享依赖选项。

~~~powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\setup.ps1 -Offline
~~~

## 首次安装与通用复现

以下命令在完整项目根目录执行。中文材料包是说明与汇报补充，未重复打包训练代码、完整语料和权重，可解压合并到先前的完整项目目录。

要求 Python 3.10 或以上版本。本次记录使用 Python 3.11、PyTorch 2.2.0+cu121、Transformers 4.44.2，以及 NVIDIA RTX 4060 Laptop GPU。程序也支持 CPU。

~~~powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\setup.ps1
if ($LASTEXITCODE -ne 0) { throw "安装失败，请先修复环境再继续" }
.\.venv\Scripts\python.exe scripts/download_data.py
.\.venv\Scripts\python.exe -m academic_rewriter.prepare
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -m academic_rewriter.run --tiny --model transformer --output runs/transformer_smoke --steps 30
.\.venv\Scripts\python.exe -m academic_rewriter.run --tiny --model hybrid --backend reference --fusion gated --output runs/hybrid_smoke --steps 30
.\.venv\Scripts\python.exe -m academic_rewriter.demo --checkpoint runs/hybrid_smoke --text "We looked at a lot of samples." --force
~~~

本机已有环境可以在不安装本项目包的情况下运行：

~~~powershell
$env:PYTHONPATH = "$PWD\src"
& D:\conda\envs\tee\python.exe -m academic_rewriter.prepare
& D:\conda\envs\tee\python.exe -m academic_rewriter.run --tiny --model hybrid --output runs/hybrid_smoke
~~~

演示程序会提示检查点仅用于流程验证。生成文本应由作者审阅后决定是否采用，数字检查本身不能验证完整语义。当前保存的验证生成样例为空，不适合真实写作。

## 下一阶段的预训练 BART 实验

~~~powershell
.\.venv\Scripts\python.exe -m academic_rewriter.run --model transformer --pretrained facebook/bart-base --output runs/bart_baseline --steps 300 --train-limit 0 --val-limit 0 --lr 0.00005 --batch-size 2
.\.venv\Scripts\python.exe -m academic_rewriter.run --model hybrid --pretrained facebook/bart-base --backend reference --freeze-backbone --output runs/bart_adapter --steps 300 --train-limit 0 --val-limit 0 --lr 0.0001 --batch-size 2
~~~

这些命令是后续实验起点，相关实验尚未完成。当前训练器只运行固定步数，没有实现早停。正式实验前需要补充按轮次验证、最佳检查点选择、学习率调度、混合精度和梯度累积，并统一长度限制与解码设置。尚未对本机在 128 token 下的完整预训练模型微调进行资源分析。

官方 Mamba 是显式选择的可选后端。建议在独立的 Linux 或 WSL2 环境中按[官方仓库](https://github.com/state-spaces/mamba)安装说明配置，验证前向、反向传播与检查点行为后，再设置 --backend official。程序不会自动替换后端。

## 完整仓库和 Git LFS

本仓库包含全部源码、测试、安装脚本、原始数据、处理后数据、五组模型权重及对应分词器和配置、训练日志、中英文文档、原始提案与幻灯片、最终汇报材料，以及现有 ZIP 材料包。仅排除本机虚拟环境、缓存、字节码和自动生成的包元数据；依赖按上面的步骤安装。现有 ZIP 为此前打包快照，当前完整内容以仓库文件为准。

模型权重通过 Git LFS 保存，两个 BART 权重超过普通 GitHub 文件大小限制。请先安装 Git LFS，再克隆仓库并下载实际权重：

~~~powershell
git lfs install
git clone https://github.com/Boruy/academic-hybrid-rewriter.git
cd academic-hybrid-rewriter
git lfs pull
git lfs fsck
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/verify_integrity.ps1
~~~

获取完整项目请使用 Git 克隆并拉取 LFS。GitHub 网页的源码 ZIP 可能仅含权重指针。详见[完整内容与上传说明](docs/GitHub_Publishing_zh.md)；[SHA256SUMS.txt](SHA256SUMS.txt)提供项目文件校验值。

## 来源与注意事项

IteraTeR 来源：[作者仓库](https://github.com/vipulraheja/iterater)、[数据集页面](https://huggingface.co/datasets/wanyu/IteraTeR_human_sent)。数据集说明标注 Apache-2.0。应保留来源与署名，重新分发前核对原始文本适用条款。本项目源码为课程新编写，尚未代团队选择其发布许可证。

文件夹中没有课程大纲原文，提交限制、占比与日期沿用对话中提供的要求。姓名沿用 Proposal.pdf 的 Licheng Wang；MLPre1.pptx 使用 Lichen Wang，提交前应核对。中文材料中的模型名、命令、路径、JSON 键和英文样本保留原始写法，以便与程序对应。
