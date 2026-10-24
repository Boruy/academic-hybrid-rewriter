# 完整项目上传与下载说明

GitHub 仓库：https://github.com/Boruy/academic-hybrid-rewriter 。初始设为私有，仅授权用户可访问。

## 内容范围

- 全部源码、测试、配置、安装和下载脚本。
- IteraTeR 作者发布的原始数据、处理后 train/validation/test 数据、示例与统计，保留来源、版本、校验值和许可证。
- runs 中 bart_adapter、bart_baseline、hybrid_local、hybrid_smoke、transformer_smoke 五组完整检查点，包括 weights.pt、分词器和模型配置，以及全部现有日志。
- 全部中英文项目说明、报告原稿、讲稿、数据卡、运行证据与排错文档。
- Proposal.pdf、MLPre1.pptx、output 中的报告、幻灯片、PDF 和现有 ZIP 包。ZIP 是此前快照，并非当前仓库的替代品。
- VS Code 设置和依赖声明。

本机 .venv、旧虚拟环境、.cache、.build、__pycache__ 和自动生成的 egg-info 属于本机环境或可生成文件，不作为项目材料上传。当前 .venv 还依赖本机 D 盘环境，复制它不能替代在目标机器安装依赖。

## 大文件与完整下载

五组 weights.pt 均由 Git LFS 管理。两个 BART 权重分别约 578 MB 和 558 MB；其他文件保留在普通 Git 中。GitHub 对普通文件设有大小限制，LFS 会把实际权重保存在独立对象存储中，仓库内则记录指针。不能把指针误当作权重已经下载。

安装 Git LFS 后执行 git lfs install，克隆项目并进入目录，再执行 git lfs pull 和 git lfs fsck。下载后运行 scripts/verify_integrity.ps1，将实际文件与 SHA256SUMS.txt 核对。校验清单不包括其自身、.git 元数据及上述运行环境文件。仓库通过 .gitattributes 保留文件原始字节，避免换行转换影响校验。

网页 Download ZIP 是否包含 LFS 文件取决于仓库设置，因此完整交接优先使用 Git 克隆。上传使用账号现有配额；未购买额外存储或修改付费设置。

## 来源与项目局限

数据和依赖来源见 THIRD_PARTY_NOTICES.md、THIRD_PARTY_NOTICES_zh.md 及数据卡。保留原始许可证；没有代团队为原创代码选择新的开源许可证。现有 README 对实验有效性和模型输出的限制仍需遵守。

官方说明：[GitHub 普通大文件限制](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github)、[Git LFS](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-git-large-file-storage)、[LFS 配额](https://docs.github.com/en/billing/concepts/product-billing/git-lfs)。
