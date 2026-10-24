# Milestone 交付与课堂检查

## 新增中文版本

本次还提供 output/Milestone_Presentation_zh.pptx 与 PDF 备份，以及一页 output/Milestone_Report_zh.pdf、可编辑 DOCX 和 docs/Milestone_Report_zh.md。其余中文说明统一从 docs/Chinese_Materials_Index_zh.md 进入。课堂汇报和提交时按课程语言要求选择相应版本。中英文内容和实验数值一致。

## 原有英文提交文件与中文辅助材料

- output/Milestone_Presentation.pptx：7 页英文演示，可编辑，含中文演讲备注。
- output/Milestone_Presentation.pdf：演示备份。
- output/Milestone_Report.pdf：一页英文报告，提交版。
- output/Milestone_Report.docx：可编辑报告；以核验后的 PDF 作为分页基准。
- docs/Presentation_Guide_zh.md：中文讲稿、演示步骤与常见问答。
- README.md、src、tests、scripts、docs、runs：本地代码仓库与证据。

## 上课前

- 核对成员姓名。Proposal 写 Licheng Wang，第一版 PPT 写 Lichen Wang；本次沿用 Proposal。
- 核对老师原始大纲和最终报告截止时间。当前文件夹没有大纲，周次、10% 和一页限制来自你提供的要求。
- 根据实际发言时间调整讲稿，当前按约 6–7 分钟准备。
- 确认报告为一页，PPT 为 7 页。
- 本地打开 PPT 浏览字体、换行和备注，带上 PDF 备份。
- 只把 30 步小模型运行称为 smoke test / 流程验证，不能说已完成预训练模型训练或证明 Hybrid 优于 Transformer。
- 样本展示标注 Human reference，不能当作模型生成结果。
- 当前 detector 为启发式规则；官方 Mamba 内核、人工质量评审尚未完成。

## 三分钟代码演示

在仓库根目录打开 PowerShell，使用 README 中现有环境命令。

1. 打开 data/processed/stats.json，说明真实数据规模、筛选与 0 重叠。
2. 打开 src/academic_rewriter/models.py，说明编码、选择性状态更新和融合。
3. 打开 runs/transformer_smoke/result.json 和 runs/hybrid_smoke/result.json，展示真实运行记录。
4. 可运行 unittest。避免在课堂依赖现场下载。
5. 若展示 demo，先说明小模型输出无法用于真实写作，重点展示加载、检测、生成和人工审阅接口。

## GitHub

本地源码已经具备仓库结构。没有远程 GitHub 地址，也未向任何账户发布。创建自己账户下的空仓库后，按 README 添加 origin 并推送。公开发布时遵循 .gitignore，保留数据来源说明，不上传缓存、原始/处理数据或模型权重。需要分享数据时先核对其授权与归属。
