# Repository File Notes

## Top level
- README.md: 项目说明、目录结构与运行流程。
- requirements.txt: Python 依赖清单（Part1/Part2）。
- function.md: 本文件，记录仓库文件用途。
- .env: APIYi LLM 配置（API key/base_url/model）。
- .gitignore: Git 忽略规则。
- conftest.py: pytest 全局配置。
- apiyi/: 预留目录（当前为空）。
- __pycache__/: Python 字节码缓存。
- .pytest_cache/: pytest 缓存。
- .vscode/settings.json: VSCode 项目设置。

## configs/
- config.yaml: 默认运行配置（ticker 列表、路径、训练参数）。
- config_stable.yaml: 稳定子集配置。
- config_stable_top.yaml: 稳定子集（Top）配置。
- prompt/: LLM prompt 模板目录（按任务拆分）。
- prompt/driver_forecast.yaml: driver 预测 prompt。
- prompt/pdf_extract.yaml: PDF 抽取 prompt。
- prompt/cfo_recommendation.yaml: CFO 建议 prompt。
- prompt/risk_score.yaml: 风险段落打分 prompt。
- prompt/statement_extract.yaml: 报表表格抽取 prompt。

## src/
- __init__.py: Python 包标记。

### src/datahandler/
- __init__.py: 子包标记。
- data_download.py: Yahoo Finance 语句下载与保存。
- data_sanity_check.py: 语句行名标准化与基础一致性检查。
- preprocess.py: 报表清洗、构建 states/drivers。
- features.py: driver 比率计算。
- dataset.py: 序列化样本构建（rolling windows）。

### src/model/
- __init__.py: 子包标记。
- dynamics_tf.py: state/driver 顺序与前向演化（TF/NumPy）。
- simulator.py: 会计恒等式驱动的确定性模拟器。

### src/utils/
- __init__.py: 子包标记。
- io.py: 配置加载、路径工具。
- logging.py: 统一日志工具。

### src/experiments/
- __init__.py: 子包标记。
- driver_workflow.py: driver 预测实验流程、基线与指标评估。

### src/llm/
- __init__.py: LLM 入口导出。
- apiyi.py: APIYi OpenAI 兼容客户端与请求工具。
- json_utils.py: LLM JSON 输出解析与数值清洗。
- driver_forecast.py: LLM driver 预测与实验封装。
- ensemble.py: driver-level ensemble 融合与权重搜索。
- robustness.py: 多次运行稳健性评估。
- recommendations.py: CFO/CEO 建议生成。
- ratios.py: 财务比率公式计算。
- pdf_extract.py: PDF 抽取 + LLM 结构化解析。
- pdf_statement_pipeline.py: pdfplumber 抽取 + LLM 表格解析 + 稳健性评估。
- credit_rating.py: 评级数据集构建、模型训练与 Evergrande 打分。
- shenanigans.py: 规则式“财务舞弊/红旗”检测（基于 raw 数据）。
- risk_warnings.py: 审计意见抽取与风险段落排序。
- loan_pricing.py: 贷款定价/转售价格/区间估计流程。
- prompt_config.py: 读取/渲染 prompt 配置（支持目录）。
- prompt_logger.py: 将 prompt/response 追加到 markdown 日志。
- reasoning_logger.py: LLM 简短理由文本日志。
- driver_pipeline.py: LLM driver 预测主流程（供脚本调用）。

## script/
- __init__.py: 子包标记。
- 00_download.py: 下载原始报表（CLI）。
- 01_preprocess.py: 预处理生成 states/drivers（CLI）。
- 02_train.py: 确定性模拟评估（CLI）。
- 03_eval.py: 额外评估入口（CLI）。
- 04_driver_pipeline.py: driver baseline 实验（CLI）。
- 05_llm_driver_pipeline.py: LLM driver 预测 + ensemble + robustness + CFO 建议（CLI）。
- 06_pdf_statement_pipeline.py: PDF 报表抽取 + 多模型稳健性评估（CLI）。

## tests/
- test_02_data_checks.py: 报表字段/恒等式检查。
- test_03_data_download.py: 下载流程测试。
- test_04_dataset.py: 序列样本构建测试。
- test_05_features.py: driver/ratio 计算测试。
- test_06_driver_metrics.py: driver 指标测试。
- test_07_llm_apiyi.py: APIYi LLM 调用示例。
- test_08_llm_utils.py: LLM JSON/ratio 工具测试。

## data/
- raw/*_IS_annual.csv, *_BS_annual.csv, *_CF_annual.csv: Yahoo 原始报表数据。
- raw/*_IS_quarterly.csv, *_BS_quarterly.csv, *_CF_quarterly.csv: Yahoo 季度报表数据。
- processed/year/*_states.csv: 年度 states (T x 15)。
- processed/year/*_drivers.csv: 年度 drivers (T-1 x 13)。
- processed/year/*_simulation.npz: simulator roll-out 结果。
- pdf/gm_2023_ar.pdf: GM 年报 PDF。
- pdf/lvmh_2024.pdf: LVMH 年报 PDF。

## results/
- driver_experiments_year_top/: Part1 driver baseline 输出。
- driver_experiments_quarter_top/: Part1 季度 baseline 输出。
- driver_experiments_*/**/preds_val.json: Part1 逐 ticker 预测（val，用于 ensemble）。
- driver_experiments_*/**/preds_test.json: Part1 逐 ticker 预测（test，用于 ensemble）。
- part2_llm_run_a2d_quarter_top/: Part2(a–d) LLM 实验（季度）。
- part2_llm_run_a2d_year_top/: Part2(a–d) LLM 实验（年度）。
- part2_llm_run_e2i/: Part2(e–i) PDF 报表抽取 + 多模型稳健性输出。

## reports/
- report.tex: Part1 报告 LaTeX 源码。
- JPMC_DataProject.pdf: Part1 报告 PDF。

## plan_prompts/
- part2_后续计划.md: Part2/Bonus 任务拆解说明。
- EXPERIMENT_PLAN.pdf: 实验计划。
- new_plan.md / new_plan_4_model.md: 其他方案草案。

## questions/
- Intern interview 2026 question 1 ver 2.docx: 题面原文。

## outlook/
- 0_pakage_related/*: 包/工具相关探索。
- 1_data_related/*: 数据探索与统计。
- 2_result_visiualizaiton/*: 结果可视化与图表。
