# Part2 + Bonus 实验记录

## 环境与说明
- LLM：APIYi（base_url `https://api.apiyi.com/v1`，model `gpt-4.1-mini`，SDK `openai==2.15.0`）
- 数据：`data/processed/year`（Part1 处理后的 states/drivers），`data/pdf/*`（年报 PDF）
- 评估指标：  
  - `MSE = mean((y_pred - y_true)^2)`  
  - `MAE = mean(|y_pred - y_true|)`  
  - `RelL1 = mean(|y_pred - y_true| / (|y_true|+1e-8))`  
  - `RelL2 = mean((y_pred - y_true)^2 / (|y_true|+1e-8)^2)`

## P2-1：LLM 预测 balance sheet（driver-level）并与 Part1 模型对比
实现方式：LLM 预测 13 维 driver，保持会计恒等式，再由 Part1 simulator 前推 state。  
实验 ticker：`AAPL, SNA, HCA, PNW, ECL, DPZ`（过滤极端 driver 数值）。

结果（见 `results/part2_llm_clean/llm_metrics.json` 与 `results/part2_llm_clean/mlp_baseline.json`）：
- LLM (driver-level)  
  - driver_test: MSE 1516.03, MAE 10.30, RelL1 1.12  
  - state_test: MSE 3.09e19, MAE 1.50e9, RelL1 0.59
- Part1 baseline (MLP driver)  
  - driver_test: MSE 8689.99, MAE 36.02, RelL1 6082.63  
  - state_test: MSE 6.07e19, MAE 2.81e9, RelL1 5.35e14  
结论：LLM driver 预测在该样本上显著优于 Part1 MLP baseline，state 误差也更低。

## P2-2：Ensemble（Part1 + LLM）
形式：driver-level 线性融合  
`d_hat = w * d_LLM + (1-w) * d_Part1`，w 在验证集网格搜索。

结果（见 `results/part2_llm_clean/ensemble_metrics.json`）：  
最优权重 `w=1.0`（验证集 MSE 最小），即 LLM 单模在此样本上最优。

## P2-3：CFO action recommendation
示例公司：AAPL（LLM driver 预测后生成建议）  
输出：`results/part2_llm_clean/cfo_recommendation.json`

摘要（节选）：
- 核心结论：现金与权益改善，但 DPO/DSO 压力上升，需关注流动性。
- 动作建议：  
  1) 延长应付账款周期（提升 DPO）  
  2) 加速应收回款（降低 DSO）  
  3) 控制 capex  
  4) 适度调整分红策略  
  5) 通过定价对冲 COGS 上升

## P2-4：Robustness（多次运行）
实验：3 次运行，temperature=0.1（见 `results/part2_llm_clean/robustness_summary.json`）  
- driver_MAE mean/std: 10.28 / 0.01  
- driver_RelL1 mean/std: 1.116 / 0.067  
- state_MAE mean/std: 1.52e9 / 1.77e7  
- model/version：`gpt-4.1-mini`, SDK `openai==2.15.0`

## P2-5：GM 年报 PDF 抽取 + 比率计算
流程：pypdf 提取页文本 → LLM 结构化抽取 → 本地算比率。  
输出：`results/part2_pdf/GM_2023/`

抽取结果（单位与原报表一致，通常为 million USD）：
- net income: 9,840  
- revenue: 171,842  
- total assets: 273,064  
- total debt: 212,741  
- total equity: 68,189  
- current assets: 101,618  
- current liabilities: 94,445  
- inventory: 16,461  

比率公式：
- cost-to-income = (COGS + SG&A) / Revenue  
- quick ratio = (Current Assets - Inventory) / Current Liabilities  
- debt-to-equity = Total Debt / Total Equity  
- debt-to-assets = Total Debt / Total Assets  
- debt-to-capital = Total Debt / (Total Debt + Total Equity)  
- debt-to-EBITDA = Total Debt / EBITDA  
- interest coverage = EBIT / Interest Expense

计算结果（见 `results/part2_pdf/GM_2023/ratios.json`）：
- cost-to-income: 1.003  
- quick ratio: 0.902  
- debt-to-equity: 3.120  
- debt-to-assets: 0.779  
- debt-to-capital: 0.757  
- debt-to-EBITDA: 13.07  
- interest coverage: 10.21

## P2-6：Generalization（LVMH 年报）
同一工具直接跑通：`results/part2_pdf/LVMH_2024/`  
抽取到净利润、资产、负债等核心项；因 PDF 文本结构差异，Revenue/COGS/SG&A 未抽取成功，故 cost-to-income 留空。  
其余比率可见 `results/part2_pdf/LVMH_2024/ratios.json`。

## Bonus B1：信用评级模型 + Evergrande
数据来源：`data/processed/year` 里的 Yahoo 财报（已在 repo）。  
构造特征（WC/TA、EBIT/TA、Debt/TA、Equity/TA、Sales/TA、Interest coverage），用 Altman-style score 生成评级桶，再训练分类器（Logistic Regression + 线性回归序数化）。

训练集保存：`data/credit_rating/rating_training_dataset.csv`  
指标：`results/part2_bonus/credit_rating_metrics.json`  
- multiclass accuracy: 0.571  
- ordinal accuracy: 0.128 (线性序数回归)

Evergrande 2022：  
抽取结果见 `results/part2_bonus/evergrande_rating.json`，评分 → 评级 `CCC`（高风险）。

### Shenanigans 检测（规则）
规则：AR/Sales 异常上升、库存异常、CFO 与 NI 长期背离。  
样本：`GM, AIG, F, GE, CCL, DAL`（历史压力样本）  
输出：`results/part2_bonus/shenanigans_flags.json`

## Bonus B2：Risk warnings & qualified opinions
规则法：关键词/正则识别审计意见类型。  
样本：HKICPA 示例审计报告 + Evergrande 2022。  
输出：
- `results/part2_bonus/audit_hkicpa_example.json`
- `results/part2_bonus/audit_evergrande_2022.json`

同时用 TF-IDF 排序风险段落（Top-k）：  
- `results/part2_bonus/topk_paragraphs.json`  
- `results/part2_bonus/human_review_sheet.csv`

## Bonus B3：Loan pricing + Resale + 95% CI
数据来源：LendingClub `LoanStats3a.csv.zip`（自动下载） + FRED Treasury yields。  
输出目录：`results/part2_bonus/loan_pricing/`

定价模型：
- Linear Ridge（baseline 可解释）
- HistGradientBoostingRegressor（非线性）

结果（见 `results/part2_bonus/loan_pricing_summary.json`）：
- spread RMSE/MAE：Linear 0.00515 / 0.00406；GBDT 0.00049 / 0.00032  
- 私有借款人场景（无市场特征）MAE 0.00273  
- 1 个月 resale return 预测：RMSE 6.78e-4，MAE 5.95e-4  
- 95% 置信区间（conformal）：coverage 0.95，avg width 0.00227

---
需要时我可以把这些实验整理成独立脚本运行入口（目前均可直接调用 `src/llm/*` 中的模块）。
