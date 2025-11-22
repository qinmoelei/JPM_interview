github Repo for JPM interview

# Q1P1   Vélez–Pareja-Style Financial Forecast (CB → IS → BS) with TensorFlow

This repository implements a **deterministic, no-plug, no-circularity** three-statement forecaster in the spirit of Vélez‑Pareja.
It uses **Yahoo Finance** (via `yfinance`) to obtain **income statements**, **balance sheets**, and **cash flow statements**,
and trains **driver networks** in TensorFlow to forecast exogenous drivers. The **accounting layer** is implemented as a
custom Keras layer that enforces the **cash-budget rules** and **accounting identities** by construction.

> **V2 一句话 / One-liner**：`Train on 1→2 & 2→3, validate on 3→4, test on 4→5; predict growth in AR(1) style using lag + rolling ratios, then enforce Assets≈Liab+Equity with a small relative penalty.`  
> 把 driver MLP + CashBudgetLayer 看成轻量结构，主监督信号是 growth（level 只是辅助），完全按时间顺序滚动，没有 per-firm z-score 或泄露。

## V2 Highlights

1. **Time splits（时间切）** – 每家公司只有 T=5，也能按 transition 切分：Train 用 1→2、2→3；Valid 用 3→4；Test 用 4→5。实现方式是滑动窗口 + mask，只在有监督的 transition 上计算 loss，历史部分纯当上下文。
2. **Targets（监督信号）** – 模型输出 `g_t = y_t / (y_{t-1}+ε) - 1`，再映射回 level。训练时 `loss = growth + α·level + λ·relative_identity (+ β·earnings)`，默认 α=0.3，λ∈[1,5]，兼顾增长率和绝对水平。
3. **Normalization（归一化）** – 不做 per-firm z-score；所有状态、driver flows 先除以训练期的 median sales/assets（size scaling），再做 global z-score（由 train transitions 估计）给 covariates。
4. **Features（输入）** – `input_t = concat(covs_t, lag y_{t-1}, rolling_mean(y), lag/rolling ratio)`，rolling mean 完全使用过去数据，保持 causality。
5. **State space（状态）** – 显式纳入 `other_assets`、`other_liab_equity`，保持 Assets≈Liab+Equity。恒等式 penalty 改成相对误差的软约束（clip 0.1，λ 小），不再出现常数大头。
6. **Sliding windows** – 新的 `mask_{split}` 数组让同一窗口可以服务 train/val/test 不同 transition，既用到 3-4 期历史，又保持时间上“用过去预测未来”。

> ✅ 这样做：干净的时间切 + growth 监督 + lag/rolling 特征，样本虽少但结构自带 prior。  
> ❌ 不再做 per-company z-score、防止把未来均值/方差泄露到 train 里。

> **Order of computation**: exogenous drivers → **Cash Budget (CB)** (decide ST/LT financing, short-term investment, lock cash target) → **Income Statement (IS)** (interests on beginning balances, taxes) → **Balance Sheet (BS)** (update states and verify `Assets = Liabilities + Equity`).

## 项目结构 / Project Structure

```
src/
  datahandler/
    data_download.py        # EN: Yahoo download of IS/BS/CF | ZH: 下载原始报表
    data_sanity_check.py    # EN: schema & identity checks | ZH: 表头/恒等式校验
    preprocess.py           # EN: Vélez–Pareja style cleaning/normalization | ZH: 现金预算优先的预处理
    features.py             # EN: driver ratios | ZH: 驱动因子/周转率
    dataset.py              # EN: rolling windows | ZH: 构造序列
  model/
    cash_budget_layer.py    # EN: deterministic cash-budget layer | ZH: 无循环现金预算层
    losses.py               # EN: penalties | ZH: 损失项
    metrics.py              # EN: metrics & identities | ZH: 评估指标
    trainer.py              # EN: TF helpers | ZH: 训练循环
  utils/
    io.py                   # EN: config helpers | ZH: 配置读写
    logging.py              # EN: logger | ZH: 日志
tests/
  test_01_cash_budget_layer.py # synthetic tests: identity holds, no circularity
  test_02_data_checks.py       # sanity checks on mocked Yahoo data
  test_03_data_download.py     # download orchestration & defaults
  test_04_dataset.py           # sequence builder windows
  test_05_features.py          # driver/ratio calculations
configs/
  config.yaml               # tickers, date ranges, training params
reports/
  report.tex                # LaTeX report / 报告
script/
  00_download.py            # EN/ZH: 下载原始报表
  01_preprocess.py          # EN: build simulation-ready states/drivers | ZH: 生成仿真输入
  02_train.py               # EN/ZH: 仿真入口（deterministic simulator）
  03_eval.py                # EN/ZH: 回测入口（示例，可按需调整）
  04_driver_pipeline.py     # EN/ZH: driver 回归/基线 (perfect/sliding/AR1/MLP)
```

- `configs/` 集中管理下载/预处理/仿真参数（含频率、路径设置）。
- `src/datahandler/` 现在专注于把 Yahoo IS/BS/CF 对齐、抽取稳健的 line item、计算 driver 比率，直接生成仿真输入。
- `src/model/` 提供 deterministic accounting simulator，可逐期滚动现金预算而无需神经网络。
- `script/` 下的 `00/01/02` 是下载→预处理→仿真的 CLI workflow，`04_driver_pipeline.py` 用于 driver 拟合/基线评估（perfect/sliding/AR1/MLP）。
- `tests/` 提供轻量合成测试保证恒等式、下载器、特征计算等模块最小正确性。

## 工作流 / Workflow

1. **Download（下载）** – `python script/00_download.py --config configs/config.yaml`，批量获取 Yahoo 年/季报。
2. **Preprocess（预处理）** – `python script/01_preprocess.py --config configs/config.yaml --variant simulation`。每个 ticker 会写入 `data/processed/<variant>/<ticker>_states.csv`、`<ticker>_drivers.csv` 与 `<ticker>_simulation.npz`（含状态矩阵 + driver 序列），同时生成 `simulation_summary.json` 可追踪可用年份。
3. **Simulate（仿真）** – `python script/02_train.py --config configs/config.yaml --variant simulation [--max-tickers 10]`。脚本会：
   - 载入每个 ticker 的初始状态 + driver 序列
   - 使用 deterministic accounting simulator 逐期推演
   - 将 MAE / 资产负债平衡误差写入 `results/simulation_<timestamp>/simulation_metrics.json`
4. **Driver 拟合/基线** – `python script/04_driver_pipeline.py --config configs/config.yaml --variant year --results-subdir driver_experiments_year`（或 `--variant quarter`），提供 perfect/sliding/AR1/MLP 多种 driver 预测基线，并写出 `analysis_*.json` 与预测。
5. **Evaluate（评估，可选）** – `python script/03_eval.py --config configs/config.yaml` 可作为自定义回测入口。

> **环境提示**：推荐使用 `conda run -n jpmc <command>` 与本地测试一致。

## 快速开始 / Quick Start

> **Note:** The environment here is offline; run the commands below *locally*.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -U -r requirements.txt
python script/00_download.py --config configs/config.yaml
python script/01_preprocess.py --config configs/config.yaml
python script/02_train.py --config configs/config.yaml
# 示例 driver 基线（year 数据）
python script/04_driver_pipeline.py --config configs/config.yaml --variant year --results-subdir driver_experiments_year
```

## References

### 关键函数 / Key Functions

- `data_download.download_universe` – EN: stream-downloads Yahoo statements with logging & retries. ZH: 批量下载 Yahoo 报表并记录失败信息。
- `preprocess.build_simulation_frames` – EN: loads raw IS/BS/CF, picks robust line items, derives Vélez-Pareja style states + driver ratios without heavy scaling. ZH: 读取三张表，按候选名称聚合出核心状态与 driver，比对年份自动对齐，生成仿真输入。
- `data/processed/<variant>/*_simulation.npz` – stores `states`（T×12）与 `drivers`（(T-1)×11），便于直接喂入 simulator，并包含 `simulation_summary.json` 方便巡查可用窗口。
- `model.simulator.AccountingSimulator` – EN: deterministic layer that enforces working-capital + financing logic; perfect to stress-test scenarios before ML. ZH: 仿真引擎内置现金预算逻辑，确保 `Assets = Liabilities + Equity`，可先跑情景再考虑机器学习。
- `script/04_driver_pipeline.py` – EN/ZH: 基线 driver 预测（perfect/sliding/AR1/MLP），输出 `analysis_*.json` 与预测，用于查看 driver/state 误差。
- `get_default_config_path` – EN: resolves `$JPM_CONFIG_PATH` or repo default for every CLI. ZH: 自动定位配置文件，方便脚本无参运行。

### References 参考文献

- Mejia‑Pelaez, F., & I. Vélez‑Pareja (2011), *Analytical Solution to the Circularity Problem in the Discounted Cash Flow Valuation Framework*, Innovar 21(42):55‑68. 该文提出“现金预算先行”的反推思路。
- Vélez‑Pareja, I. (2007), *Forecasting Financial Statements with No Plugs and No Circularity* (SSRN 1031735). 指导数据缺失填补与现金、债务的因果次序。
- Vélez‑Pareja, I. (2009), *Constructing Consistent Financial Planning Models for Valuation* (SSRN 1455304). 说明了如何在估值模型中处理税率、融资与股东回报，本仓库的预处理/正则化策略据此设定。



