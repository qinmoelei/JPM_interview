github Repo for JPM interview

# Q1P1   Vélez–Pareja-Style Financial Forecast (CB → IS → BS) with TensorFlow

Deterministic, no-plug, no-circularity three-statement forecaster inspired by Vélez‑Pareja.  
Pipeline: Yahoo Finance → preprocessing to states/drivers → deterministic simulator → driver baselines (perfect / sliding / AR1 / MLP).

**Order of computation**: exogenous drivers → **Cash Budget (CB)** → **Income Statement (IS)** → **Balance Sheet (BS)** while enforcing `Assets = Liabilities + Equity`.

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

## Driver 实验摘要（中文）
- 使用稳定 Top10 公司（AAPL, AMAT, HCA, WMT, LMT, DPZ, SNA, ECL, PNW, AEE）。
- 年度：only sliding_mean 有效，driver MSE test≈6.9；state MSE test≈1.9e19；AR1/MLP 因样本不足跳过。
- 季度：sliding_mean driver MSE test≈4.66e2（state≈2.33e19）；AR1 driver MSE test≈5.78e2；MLP driver MSE test≈4.78e3，state 依然较大。per_ticker 指标写在各实验目录 `per_ticker.json`。

### References 参考文献

- Mejia‑Pelaez, F., & I. Vélez‑Pareja (2011), *Analytical Solution to the Circularity Problem in the Discounted Cash Flow Valuation Framework*, Innovar 21(42):55‑68. 该文提出“现金预算先行”的反推思路。
- Vélez‑Pareja, I. (2007), *Forecasting Financial Statements with No Plugs and No Circularity* (SSRN 1031735). 指导数据缺失填补与现金、债务的因果次序。
- Vélez‑Pareja, I. (2009), *Constructing Consistent Financial Planning Models for Valuation* (SSRN 1455304). 说明了如何在估值模型中处理税率、融资与股东回报，本仓库的预处理/正则化策略据此设定。


