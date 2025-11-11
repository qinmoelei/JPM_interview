github Repo for JPM interview

# Q1P1   Vélez–Pareja-Style Financial Forecast (CB → IS → BS) with TensorFlow

This repository implements a **deterministic, no-plug, no-circularity** three-statement forecaster in the spirit of Vélez‑Pareja.
It uses **Yahoo Finance** (via `yfinance`) to obtain **income statements**, **balance sheets**, and **cash flow statements**,
and trains **driver networks** in TensorFlow to forecast exogenous drivers. The **accounting layer** is implemented as a
custom Keras layer that enforces the **cash-budget rules** and **accounting identities** by construction.

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
  00_preprocess.py          # EN: clean/normalize raw statements | ZH: 数据预处理
  01_download.py            # EN/ZH: 下载原始报表
  02_train.py               # EN/ZH: 训练入口
  03_eval.py                # EN/ZH: 回测入口（示例）
  run_baselines.py          # EN: ARIMA/GARCH baselines + 可视化 | ZH: 统计基准与画图
```

## 工作流 / Workflow

1. **Download（下载）** – `python script/01_download.py --config configs/config.yaml`，批量获取 Yahoo 年/季报。
2. **Preprocess（预处理）** – `python script/00_preprocess.py --config configs/config.yaml --variant base`。各变体会写入 `data/processed/<variant>/` 并生成 `training_data.npz`、`training_summary.json`，可并行保存多套处理方案（如 `--variant bounded`）。
3. **Train（训练）** – `python script/02_train.py --config configs/config.yaml --variant base [--experiment-tag demo]`。脚本会：
   - 载入 `data/processed/<variant>/training_data.npz`
   - 使用轻量级 SimpleRNN driver net（16 hidden）
   - 将 loss/metrics 分别写入 `results/run_<timestamp>/training_logs.json`
   - 在同一目录保存 `learner.json`（配置+模型+设备）与 `model.weights.h5`。
4. **Compare（统计基准）** – 训练脚本会自动调用 `script/run_baselines.py`，在同一个 run 目录生成 `baseline_metrics.json` 与 `arima_samples.png`（2×2 子图展示 ARIMA 拟合 vs 真实轨迹）。你也可以手动运行：
   ```bash
   python script/run_baselines.py --config configs/config.yaml --variant base --run-dir results/run_YYYYMMDD_HHMMSS
   ```
5. **Evaluate（评估）** – `python script/03_eval.py --config configs/config.yaml` 仍可作为自定义回测入口。

> **环境提示**：推荐使用 `conda run -n jpmc <command>` 与本地测试一致。

## 快速开始 / Quick Start

> **Note:** The environment here is offline; run the commands below *locally*.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -U -r requirements.txt
python script/01_download.py --config configs/config.yaml
python script/00_preprocess.py --config configs/config.yaml
python script/02_train.py --config configs/config.yaml
python script/03_eval.py  --config configs/config.yaml
```

## References

### 关键函数 / Key Functions

- `data_download.download_universe` – EN: stream-downloads Yahoo statements with logging & retries. ZH: 批量下载 Yahoo 报表并记录失败信息。
- `preprocess.run_preprocessing_pipeline` – EN: imputes NaNs (zero/ffill/interp/constraints), enforces `Assets = Liabilities + Equity`, normalizes firms by median sales/asset scale, and exports Vélez–Pareja style state/driver tensors. ZH: 按现金预算→IS→BS 顺序处理缺失、约束并归一化，输出训练所需的状态与驱动序列。
- `data/processed/<variant>/training_summary.json` – 新增分层结构，记录 seq_len、样本数、协变量列名；可同时保存多个预处理版本，方便 A/B 实验。
- `results/run_*/` – 每次训练自动生成独立目录，内含 `learner.json`、`training_logs.json`、`model.weights.h5`、`baseline_metrics.json`、`arima_samples.png`，便于复现实验与汇报。
- `CashBudgetLayer` – EN: deterministic CB layer used during training to keep causality `CB → IS → BS`. ZH: 现金预算层确保先决现金策略再推导 IS/BS，避免循环与“塞数”。
- `get_default_config_path` – EN: resolves `$JPM_CONFIG_PATH` or repo default for every CLI. ZH: 自动定位配置文件，方便脚本无参运行。

### References 参考文献

- Mejia‑Pelaez, F., & I. Vélez‑Pareja (2011), *Analytical Solution to the Circularity Problem in the Discounted Cash Flow Valuation Framework*, Innovar 21(42):55‑68. 该文提出“现金预算先行”的反推思路。
- Vélez‑Pareja, I. (2007), *Forecasting Financial Statements with No Plugs and No Circularity* (SSRN 1031735). 指导数据缺失填补与现金、债务的因果次序。
- Vélez‑Pareja, I. (2009), *Constructing Consistent Financial Planning Models for Valuation* (SSRN 1455304). 说明了如何在估值模型中处理税率、融资与股东回报，本仓库的预处理/正则化策略据此设定。
