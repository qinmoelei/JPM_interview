## 0. 模型世界的约定

我们只显式建模这几类资产负债：

- 资产：`CASH, AR, INV, PPE`
- 负债：`AP, DEBT`
- 其他所有资产/负债（预付、其他应收、递延项目等）**不单独建模，它们的影响被折叠到“外部权益变动 ΔEQ_ext”里**。

因此在**模型世界**里约定：

\[
ASSETS_t^{model} = CASH_t + AR_t + INV_t + PPE_t
\]

\[
LIAB_t^{model}   = AP_t + DEBT_t
\]

\[
EQ_t^{model}     = ASSETS_t^{model} - LIAB_t^{model}
\]

如果你已经在预处理里把报表里的 `eq` / `re` 换成了这一口径（再配上 clean surplus 滚动 RE），那它们就直接是下面用的 `EQ_t` 和 `RE_t`。

---

## 1. 从原始三表构造 15 维状态向量 \(y_t\)

对每个年份 \(t\)，从三张年报中读取（列名示意）：

- **损益表 IS：**
  - `sales_t`
  - `cogs_t`
  - `opex_t`（销售+管理等）
  - `depreciation_t`
  - `income_tax_expense_t`
  - `interest_expense_t`
  - `dividend_paid_t`（注意取绝对值，表示现金流出）

- **资产负债表 BS：**
  - `cash_and_equivalents_t`
  - `accounts_receivable_t`
  - `inventory_t`
  - `accounts_payable_t`
  - `net_ppe_t`
  - `total_interest_bearing_debt_t`（短 + 长）

- **保留盈余 RE：**
  - 如果已经有「模型口径」RE 列，直接用；
  - 如果没有，可以按 clean surplus 自己滚动一遍（见 1.2）。

### 1.1 模型口径的 EQ

用简化资产负债表重构「模型 EQ」：

\[
ASSETS_t^{model} = CASH_t + AR_t + INV_t + PPE_t
\]

\[
LIAB_t^{model}   = AP_t + DEBT_t
\]

\[
EQ_t^{model}     = ASSETS_t^{model} - LIAB_t^{model}
\]

如果你已经在 CSV 里写好了这个结果，就直接把那一列当作 `EQ_t`。

### 1.2 可选：clean surplus 生成 RE（如果不用报表 RE）

先用 IS 数据算当期净利润：

\[
EBIT_t = S_t - C_t - SG_t - D_t
\]

\[
EBT_t = EBIT_t - INT_t
\]

\[
NI_t  = EBT_t - TAX_t
\]

然后按 clean surplus：

\[
RE_t^{model} = RE_{t-1}^{model} + NI_t - DIV_t
\]

### 1.3 15 维状态向量定义

定义每一年 \(t\) 的 15 维 state：

\[
y_t =
\big(
S_t, C_t, SG_t, D_t,\;
AR_t, INV_t, AP_t,\;
PPE_t,\; CASH_t,\; DEBT_t,\;
EQ_t,\; RE_t,\;
TAX_t,\; INT_t,\; DIV_t
\big)
\]

对应原始列：

- **Flow 型（当期损益 / 现金流）**
  - \(S_t\)：IS.`sales_t`
  - \(C_t\)：IS.`cogs_t`
  - \(SG_t\)：IS.`opex_t`
  - \(D_t\)：IS.`depreciation_t`
  - \(TAX_t\)：IS.`income_tax_expense_t`
  - \(INT_t\)：IS.`interest_expense_t`
  - \(DIV_t\)：CF.`dividend_paid_t`（取正）

- **Tank 型（期末余额）**
  - \(AR_t\)：BS.`accounts_receivable_t`
  - \(INV_t\)：BS.`inventory_t`
  - \(AP_t\)：BS.`accounts_payable_t`
  - \(PPE_t\)：BS.`net_ppe_t`
  - \(CASH_t\)：BS.`cash_and_equivalents_t`
  - \(DEBT_t\)：BS.`total_interest_bearing_debt_t`
  - \(EQ_t\)：上面定义的 \(EQ_t^{model}\)（或你预先算好的）
  - \(RE_t\)：clean surplus 生成的 RE 或报表 RE（只要口径一致）

> 之后所有 forward / inverse / 评估，都以这套 15 维 state 为准，不再用原始 total_equity_book。

---

## 2. 13 维 driver 向量 \(x_t\)

driver 是「政策参数」，用来驱动 state 的演化：

\[
x_t =
\big(
gS_t, gm_t, sga_t, dep_t,\;
dso_t, dio_t, dpo_t,\;
capex_t,\;
\tau_t, r_t, pay_t,\;
ndebt_t,\; nequity_t
\big)
\]

含义如下：

1. **经营结构相关：**

- \(gS_t\)：`g_sales`，销售增速  
- \(gm_t\)：`gross_margin`，毛利率  
- \(sga_t\)：`opex_ratio`，Opex / Sales  
- \(dep_t\)：`dep_rate`，当期折旧率（对期初 PPE）

2. **营运资金 policy：**

- \(dso_t\)：`dso`，应收账款周转天数  
- \(dio_t\)：`dio`，存货周转天数  
- \(dpo_t\)：`dpo`，应付账款周转天数  

3. **Capex policy：**

- \(capex_t\)：`capex_sales_ratio`，Capex / Sales

4. **税 / 利息 / 分红：**

- \(\tau_t\)：`tax_rate`，有效税率  
- \(r_t\)：`int_rate`，债务利率（对期初 DEBT）  
- \(pay_t\)：`payout_ratio`，分红率（对当期 NI）

5. **融资决策（最关键）：**

- \(ndebt_t\)：`net_debt_issuance_ratio`，净举债 / Sales  
  \[
  \Delta DEBT_t = ndebt_t \cdot S_t
  \]
- \(nequity_t\)：`net_equity_issuance_ratio`，净「外部权益流入」/ Sales  
  \[
  \Delta EQ_t^{ext} = nequity_t \cdot S_t
  \]

> 注意：\(\Delta EQ_t^{ext}\) 在模型里不仅包含真实的增发/回购，  
> 还吸收了我们**没显式建模的其他资产/负债流**对权益的综合影响。

---

## 3. Forward：一步模拟 \(y_t = f(y_{t-1}, x_t)\)

`simulate` 本质上就是在时间维度 for 循环，一次调用这个 `forward_step`。

假设上一期 state \(y_{t-1}\) 全部已知：

\[
y_{t-1} =
\big(
S_{t-1}, C_{t-1}, SG_{t-1}, D_{t-1},
AR_{t-1}, INV_{t-1}, AP_{t-1},
PPE_{t-1}, CASH_{t-1}, DEBT_{t-1},
EQ_{t-1}, RE_{t-1},
TAX_{t-1}, INT_{t-1}, DIV_{t-1}
\big)
\]

### 3.1 经营块：Sales / COGS / Opex / Dep

1. 销售：

\[
S_t = S_{t-1} \cdot (1 + gS_t)
\]

2. 成本、费用、折旧：

\[
\begin{aligned}
C_t  &= (1 - gm_t) \cdot S_t \\
SG_t &= sga_t \cdot S_t \\
D_t  &= dep_t \cdot PPE_{t-1}
\end{aligned}
\]

### 3.2 营运资金：AR / INV / AP

根据 DSO / DIO / DPO 直接给 tank：

\[
\begin{aligned}
AR_t  &= \frac{dso_t}{365} \cdot S_t \\
INV_t &= \frac{dio_t}{365} \cdot C_t \\
AP_t  &= \frac{dpo_t}{365} \cdot C_t
\end{aligned}
\]

定义简化营运资本：

\[
WC_t     = AR_t + INV_t - AP_t
\]

\[
WC_{t-1} = AR_{t-1} + INV_{t-1} - AP_{t-1}
\]

\[
\Delta WC_t = WC_t - WC_{t-1}
\]

### 3.3 Capex & PPE

\[
CAPEX_t = capex_t \cdot S_t
\]

\[
PPE_t = PPE_{t-1} + CAPEX_t - D_t
\]

### 3.4 P&L：INT / TAX / NI / DIV

\[
EBIT_t = S_t - C_t - SG_t - D_t
\]

\[
INT_t = r_t \cdot DEBT_{t-1}
\]

\[
EBT_t = EBIT_t - INT_t
\]

\[
TAX_t = \tau_t \cdot \max(EBT_t, 0)
\]

\[
NI_t = EBT_t - TAX_t
\]

\[
DIV_t = pay_t \cdot \max(NI_t, 0)
\]

### 3.5 FCF

\[
FCF_t = NI_t + D_t - \Delta WC_t - CAPEX_t
\]

### 3.6 融资 & Tanks 更新

**债务：**

\[
\Delta DEBT_t = ndebt_t \cdot S_t
\]

\[
DEBT_t = DEBT_{t-1} + \Delta DEBT_t
\]

**保留盈余（clean surplus）：**

\[
RE_t = RE_{t-1} + NI_t - DIV_t
\]

**外部权益净流入：**

\[
\Delta EQ_t^{ext} = nequity_t \cdot S_t
\]

**总权益：**

\[
EQ_t = EQ_{t-1} + \Delta EQ_t^{ext} + NI_t - DIV_t
\]

> 只要初始时满足  
> \(EQ_0 = CASH_0 + AR_0 + INV_0 + PPE_0 - AP_0 - DEBT_0\)，  
> 且每一步 CASH 按下面的现金恒等式更新，  
> 这条等式在所有 \(t\) 都会自动保持，不需要再手动「重平衡」。

**现金 tank（核心恒等式）：**

\[
CASH_t
= CASH_{t-1}
+ FCF_t
+ \Delta DEBT_t
+ \Delta EQ_t^{ext}
- DIV_t
\]

最终得到新一年的 state：

\[
y_t =
\big(
S_t, C_t, SG_t, D_t,
AR_t, INV_t, AP_t,
PPE_t, CASH_t, DEBT_t,
EQ_t, RE_t,
TAX_t, INT_t, DIV_t
\big)
\]

---

## 4. Perfect 反演：从 \((y_{t-1}^{data}, y_t^{data})\) 反推出 \(x_t^\*\)

现在反过来：  

给定真实数据构造好的两期「模型 state」：

- 上一年：\(y_{t-1}^{data}\)  
- 当年：\(y_t^{data}\)

直接从这两期 state 里**反推出一组 driver \(x_t^\*\)**，使得：

\[
f\big(y_{t-1}^{data}, x_t^\*\big) \approx y_t^{data}
\]

理论上在不考虑浮点误差的情况下应该是完全一致。

记 \(\varepsilon\) 是一个很小的数（例如 \(10^{-6}\)），防止除 0。

### 4.1 经营 & 营运资金 8 个 driver

1. 销售增速 \(gS_t\)：

\[
gS_t^\* = \frac{S_t - S_{t-1}}{\max(S_{t-1}, \varepsilon)}
\]

2. 毛利率 \(gm_t\)：

\[
gm_t^\* = 1 - \frac{C_t}{\max(S_t, \varepsilon)}
\]

3. Opex 比例 \(sga_t\)：

\[
sga_t^\* = \frac{SG_t}{\max(S_t,\varepsilon)}
\]

4. 折旧率 \(dep_t\)：

\[
dep_t^\* = \frac{D_t}{\max(PPE_{t-1},\varepsilon)}
\]

5–7. DSO / DIO / DPO：

\[
dso_t^\* = 365 \cdot \frac{AR_t}{\max(S_t,\varepsilon)}
\]

\[
dio_t^\* = 365 \cdot \frac{INV_t}{\max(C_t,\varepsilon)}
\]

\[
dpo_t^\* = 365 \cdot \frac{AP_t}{\max(C_t,\varepsilon)}
\]

8. Capex / Sales 比 \(capex_t\)：

从 PPE tank 方程反解：

\[
CAPEX_t^{data} = PPE_t - PPE_{t-1} + D_t
\]

\[
capex_t^\* = \frac{CAPEX_t^{data}}{\max(S_t,\varepsilon)}
\]

### 4.2 税 / 利息 / 分红 3 个 driver

先用 state 里的数据重算 EBIT / EBT / NI：

\[
EBIT_t^{data} = S_t - C_t - SG_t - D_t
\]

\[
EBT_t^{data}  = EBIT_t^{data} - INT_t
\]

\[
NI_t^{data}   = EBT_t^{data} - TAX_t
\]

9. 利率 \(r_t\)：

\[
r_t^\* = \frac{INT_t}{\max(DEBT_{t-1}, \varepsilon)}
\]

10. 税率 \(\tau_t\)：

\[
\tau_t^\* =
\begin{cases}
\frac{TAX_t}{\max(EBT_t^{data},\varepsilon)}, & EBT_t^{data} > 0 \\
0, & EBT_t^{data} \le 0
\end{cases}
\]

11. 分红率 \(pay_t\)：

\[
pay_t^\* = \frac{DIV_t}{\max(NI_t^{data},\varepsilon)}
\]

### 4.3 融资相关 2 个 driver

12. 净举债 / Sales：`net_debt_issuance_ratio`

\[
\Delta DEBT_t^{data} = DEBT_t - DEBT_{t-1}
\]

\[
ndebt_t^\* = \frac{\Delta DEBT_t^{data}}{\max(S_t,\varepsilon)}
\]

13. 净外部权益流入 / Sales：`net_equity_issuance_ratio`

利用 clean surplus：

\[
EQ_t = EQ_{t-1} + \Delta EQ_t^{ext} + NI_t^{data} - DIV_t
\]

整理得到：

\[
\Delta EQ_t^{ext\,data}
= EQ_t - EQ_{t-1} - (NI_t^{data} - DIV_t)
\]

于是：

\[
nequity_t^\*
= \frac{\Delta EQ_t^{ext\,data}}{\max(S_t,\varepsilon)}
\]

> 由于 EQ 在预处理阶段就是按  
> \(EQ_t = CASH_t + AR_t + INV_t + PPE_t - AP_t - DEBT_t\)  
> 定义出来的，任何「未建模的其他资产/负债变动」都会通过 CASH / DEBT 的变化体现在 EQ 上，  
> 最终全部被 \(\Delta EQ_t^{ext\,data}\) 吃掉，  
> 所以 `nequity_t^\*` 里已经含有「真实增发/回购 + 残差项」，  
> 这样 forward 的现金恒等式就能严格闭合，从而可以做到 perfect ≈ 0。

---

## 5. 和现有流水线的关系（简要）

只要满足：

1. 预处理时 state 里的 `EQ` / `RE` 已按模型口径重算（见第 1 节）；
2. forward 用的是第 3 节的公式，尤其是现金恒等式：
   \[
   CASH_t = CASH_{t-1} + FCF_t + \Delta DEBT_t + \Delta EQ_t^{ext} - DIV_t
   \]
3. perfect 反演用的是第 4 节的 13 个公式，

那么对于历史数据：

- 用反演得到的 \(x_t^\*\) 丢回 forward，理论上可以精确还原 \(y_t^{data}\)（只差浮点误差）；
- 所谓“其他资产/负债导致 perfect 做不到 0”的问题，在这套构造下被折叠进 `nequity_t` 这一个 driver 中，不再破坏闭合。


---

### 4.4 「完美反演」为何成立（直觉解释）

- forward 里，每一个 \(y_t\) 的分量都只依赖：
  - \(y_{t-1}\) 中的 tank（\(PPE_{t-1}, AR_{t-1}, INV_{t-1}, AP_{t-1}, CASH_{t-1}, DEBT_{t-1}, EQ_{t-1}, RE_{t-1}\) 等），以及  
  - 当期 driver \(x_t\)。
- inverse 部分，我们正好把这些 driver 用**等价定义**反了回来 ——  
  每条 driver 要么是「某个比率 / 周转天数」，要么是「某个差分 / Sales」。

只要原始三张报表在会计上自洽（IS / BS / CF 挂得上），则：

- 真实世界生成 \(y_{t-1}^{data} \to y_t^{data}\) 的 hidden policy，  
  都可以在这套参数化里映射到一组 \(x_t^\*\)；
- 用这组 \(x_t^\*\) 回跑 forward 方程，会得到同一组 \(y_t^{data}\)（数值上只差舍入误差）。

换句话说：

> - 旧版：融资部分靠硬编码的 if / max 规则；  
> - 新版：融资行为浓缩进两个 driver：**净举债 & 净增发**，  
>   这两个完全由两期 BS + NI/DIV 反演出来，forward / inverse 都是显式的代数关系。

这样，你后面要做任何时间序列 / 生成模型时：

- 可以对 13 维 driver 序列 \(x_t^\*\) 建模（ARIMA / RNN / Transformer 都行）；  
- 再把生成的 driver 喂给 forward tank 模型，就能得到一条完整的模拟财报路径，而不会被硬编码债务规则所限制。

---

## 5. 完美 driver 的构造 & 我们要建模的对象

### 5.1 perfect driver 的定义

对每个公司、每个年份 \(t\)：

- 从报表构造 \(y_t^{data}\)（当年 state）和 \(y_{t+1}^{data}\)（下一年 state）
- 用上面的反演公式，算出一组「完美」 driver：

\[
x_t^\* = g(y_t^{data}, y_{t+1}^{data}, \text{IS/CF 报表行})
\]

这个 \(x_t^\*\) 被视为「真实世界在 \(t\to t+1\) 间**实际生效**的 driver」。

### 5.2 我们真正要建模的是什么？

重点：**我们不直接黑箱拟合 \(y_{t+1}\)，而是只建模 driver \(x_t\) 的时间序列。**

原因：

- \(x_t^\*\) 反演时需要 \(y_{t+1}\)（真实下一个年份），所以只有数据集阶段才能算出 \(x_t^\*\)。
- 在真正「预测未来」时，我们没有 \(y_{t+1}\) 的真值，只能：
  - 用 \(y_t\)（上一年的状态）
  - 加上我们自己拟合出来的 driver 预测 \(x_t\)
  - 再用演化方程算出预测的 \(y_{t+1}\)：

\[
\hat{x}_t \approx x_t^\*, \qquad
\hat{y}_{t+1} = f(y_t, \hat{x}_t)
\]

因此：

- **时序建模主要在 \(x_t\) 上**；
- \(y\)-空间的演化完全由明确的结构方程 \(f\) 给出，不用黑箱。

---

## 6. 样本结构 & 时间序列长度

每个公司有一个 driver 序列：

\[
x_0^\*, x_1^\*, x_2^\*, \ldots
\]

考虑到可用年数有限，比如每个公司只有 4 个年份，则：

- \(y\)：有 4 年
- driver \(x_t^\*\)：只有 \(3\) 个（依次驱动 \(0\to1\), \(1\to2\), \(2\to3\)）
- 单个公司的 driver 序列长度 **只有 3**，因此：

  - 我们可以拿「前两个点」做训练（2 个映射），
  - 「最后一个点」做测试。

但是公司很多：N 家 × 每家 3 条 driver，就形成一个大样本池，可以用来拟合全局的 driver 过程。

---

## 7. 数据预处理流程（逻辑）

### 7.1 从原始报表到 perfect driver 表

1. **读取原始数据**

   - `A_IS_annual.csv`, `A_BS_annual.csv`, `A_CF_annual.csv`
   - 主键：`firm_id`（或 `ticker`）和 `year`（或 `fiscal_year`）

2. **按主键 merge**

   - 得到一个宽表，包含：
     - IS 中流量行：`sales`, `cogs`, `opex`, `reconciled_depreciation`, `ni`, `interest_expense`, `tax_provision`
     - BS 中存量行：`ar`, `inv_stock`, `ap`, `net_ppe`, `cash`, `total_debt`, `stockholders_equity`, `re`
     - CF 中：`common_stock_dividend_paid`

3. **对每个公司，按年份排序，构造 \(t\) 与 \(t+1\)**

   - 对所有用于反演的列，构造 `_next` 版本：
     - 如 `sales_next = groupby(firm).shift(-1)` 等
   - 只保留那些 `sales_next` 非空的行：每行代表一个 \(t \to t+1\) 的 transition。

4. **构造状态向量 \(y_t, y_{t+1}\)**

   - \(y_t\)：用当前行的 `sales`, `cogs`, …, `re`
   - \(y_{t+1}\)：用 `_next` 版本的这些列

5. **按第 4 节反演公式构造 perfect drivers \(x_t^\*\)**

   - 先算 8 个仅依赖 \(y_t, y_{t+1}\) 的 driver
   - 再用 IS/CF 里 `ni`, `interest_expense`, `tax_provision`, `dividend_paid` 的 `t+1` 值算出 `tax_rate`, `int_rate`, `payout_ratio`

6. **标记 train / test**

   - 以公司为单位 group：
     - 每家公司最后一个 \(t\to t+1\) transition 做 test
     - 其余的做 train
   - 这样每家公司大致是「2 条 train + 1 条 test」（如果总共 3 条 driver）。

7. **写入预处理数据**

   - 存到 `preprocessed/` 下，比如：
     - `perfect_drivers.parquet`
   - 每一行是一个 transition：

     - 主键：`firm_id`, `year_t`
     - 状态：`y_t` 的 12 个分量，`y_{t+1}` 的 12 个分量（加 `_next` 后缀）
     - driver：`x_t^\*` 的 11 个分量
     - 标记：`is_train` / `is_test`

---

## 8. driver 的时间序列建模

在有了 perfect driver 表之后，我们要在 **driver 维度的时间序列** 上做模型拟合。

整体逻辑：

1. 对每个公司构造 \(x_t^\*\) 序列；
2. 用「上一期的 driver」去预测「当前期的 driver」：
   - 输入：\(x_{t-1}^\*\)（或更长历史窗口）
   - 输出：\(x_t^\*\)

### 8.1 supervised 数据构造

对每个公司：

- 有 driver 序列：\(x_0^\*, x_1^\*, x_2^\*, \dots, x_{T-1}^\*\)
- 构造 supervised 样本：
  
  - 当 history_lag = 1 时：
    
    - 输入：\(x_{t-1}^\*\)
    - 输出：\(x_t^\*\)
  
  - 对所有公司所有满足条件的 (t-1, t) 配对做 pooling，拼成统一的训练集合。

训练/测试拆分：

- target 对应的 base-year \(t\) 如果是 train 行，则这个 supervised 样本属于 train；
- 每家公司最后一个 \(t\to t+1\) transition 被标记为 test 对应的样本。

---

## 9. 三种 driver 模型

我们计划在 \(x_t\) 上尝试三种非常简单的模型：

### 9.1 自动 sliding window mean（单个公司）

思想：

- 对单个公司，时间 \(t\) 的预测 driver 简单取「过去 \(k\) 期的均值」。

形式：

\[
\hat{x}_t = \frac{1}{k} \sum_{i=1}^{k} x_{t-i}^\*
\]

实现上的细节：

- \(k\) 可取 2（或者其他）
- 如果历史长度不足 \(k\)，就取能拿到的全部历史（如只有 1 期，就用那 1 期）
- 因为我们只用最后一条做 test，对 test 点来说总有至少 1 条历史可用。

特点：

- 完全 per-firm，不 pooling；
- 相当于一个非常简单的 smoothing baseline。

---

### 9.2 类 ARIMA 拟合：AR(1)（所有公司一起）

为了简单起见，采用最基础的 AR(1) 模型，而不是完整 ARIMA：

对每个 driver 维度 \(j\)，拟合：

\[
x_{t}^{(j)} = a^{(j)} + \phi^{(j)} x_{t-1}^{(j)} + \epsilon_t
\]

样本构造：

- 针对所有公司，所有 train 样本：
  - x_prev = \(x_{t-1}^\*\)
  - x_curr = \(x_t^\*\)
  - 只要 target \(x_t^\*\) 属于 train 区间，就拿来拟合 AR(1)

拟合方法：

- 每个 driver 维度独立，做一维 OLS：

  - 估计 \(\phi^{(j)} = \dfrac{\text{cov}(x_{t-1}^{(j)}, x_t^{(j)})}{\text{var}(x_{t-1}^{(j)})}\)
  - 截距 \(a^{(j)} = \mathbb{E}[x_t^{(j)}] - \phi^{(j)} \mathbb{E}[x_{t-1}^{(j)}]\)

预测：

\[
\hat{x}_t^{(j)} = a^{(j)} + \phi^{(j)} x_{t-1}^{(j)}
\]

- 在每个公司内部，用真实的上一期 driver 做输入，得到所有期（包括 test）的 AR 预测。

特点：

- pooling 所有公司，增加拟合稳健性；
- 各 driver 维度之间独立处理，模型非常轻量。

---

### 9.3 小网络（所有公司一起）

构造 supervised 样本：

- history_lag = 1（简单情况下）：

  - 输入：\(x_{t-1}^\*\) 展平成向量（11 维）
  - 输出：\(x_t^\*\)（11 维）

- 训练集：所有 target 属于 train 的样本；
- 测试集：target 属于 test 的样本（每家公司最后一个）。

网络结构（示意）：

- 输入层：维度 = history_lag * 11（例如 1×11）
- 隐含层：比如 2 层，每层 32 个神经元，激活 ReLU
- 输出层：11 维线性输出，对应下一期的 driver 向量

训练细节：

- 损失：MSE（所有 driver 维度一起）
- 优化器：Adam
- 需要做标准化：
  - 输入标准化：对 train 的 \(x_{t-1}^\*\) 做 z-score
  - 输出标准化：对 train 的 \(x_t^\*\) 做 z-score
  - 预测时反标准化回原始 scale

特点：

- 较 AR(1) 多一个非线性逼近的能力；
- 仍然非常小，参数量极少，基本不会 overfit 到每家只有 2 条 train 的程度（因为是在所有公司上一起训练）。

---

## 10. 回测：从 driver 预测到 state 预测

### 10.1 四个模型

从整体视角，我们有四个「模型」：

1. **Perfect drivers baseline**  
   - 直接用反演得到的 \(x_t^\*\) 驱动演化器：
     \[
     \hat{y}_{t+1}^{(perfect)} = f(y_t^{data}, x_t^\*)
     \]
   - 这是「演化方程本身的误差上界」：如果 MSE 接近 0，说明结构方程能几乎完美重构真实状态；后面 driver 模型带来的误差都是额外的。

2. **Sliding mean driver 模型**  
   - 对 test 样本，用第 9.1 节的滑动平均得到 \(\hat{x}_t^{(sliding)}\)：
     \[
     \hat{y}_{t+1}^{(sliding)} = f(y_t^{data}, \hat{x}_t^{(sliding)})
     \]

3. **AR(1) driver 模型**  
   - 对 test 样本，用 AR(1) 模型预测 \(\hat{x}_t^{(AR1)}\)：
     \[
     \hat{y}_{t+1}^{(AR1)} = f(y_t^{data}, \hat{x}_t^{(AR1)})
     \]

4. **小网络 driver 模型**  
   - 对 test 样本，用小网络预测 \(\hat{x}_t^{(NN)}\)：
     \[
     \hat{y}_{t+1}^{(NN)} = f(y_t^{data}, \hat{x}_t^{(NN)})
     \]

### 10.2 评价指标

对每个模型（perfect / sliding / AR1 / NN），在所有 test transition 上计算：

- \(y_{t+1}^{true}\)：报表反推出的真实下一期状态
- \(\hat{y}_{t+1}\)：通过演化方程和对应 driver 预测出来的状态

定义：

- 逐维误差：\(e = \hat{y}_{t+1} - y_{t+1}^{true}\)
- **MSE**（全维度平均）：
  \[
  \text{MSE} = \mathbb{E}\big[e^2\big]
  \]
- **MAE**：
  \[
  \text{MAE} = \mathbb{E}\big[|e|\big]
  \]

可以：

- 输出整体 MSE/MAE（12 维平均）；
- 也可以按维度输出 MSE/MAE，观察哪些 state 分量更难预测（比如 CASH / DEBT 可能更敏感于 driver 模型）。

---

## 11. 小结

1. **结构部分**：

   - 用 Vélez‑Pareja 的「tank 模型」构造 12 维状态向量 \(y_t\)；
   - 给出 11 维 driver 向量 \(x_t\)；
   - 建立显式的演化方程 \(y_{t+1} = f(y_t, x_t)\)。

2. **反演部分**：

   - 利用连续两期的 \(y_t, y_{t+1}\) 和 IS/CF 额外报表行，构造完美 driver \(x_t^\*\)；
   - 这是「真实世界」在模型结构下的 driver 序列。

3. **建模部分**：

   - 我们不黑箱拟合 \(y_{t+1}\)，而是建模 \(x_t^\*\) 的时间序列；
   - 三种 driver 模型：
     - per-firm sliding window mean；
     - pooled AR(1)；
     - pooled 小网络（带标准化）。

4. **回测部分**：

   - 固定演化器 \(f\)；
   - 用不同的 driver 来源（perfect / sliding / AR1 / NN）生成 \(\hat{y}_{t+1}\)；
   - 在 state 空间上评估 MSE / MAE，量化各个 driver 模型带来的误差。



不同的结果保存的时候也是存在result里面 然后用exp_id来标记 每个实验下面的有对应的learner.json 以及实验的log 最终做一个analysis坐标 代表不同方式情况下 还原财报的结果的MSE以及MAE的 Relative L1以及Relative L2的结果

