
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

### 9.1 自动 sliding window mean/ AR(1)（单个公司）

思想：

- 对单个公司，时间 \(t\) 的预测 driver 简单取「过去 \(k\) 期的均值」。

形式：

\[
\hat{x}_t = \frac{1}{k} \sum_{i=1}^{k} x_{t-i}^\*
\]

实现上的细节：

- \(k\) 可取 2（或者其他）
- 如果历史长度不足 \(k\)，就取能拿到的全部历史（如只有 1 期，就用那 1 期）
- 因为我们只用最后一条做 test，对 test 点来说总有至少 2 条历史可用 (因为train的数据有两条driver 可以用这两条driver)。

特点：

- 完全 per-firm，不 pooling；
- 相当于一个非常简单的 smoothing baseline。

---

### 9.2 类 ARIMA 拟合：（所有公司一起）

train 两个数据点 fit一下 arima的数据 ARIMA （因为做完integrated只有一个数据点了 因此可能只能arma）：

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

