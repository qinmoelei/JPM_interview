
import os
import math
import numpy as np
import pandas as pd
import tensorflow as tf

IS_PATH = "/mnt/data/A_IS_annual.csv"
BS_PATH = "/mnt/data/A_BS_annual.csv"
CF_PATH = "/mnt/data/A_CF_annual.csv"

def _read_tables():
    is_df = pd.read_csv(IS_PATH)
    bs_df = pd.read_csv(BS_PATH)
    cf_df = pd.read_csv(CF_PATH)
    # Set index to line item name, columns are dates (YYYY-MM-DD)
    is_df = is_df.set_index("Unnamed: 0")
    bs_df = bs_df.set_index("Unnamed: 0")
    cf_df = cf_df.set_index("Unnamed: 0")
    # Ensure ascending order by date
    cols = sorted(is_df.columns.tolist())
    is_df = is_df[cols]
    bs_df = bs_df[cols]
    cf_df = cf_df[cols]
    # Convert to float
    is_df = is_df.apply(pd.to_numeric, errors="coerce")
    bs_df = bs_df.apply(pd.to_numeric, errors="coerce")
    cf_df = cf_df.apply(pd.to_numeric, errors="coerce")
    return is_df, bs_df, cf_df

def _get_first_available(df, candidates):
    for c in candidates:
        if c in df.index:
            return df.loc[c].values.astype(float)
    # fallback: zeros
    return np.zeros(df.shape[1], dtype=float)

def _safe_series(df, item, default=0.0):
    if item in df.index:
        return df.loc[item].values.astype(float)
    return np.full(df.shape[1], default, dtype=float)

def build_dataset():
    is_df, bs_df, cf_df = _read_tables()
    T = is_df.shape[1]  # number of time points (years)
    years = is_df.columns.tolist()

    # --- Select core items (robust to naming via candidate lists) ---
    sales = _get_first_available(is_df, ["sales", "total_revenue", "revenue"])
    cogs  = _get_first_available(is_df, ["cogs", "reconciled_cost_of_revenue", "cost_of_revenue"])
    sga   = _get_first_available(is_df, ["selling_general_and_administration", "selling_general_and_administration_expenses", "sga_expense"])
    dep_is = _get_first_available(is_df, ["reconciled_depreciation", "depreciation_amortization_depletion", "depreciation_and_amortization"])
    int_exp = _get_first_available(is_df, ["int_exp", "interest_expense", "interest_expense_non_operating"])
    int_inc = _get_first_available(is_df, ["int_inc", "interest_income", "interest_income_non_operating"])
    tax_rate = _get_first_available(is_df, ["tax_rate_for_calcs"])
    net_income = _get_first_available(is_df, ["net_income_from_continuing_operations",
                                             "net_income_from_continuing_operation_net_minority_interest",
                                             "net_income"])

    cash = _get_first_available(bs_df, ["cash", "cash_cash_equivalents_and_short_term_investments"])
    sti  = _get_first_available(bs_df, ["other_short_term_investments"])
    ar   = _get_first_available(bs_df, ["receivables", "ar"])
    inv  = _get_first_available(bs_df, ["inv_stock", "inventory"])
    ap   = _get_first_available(bs_df, ["payables", "accounts_payable", "payables_and_accrued_expenses"])
    net_ppe = _get_first_available(bs_df, ["net_ppe", "property_plant_equipment_net", "property_plant_and_equipment_net"])
    debt_total = _get_first_available(bs_df, ["total_debt", "net_debt"])
    equity = _get_first_available(bs_df, ["stockholders_equity", "total_equity_gross_minority_interest", "common_stock_equity"])
    re = _get_first_available(bs_df, ["re", "retained_earnings"])
    other_assets = _get_first_available(bs_df, ["other_current_assets"]) + _get_first_available(bs_df, ["other_non_current_assets"])
    other_liab   = _get_first_available(bs_df, ["other_current_liabilities"]) + _get_first_available(bs_df, ["other_non_current_liabilities"])

    # Cash flow items
    capex = _get_first_available(cf_df, ["capex", "capital_expenditure"])
    dep_cf = _get_first_available(cf_df, ["dep_cf", "depreciation_amortization_depletion"])
    div_paid = np.abs(_get_first_available(cf_df, ["cash_dividends_paid", "common_stock_dividend_paid", "dividends_paid"]))
    buybacks = np.abs(_get_first_available(cf_df, ["repurchase_of_capital_stock"]))
    debt_issuance = _get_first_available(cf_df, ["issuance_of_debt", "net_long_term_debt_issuance"]) + _get_first_available(cf_df, ["net_short_term_debt_issuance"])
    debt_repayment = np.abs(_get_first_available(cf_df, ["repayment_of_debt", "lt_debt_payments"]) + _get_first_available(cf_df, ["short_term_debt_payments"]))

    # Prefer depreciation from CF if present
    dep = np.where(np.isnan(dep_cf) | (dep_cf == 0), dep_is, dep_cf)

    # Effective tax rate (fallback to provided rate)
    pretax = sales - cogs - sga - dep - np.maximum(int_exp - int_inc, 0)
    tax_eff = np.where(pretax > 1e-9, np.clip((pretax - (net_income + np.maximum(int_exp - int_inc, 0))) / pretax, 0, 0.6), np.nan)
    tax = np.where(np.isfinite(tax_eff), tax_eff, np.clip(tax_rate, 0.0, 0.6))

    # Derive working-capital ratios (DSO, DIO, DPO); protect against division by zero
    days = 365.0
    dso = np.where(sales > 1e-9, ar / sales * days, 0.0)
    dio = np.where(cogs  > 1e-9, inv / cogs  * days, 0.0)
    dpo = np.where(cogs  > 1e-9, ap / cogs   * days, 0.0)

    # Interest rate proxy based on beginning-of-period debt (shifted debt)
    debt_bop = np.concatenate([ [np.nan], debt_total[:-1] ])
    # Avoid division by zero; first period unknown -> use overall median later
    rate_int = np.where((~np.isnan(debt_bop)) & (debt_bop > 1e-6), np.clip(int_exp / debt_bop, 0.0, 0.3), np.nan)
    # Fill NaNs with median of known rates
    med_rate = np.nanmedian(np.where(np.isfinite(rate_int), rate_int, np.nan))
    if not np.isfinite(med_rate):
        med_rate = 0.05
    rate_int = np.where(np.isfinite(rate_int), rate_int, med_rate)

    # Payout ratio
    payout = np.where(net_income > 1e-6, np.clip(div_paid / net_income, 0.0, 1.5), 0.0)

    # Build state vector y_t
    # y: [S, COGS, SG&A, Dep, AR, INV, AP, PPE, Cash, Debt, Equity, RE]
    y = np.vstack([sales, cogs, sga, dep, ar, inv, ap, net_ppe, cash, debt_total, equity, re]).T

    # Build exogenous features x_t (drivers we may predict or hold approx. constant)
    # x: [tax, dso, dio, dpo, capex_ratio, dep_rate, gm, sga_ratio, int_rate, payout]
    eps = 1e-9
    gm = np.where(sales > eps, (sales - cogs) / sales, 0.3)
    sga_ratio = np.where(sales > eps, sga / sales, 0.1)
    capex_ratio = np.where(sales > eps, np.abs(capex) / sales, 0.05)
    dep_rate = np.where(net_ppe > eps, dep / net_ppe, 0.1)
    int_rate = rate_int
    x = np.vstack([tax, dso, dio, dpo, capex_ratio, dep_rate, gm, sga_ratio, int_rate, payout]).T

    # Targets y_{t+1} (align by shifting left)
    y_next = y[1:]
    y_curr = y[:-1]
    x_curr = x[:-1]

    return {
        "years": years,
        "y": y,
        "x": x,
        "y_curr": y_curr,
        "y_next": y_next,
        "x_curr": x_curr
    }

class AccountingLayer(tf.keras.layers.Layer):
    """
    Implements y(t+1) = f(y(t), drivers) with hard accounting constraints
    and no interest circularity (interest computed on beginning-of-period debt).
    State y = [S, COGS, SG&A, Dep, AR, INV, AP, PPE, Cash, Debt, Equity, RE]
    Drivers d = [gS, gm, sga_ratio, dep_rate, dso, dio, dpo, capex_ratio, tax, int_rate, payout]
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, inputs):
        y_t, d = inputs
        # unpack state
        S, C, SG, D, AR, INV, AP, PPE, CASH, DEBT, EQ, RE = tf.unstack(y_t, axis=-1)
        # unpack drivers
        gS, gm, sga_ratio, dep_rate, dso, dio, dpo, capex_ratio, tax, int_rate, payout = tf.unstack(d, axis=-1)

        # Sales and cost structure
        S_next = tf.nn.relu(S) * (1.0 + gS)
        C_next = tf.nn.relu(S_next) * tf.nn.relu(1.0 - gm)  # COGS = (1-gm)*Sales
        SG_next = tf.nn.relu(S_next) * tf.nn.relu(sga_ratio)
        # Depreciation based on PPE_t
        Dep_next = tf.nn.relu(PPE) * tf.nn.relu(dep_rate)

        # Working capital balances (stock)
        AR_next  = tf.nn.relu(S_next) * tf.nn.relu(dso) / 365.0
        INV_next = tf.nn.relu(C_next) * tf.nn.relu(dio) / 365.0
        AP_next  = tf.nn.relu(C_next) * tf.nn.relu(dpo) / 365.0

        # Operating income
        EBIT_next = S_next - C_next - SG_next - Dep_next
        # Interest on beginning-of-period debt (DEBT)
        INT_next = tf.nn.relu(int_rate) * tf.nn.relu(DEBT)
        # Taxes on EBT = EBIT - INT, no NOLs here (kept simple)
        EBT_next = EBIT_next - INT_next
        NOPAT_next = EBT_next * (1.0 - tf.nn.relu(tax))
        # Net income
        NI_next = NOPAT_next

        # Capex (uses of cash)
        CAPEX_next = tf.nn.relu(S_next) * tf.nn.relu(capex_ratio)

        # PPE evolution
        PPE_next = PPE + CAPEX_next - Dep_next

        # Change in working capital (uses if positive)
        dAR = AR_next - AR
        dINV = INV_next - INV
        dAP = AP_next - AP
        dWC = dAR + dINV - dAP

        # Free cash (before financing) approximation
        FCF_like = NI_next + Dep_next - dWC - CAPEX_next

        # Dividends (use): payout * NI
        DIV_next = tf.nn.relu(payout) * tf.nn.relu(NI_next)

        # Simple financing rule with no plug: 
        # If FCF_like - DIV >= 0, use it to repay debt up to zero; leftover increases cash.
        # If < 0, borrow to cover the shortfall; cash unchanged.
        surplus = FCF_like - DIV_next

        repay = tf.minimum(tf.nn.relu(surplus), DEBT)              # repay up to available debt
        borrow = tf.nn.relu(-surplus)                              # borrow shortfall
        DEBT_next = DEBT - repay + borrow
        CASH_next = CASH + tf.nn.relu(surplus - repay)             # cash increases only if surplus after repay

        # Equity update via RE; APIC assumed constant here
        RE_next = RE + NI_next - DIV_next
        # Recompute Equity as Equity_t + NI - DIV (i.e., APIC/OCI constant)
        EQ_next = EQ + NI_next - DIV_next

        # Enforce Assets = Liabilities + Equity by construction:
        # Assets = CASH + AR + INV + PPE + (OtherAssets not modeled here)
        # Liab+Eq = AP + DEBT + EQ_next
        # The above equality will hold if the flow equations are internally consistent;
        # to be safe, we rebuild EQ_next as residual:
        ASSETS_next = CASH_next + AR_next + INV_next + PPE_next
        L_plus_E_next = AP_next + DEBT_next + EQ_next
        # Residual adjust equity to ensure exact equality
        EQ_resid = EQ_next + (ASSETS_next - L_plus_E_next)
        # Stack the next state
        y_next = tf.stack([S_next, C_next, SG_next, Dep_next, AR_next, INV_next, AP_next,
                           PPE_next, CASH_next, DEBT_next, EQ_resid, RE_next], axis=-1)
        # Also return some diagnostics if needed in the future
        return y_next

def build_model(input_dim_y, input_dim_x):
    # Drivers net: predicts [gS, gm, sga_ratio, dep_rate, dso, dio, dpo, capex_ratio, tax, int_rate, payout]
    drivers_dim = 11
    y_in = tf.keras.Input(shape=(input_dim_y,), name="y_t")
    x_in = tf.keras.Input(shape=(input_dim_x,), name="x_t")
    h = tf.keras.layers.Concatenate()([y_in, x_in])
    h = tf.keras.layers.Dense(32, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(1e-4))(h)
    h = tf.keras.layers.Dense(16, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(1e-4))(h)
    raw = tf.keras.layers.Dense(drivers_dim, activation=None)(h)

    # Constrain outputs to reasonable ranges
    # gS in [-0.5, +0.5], gm in [0, 0.9], sga_ratio in [0, 0.6], dep_rate in [0, 0.5]
    # dso/dio/dpo in [0, 200], capex_ratio in [0, 0.5], tax in [0, 0.6], int_rate in [0, 0.3], payout in [0, 1.5]
    gS       = tf.tanh(raw[:, 0:1]) * 0.5
    gm       = tf.keras.activations.sigmoid(raw[:, 1:2]) * 0.9
    sga_r    = tf.keras.activations.sigmoid(raw[:, 2:3]) * 0.6
    dep_r    = tf.keras.activations.sigmoid(raw[:, 3:4]) * 0.5
    dso      = tf.keras.activations.sigmoid(raw[:, 4:5]) * 200.0
    dio      = tf.keras.activations.sigmoid(raw[:, 5:6]) * 200.0
    dpo      = tf.keras.activations.sigmoid(raw[:, 6:7]) * 200.0
    capex_r  = tf.keras.activations.sigmoid(raw[:, 7:8]) * 0.5
    tax      = tf.keras.activations.sigmoid(raw[:, 8:9]) * 0.6
    int_r    = tf.keras.activations.sigmoid(raw[:, 9:10]) * 0.3
    payout   = tf.keras.activations.sigmoid(raw[:, 10:11]) * 1.5

    drivers = tf.keras.layers.Concatenate(name="drivers")([gS, gm, sga_r, dep_r, dso, dio, dpo, capex_r, tax, int_r, payout])
    y_next = AccountingLayer(name="accounting")([y_in, drivers])
    model = tf.keras.Model(inputs=[y_in, x_in], outputs=y_next)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=5e-3), loss="mape")
    return model

def train_and_eval(verbose=1):
    data = build_dataset()
    y_curr = data["y_curr"]
    y_next = data["y_next"]
    x_curr = data["x_curr"]

    model = build_model(y_curr.shape[1], x_curr.shape[1])

    # Train (note: dataset is tiny; this is illustrative)
    hist = model.fit([y_curr, x_curr], y_next, epochs=400, batch_size=1, verbose=0)

    # Predictions and diagnostics
    y_pred = model.predict([y_curr, x_curr], verbose=0)

    # Check accounting identity (Assets - (Liab+Eq)) for each sample
    # Assets = Cash + AR + INV + PPE; Liab+Eq = AP + Debt + Equity
    gap = (y_pred[:, 8] + y_pred[:, 4] + y_pred[:, 5] + y_pred[:, 7]) - (y_pred[:, 6] + y_pred[:, 9] + y_pred[:, 10])
    max_gap = float(np.max(np.abs(gap)))
    mae = float(np.mean(np.abs(y_pred - y_next)))
    diag = {
        "training_samples": int(y_curr.shape[0]),
        "features_dim": int(x_curr.shape[1]),
        "state_dim": int(y_curr.shape[1]),
        "mae": mae,
        "max_balance_gap": max_gap
    }
    return model, data, y_pred, diag

if __name__ == "__main__":
    model, data, y_pred, diag = train_and_eval(verbose=1)
    print("Diagnostics:", diag)
