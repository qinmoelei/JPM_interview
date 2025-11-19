from __future__ import annotations
import tensorflow as tf

class CashBudgetLayer(tf.keras.layers.Layer):
    """
    Differentiable implementation of the Vélez-Pareja cash-budget & statements engine.

    Inputs (per time step):
        states_{t-1}: [C, INV, K, B_ST, B_LT, RE, PIC, AR, AP, INV_STOCK, OtherA, OtherLE]
        drivers_t: [Sales, COGS, Opex, Dep, Capex, r_ST, r_LT, r_INV, Amort_LT, Cbar, EI, Div, tau, DSO, DPO, DIO]

    Outputs:
        Next-period states plus key IS components (EBITDA, EBIT, EBT, taxes, NI, interest flows).

    Reference:
        Vélez-Pareja, I. (2005). "The Cash Budget and Cash Flow Forecast."
        SSRN: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=742285

    Example:
        >>> layer = CashBudgetLayer()
        >>> states_prev = tf.zeros((1, 12))
        >>> drivers = tf.ones((1, 16))
        >>> next_states, is_out = layer((states_prev, drivers))
        >>> next_states.shape
        TensorShape([1, 12])
    """
    def call(self, inputs):
        states_prev, drivers = inputs
        # Unpack states
        (C_prev,
         INV_prev,
         K_prev,
         BST_prev,
         BLT_prev,
         RE_prev,
         PIC_prev,
         AR_prev,
         AP_prev,
         INVST_prev,
         OA_prev,
         OLE_prev) = tf.unstack(states_prev, axis=-1)
        # Unpack drivers
        (Sales, COGS, Opex, Dep, Capex, rST, rLT, rINV, AmortLT, Cbar, EI, Div, tau, DSO, DPO, DIO) = tf.unstack(drivers, axis=-1)

        # Step 1: Working capital stocks (days method)
        AR_t   = DSO/365.0 * Sales
        AP_t   = DPO/365.0 * tf.where(tf.not_equal(COGS, 0.0), COGS, tf.ones_like(COGS))
        INVST_t= DIO/365.0 * tf.where(tf.not_equal(COGS, 0.0), COGS, tf.ones_like(COGS))
        dWC    = (AR_t + INVST_t - AP_t) - (AR_prev + INVST_prev - AP_prev)

        # Step 2: Interests on beginning balances
        iST = rST * BST_prev
        iLT = rLT * BLT_prev
        iINV = rINV * INV_prev

        # Step 3: EBITDA and IS skeleton
        EBITDA = Sales - COGS - Opex
        EBIT   = EBITDA - Dep
        EBT    = EBIT - iST - iLT + iINV
        Tax    = tf.where(EBT > 0.0, tau * EBT, tf.zeros_like(EBT))
        NI     = EBT - Tax

        # Step 4: Short-term stage of CB (pre-capex)
        # Operating cash proxy before long-term service/capex
        # Note: here dividends reduce cash in CB; they enter RE via NI-Div later
        pST = BST_prev
        NCB_oper = EBITDA - dWC - iST - pST - Tax - Div + iINV

        ST_new = tf.nn.relu(Cbar - (C_prev + NCB_oper))

        # Step 5: Long-term stage (post-capex and LT service, net of EI)
        pLT = AmortLT
        Cash_postInv = C_prev + NCB_oper + ST_new - Capex - pLT - iLT + EI
        LT_new = tf.nn.relu(Cbar - Cash_postInv)

        # Step 6: Close cash and update short-term investments
        dINV = Cash_postInv + LT_new - Cbar
        C_t  = Cbar
        INV_t = tf.nn.relu(INV_prev + dINV)

        # Step 7: Update remaining balances
        K_t   = K_prev + Capex - Dep
        BST_t = ST_new
        BLT_t = BLT_prev + LT_new - pLT
        PIC_t = PIC_prev + EI
        RE_t  = RE_prev + NI - Div

        # Output packing
        eps = tf.constant(1e-6, dtype=states_prev.dtype)
        assets_core_prev = C_prev + INV_prev + K_prev + AR_prev + INVST_prev
        assets_core_t = C_t + INV_t + K_t + AR_t + INVST_t
        ratio = tf.where(
            tf.abs(assets_core_prev) > eps,
            OA_prev / (assets_core_prev + tf.sign(assets_core_prev) * eps),
            tf.zeros_like(OA_prev),
        )
        other_assets_t = ratio * assets_core_t
        assets_total = assets_core_t + other_assets_t
        liab_core = BST_t + BLT_t + AP_t
        equity_core = PIC_t + RE_t
        diff = assets_total - (liab_core + equity_core + OLE_prev)
        other_lae_t = OLE_prev + diff

        states_t = tf.stack(
            [C_t, INV_t, K_t, BST_t, BLT_t, RE_t, PIC_t, AR_t, AP_t, INVST_t, other_assets_t, other_lae_t],
            axis=-1,
        )
        is_outputs = tf.stack([EBITDA, EBIT, EBT, Tax, NI, iST, iLT, iINV], axis=-1)
        return states_t, is_outputs
