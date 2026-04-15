"""
StrategyOrchestrator — Cross-Sectional Only

Lean allocation pipeline:
  Step 1: compute CS raw weights from CrossSectionalAllocator
  Step 2: apply CS risk gating (breadth → defensive assets)
  Step 3: apply sleeve cap → w_final; run invariants
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple
from core.cross_sectional_allocator import CrossSectionalAllocator

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _scale_weights(raw: Dict[str, float], target_total: float) -> Tuple[Dict[str, float], float]:
    """
    Proportionally scale a weight dict so its sum == target_total.
    Returns (scaled_dict, actual_total_applied).
    If raw sum is 0, returns zeros.
    """
    raw_sum = sum(raw.values())
    if raw_sum <= 0:
        return {s: 0.0 for s in raw}, 0.0
    scale = target_total / raw_sum
    scaled = {s: w * scale for s, w in raw.items()}
    return scaled, sum(scaled.values())


# ---------------------------------------------------------------------------
# main orchestrator
# ---------------------------------------------------------------------------

class StrategyOrchestrator:
    """
    Cross-Sectional-only strategy logic for production / live trading.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.cs_allocator = CrossSectionalAllocator(config)

    def generate_portfolio_intents(
        self,
        multi_history: Dict[str, pd.Series],
        current_equity: float,
        current_positions: Dict[str, float],
        current_date: pd.Timestamp,
        runtime_state: Dict[str, Any] = None,
    ) -> List[Dict[str, Any]]:
        """
        Returns intents: List[{symbol, signal, weight, attribution, reason}]
        """
        # ── read config ────────────────────────────────────────────────────
        budgets = self.config.get('budgets', {})
        cs_total_max = budgets.get('cs_total_max', 0.98)
        cash_buffer  = budgets.get('cash_buffer', 0.02)

        # CS risk gating configuration
        cs_risk_conf = self.config.get('cs_risk', {})
        cs_risk_enabled = cs_risk_conf.get('enabled', False)
        cs_lookback = cs_risk_conf.get('lookback_days', 21)
        cs_pct_threshold = cs_risk_conf.get('pct_negative_threshold', 0.70)
        cs_min_days = cs_risk_conf.get('min_days_in_state', 3)
        defensive_assets = cs_risk_conf.get('defensive_assets', ["SHY", "IEF"])
        defensive_split = cs_risk_conf.get('defensive_split', 'equal')
        cash_fraction_of_cs = cs_risk_conf.get('cash_fraction_of_cs', 0.50)
        cs_log_debug = cs_risk_conf.get('log_debug', False)

        # ══════════════════════════════════════════════════════════════════
        # STEP 1 — CS RAW WEIGHTS
        # ══════════════════════════════════════════════════════════════════
        w_cs_raw = self.cs_allocator.get_target_weights(multi_history, current_date)
        cs_debug = self.cs_allocator.last_debug

        # --- Compute CS risk raw signal (pct_negative) ---
        pct_negative = 0.0
        raw_risk_off = False
        cs_universe = getattr(self.cs_allocator, 'universe', [])
        try:
            neg_count = 0
            considered = 0
            for t in cs_universe:
                hist = multi_history.get(t)
                if hist is None or len(hist) <= cs_lookback:
                    continue
                close_now = float(hist.iloc[-1])
                close_then = float(hist.iloc[-1 - cs_lookback])
                R = (close_now / close_then) - 1.0 if close_then != 0 else 0.0
                considered += 1
                if R < 0:
                    neg_count += 1
            pct_negative = (neg_count / considered) if considered > 0 else 0.0
            raw_risk_off = pct_negative >= cs_pct_threshold
        except Exception:
            pct_negative = 0.0
            raw_risk_off = False

        # --- Runtime-state persistence for CS risk ---
        if runtime_state is None:
            runtime_state = {}
        current_state = runtime_state.get('cs_risk_state', 'RISK_ON')
        streak = int(runtime_state.get('cs_risk_streak', 0))
        desired_state = 'RISK_OFF' if raw_risk_off else 'RISK_ON'
        if desired_state == current_state:
            streak += 1
        else:
            streak = 1
        if desired_state != current_state and streak >= int(cs_min_days):
            current_state = desired_state
            streak = 0
        # update runtime_state (mutate in-place)
        runtime_state['cs_risk_state'] = current_state
        runtime_state['cs_risk_streak'] = streak

        # Debug log
        if cs_log_debug:
            logger.info("CS_RISK_DEBUG pct_negative=%.3f raw_risk_off=%s state=%s streak=%d",
                        pct_negative, raw_risk_off, current_state, streak)

        # ══════════════════════════════════════════════════════════════════
        # STEP 2 — APPLY SLEEVE CAP + CS RISK GATING
        # ══════════════════════════════════════════════════════════════════
        budget_remaining = 1.0 - cash_buffer
        cs_budget = min(cs_total_max, max(budget_remaining, 0.0))

        cs_mode = 'NORMAL'
        if cs_risk_enabled and current_state == 'RISK_OFF':
            cs_mode = 'DEFENSIVE'
            cs_budget_val = cs_budget
            if cs_budget_val <= 0 or not self.config.get('cs_allocator', {}).get('enabled', True):
                w_cs_raw = {t: 0.0 for t in cs_universe}
                w_cs_scaled = dict(w_cs_raw)
                cs_total = 0.0
            else:
                available_def = [a for a in defensive_assets if a in multi_history and len(multi_history[a]) > 0]
                if len(available_def) != len(defensive_assets):
                    w_cs_raw = {t: 0.0 for t in cs_universe}
                    w_cs_scaled = dict(w_cs_raw)
                    cs_total = 0.0
                else:
                    defensive_weight_total = (1.0 - float(cash_fraction_of_cs)) * cs_budget_val
                    ndef = len(available_def)
                    per = defensive_weight_total / max(1, ndef)
                    new_raw = {t: 0.0 for t in cs_universe}
                    for a in available_def:
                        new_raw[a] = per
                    w_cs_raw = new_raw
                    w_cs_scaled = dict(w_cs_raw)
                    cs_total = sum(w_cs_raw.values())
        else:
            # Normal CS behavior
            cs_raw_sum = sum(w_cs_raw.values())
            if cs_raw_sum > cs_budget and cs_raw_sum > 0:
                w_cs_scaled, cs_total = _scale_weights(w_cs_raw, cs_budget)
                w_cs_raw = dict(w_cs_scaled)
            else:
                w_cs_scaled = dict(w_cs_raw)
                cs_total = cs_raw_sum

        # Augment CS debug with gating fields
        cs_debug_aug = dict(cs_debug or {})
        cs_debug_aug.update({
            'cs_risk_state': runtime_state.get('cs_risk_state'),
            'cs_risk_streak': runtime_state.get('cs_risk_streak'),
            'pct_negative': pct_negative,
            'raw_risk_off': raw_risk_off,
            'cs_mode': cs_mode,
        })

        logger.info("CS_DEBUG universe=%s eligible=%s scores=%s ranked=%s selected=%s cs_total=%.4f cs_mode=%s pct_negative=%.3f",
                    cs_debug_aug.get('universe', []), cs_debug_aug.get('eligible', []),
                    cs_debug_aug.get('scores', {}), cs_debug_aug.get('ranked', []),
                    cs_debug_aug.get('selected', []), cs_total, cs_mode, pct_negative)

        # ══════════════════════════════════════════════════════════════════
        # STEP 3 — BUILD FINAL WEIGHTS
        # ══════════════════════════════════════════════════════════════════
        all_symbols = set(multi_history.keys())
        final_weights: Dict[str, float] = {s: 0.0 for s in all_symbols}

        for s, w in w_cs_scaled.items():
            final_weights[s] = final_weights.get(s, 0.0) + w

        # Log final breakdown
        for s, w_final in final_weights.items():
            w_cs_attr = float(w_cs_scaled.get(s, 0.0))
            logger.info(
                "FINAL_TARGET_BREAKDOWN %s: w_cs=%.4f w_final=%.4f",
                s, w_cs_attr, w_final,
            )

        # ══════════════════════════════════════════════════════════════════
        # INVARIANT CHECKS — FAIL FAST
        # ══════════════════════════════════════════════════════════════════

        # 1. Budget overflow
        total_weight = sum(final_weights.values())
        if total_weight > 1.0 + 1e-6:
            raise RuntimeError(
                f"Budget overflow detected: sum(w_final)={total_weight:.6f} > 1.0")

        # 2. No negatives (CS is long-only)
        neg = {s: w for s, w in final_weights.items() if w < 0}
        if neg:
            raise RuntimeError(
                f"Negative weights detected (CS is long-only): {neg}")

        # 3. CS K check
        cs_top_k = self.cs_allocator.top_k
        cs_selected = cs_debug.get('selected', []) if cs_debug else []
        cs_eligible = cs_debug.get('eligible', []) if cs_debug else []
        if len(cs_eligible) >= cs_top_k and len(cs_selected) < cs_top_k:
            raise RuntimeError(
                f"CS top_k={cs_top_k} but only {len(cs_selected)} selected "
                f"from {len(cs_eligible)} eligible — check CS logic")

        # ══════════════════════════════════════════════════════════════════
        # BUILD INTENTS
        # ══════════════════════════════════════════════════════════════════
        intents = []
        for s, w in final_weights.items():
            w_cs_attr = float(w_cs_scaled.get(s, 0.0))

            if w_cs_attr > 0:
                reason = "CS_Allocator"
            else:
                reason = "Flat"

            intents.append({
                "symbol":  s,
                "signal":  "TARGET_WEIGHT",
                "weight":  float(w),
                "attribution": {
                    "w_cs":     w_cs_attr,
                    "cs_total": cs_total,
                    "cs_debug": cs_debug_aug,
                },
                "reason": reason,
            })

        return intents
