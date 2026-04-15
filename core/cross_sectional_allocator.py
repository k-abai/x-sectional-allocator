import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

class CrossSectionalAllocator:
    """
    Cross-Sectional MA200 Allocator.
    Selects the top-K performing assets from a universe based on proximity to MA200.
    Monthly rebalancing.
    """
    def __init__(self, config: Dict):
        self.config = config.get('cs_allocator', {})
        self.universe = self.config.get('universe', ['QQQ', 'EFA', 'EEM'])
        self.ma_window = self.config.get('ma_window', 200)
        self.top_k = self.config.get('top_k', 3)
        self.allocation_mode = self.config.get('allocation_mode', 'equal')
        self.current_weights = {ticker: 0.0 for ticker in self.universe}
        self.last_rebalance_month = None
        self.last_debug = {}  # Store last CS_DEBUG for external access

    def get_target_weights(self, multi_history: Dict[str, pd.Series], current_date: pd.Timestamp) -> Dict[str, float]:
        """
        Returns target weights for the universe.
        Only rebalances on the first trading day of the month.
        """
        current_month = current_date.month
        
        # Rebalance only if month has changed
        if self.last_rebalance_month != current_month:
            self.last_rebalance_month = current_month
            self.current_weights, self.last_debug = self._calculate_selection(multi_history)
            
        return self.current_weights

    def _calculate_selection(self, multi_history: Dict[str, pd.Series]):
        """
        Logic:
        1. Compute s_i = P/MA200 - 1
        2. Filter eligible (s_i > 0)
        3. Rank descending
        4. Select top-K
        5. Allocate within sleeve (equal or proportional)

        Returns:
            (target_weights, cs_debug_dict)
        """
        scores = {}
        eligibility = {}

        for ticker in self.universe:
            history = multi_history.get(ticker)
            if history is not None and len(history) >= self.ma_window:
                price = history.iloc[-1]
                ma = history.rolling(window=self.ma_window).mean().iloc[-1]
                score = (price / ma) - 1.0
                scores[ticker] = score
                if score > 0:
                    eligibility[ticker] = "ELIGIBLE"
                else:
                    eligibility[ticker] = f"INELIGIBLE: score={score:.4f} <= 0"
            else:
                data_len = len(history) if history is not None else 0
                scores[ticker] = -np.inf
                eligibility[ticker] = f"INELIGIBLE: insufficient data ({data_len} < {self.ma_window})"

        # Filter for s_i > 0
        valid_scores = {t: s for t, s in scores.items() if s > 0}

        target_weights = {ticker: 0.0 for ticker in self.universe}

        # Rank and select top-K
        ranked = sorted(valid_scores.keys(), key=lambda t: valid_scores[t], reverse=True)
        selected = ranked[:self.top_k]

        if selected:
            if self.allocation_mode == 'proportional' and len(selected) > 0:
                total_score = sum(valid_scores[t] for t in selected)
                if total_score > 0:
                    for t in selected:
                        target_weights[t] = valid_scores[t] / total_score
                else:
                    w_per = 1.0 / len(selected)
                    for t in selected:
                        target_weights[t] = w_per
            else:  # equal
                w_per = 1.0 / len(selected)
                for t in selected:
                    target_weights[t] = w_per

        # Build CS_DEBUG
        cs_debug = {
            'universe': list(self.universe),
            'eligible': [t for t in self.universe if eligibility.get(t, '').startswith('ELIGIBLE')],
            'scores': {t: float(s) if s != -np.inf else None for t, s in scores.items()},
            'eligibility': eligibility,
            'ranked': ranked,
            'selected': selected,
            'top_k': self.top_k,
            'allocation_mode': self.allocation_mode,
            'weights': dict(target_weights),
        }

        # Log CS_DEBUG
        logger.info("CS_DEBUG: universe=%s eligible=%s scores=%s ranked=%s selected=%s weights=%s",
                     cs_debug['universe'], cs_debug['eligible'], cs_debug['scores'],
                     cs_debug['ranked'], cs_debug['selected'], cs_debug['weights'])

        return target_weights, cs_debug

    def get_debug_info(self, multi_history: Dict[str, pd.Series] = None) -> Dict:
        """Returns latest CS_DEBUG dict, or computes fresh if multi_history provided."""
        if multi_history is not None:
            _, debug = self._calculate_selection(multi_history)
            return debug
        return self.last_debug
