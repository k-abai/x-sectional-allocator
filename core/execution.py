from core.logger import SystemLogger
from core.safety import SafetyManager
from core.notifier import Notifier
from core.alpaca_broker import AlpacaBroker
import json
from datetime import datetime, timedelta


def clamp_weight(w: float, w_max: float) -> float:
    """Clamp weight to [0, w_max] — long-only for CS."""
    return max(0.0, min(w_max, w))


class ExecutionEngine:
    def __init__(self, config):
        self.config = config
        self.logger = SystemLogger()
        self.safety = SafetyManager(config)
        self.notifier = Notifier(config)

    def is_dry_run(self) -> bool:
        """Return True if execution is configured as dry-run (no orders)."""
        strat = self.config.get('strategy', {}) if isinstance(self.config, dict) else {}
        return bool(strat.get('DRY_RUN_ALLOC', False))

    def validate_exposure_invariants(self, intents, strat_config):
        """Validate exposure invariants before execution."""
        final_weights = {intent['symbol']: intent['weight'] for intent in intents}

        # 1. Gross exposure check
        gross_exposure = sum(abs(w) for w in final_weights.values())
        portfolio_w_max = strat_config.get('portfolio', {}).get('W_max', 1.0)
        if gross_exposure > portfolio_w_max + 1e-6:
            raise RuntimeError(f"Gross exposure violation: {gross_exposure:.6f} > {portfolio_w_max}")

        # 2. CS sleeve total
        budgets = strat_config.get('budgets', {})
        cs_total_max = budgets.get('cs_total_max', 0.98)

        cs_total = sum(intent['attribution']['w_cs'] for intent in intents)
        if cs_total > cs_total_max + 1e-6:
            raise RuntimeError(f"CS total violation: {cs_total:.6f} > {cs_total_max}")

        # 3. No shorts (CS is long-only)
        neg_weights = {s: w for s, w in final_weights.items() if w < 0}
        if neg_weights:
            raise RuntimeError(f"Negative weights detected (CS is long-only): {neg_weights}")

        self.logger.log_event("EXPOSURE_VALIDATION_PASSED", {
            "gross_exposure": gross_exposure,
            "cs_total": cs_total,
        })

    def execute_netted_rebalance(self, intents, last_trade_timestamps):
        """Execute netted rebalance with guardrails."""
        # Convert intents to target weights
        target_weights = {intent['symbol']: intent['weight'] for intent in intents}

        # Get current portfolio
        account = self.broker.get_account()
        equity = account['equity']
        positions = self.broker.get_positions()
        current_weights = {}

        for pos in positions:
            symbol = pos['symbol']
            qty = pos['qty']
            current_price = pos['current_price']
            weight = (qty * current_price) / equity if equity > 0 else 0
            current_weights[symbol] = weight

        # Compute net deltas
        all_symbols = set(target_weights.keys()) | set(current_weights.keys())
        deltas = {}

        for symbol in all_symbols:
            target = target_weights.get(symbol, 0.0)
            current = current_weights.get(symbol, 0.0)
            delta = target - current
            if abs(delta) >= 0.02:  # Minimum rebalance threshold
                deltas[symbol] = delta

        # Apply guardrails
        filtered_deltas = {}
        for symbol, delta in deltas.items():
            # Check cooldown (6 hours)
            last_trade = last_trade_timestamps.get(symbol)
            if last_trade:
                last_trade_dt = datetime.fromisoformat(last_trade)
                if datetime.now() - last_trade_dt < timedelta(hours=6) and abs(delta) < 0.05:
                    continue

            # Check for open orders
            if self.broker.has_open_order(symbol):
                continue

            filtered_deltas[symbol] = delta

        # Sort: sells first, then buys
        sells = {s: d for s, d in filtered_deltas.items() if d < 0}
        buys = {s: d for s, d in filtered_deltas.items() if d > 0}

        # Execute sells first
        for symbol, delta in sells.items():
            self._execute_weight_delta(symbol, delta, equity)
            last_trade_timestamps[symbol] = datetime.now().isoformat()

        # Execute buys
        for symbol, delta in buys.items():
            self._execute_weight_delta(symbol, delta, equity)
            last_trade_timestamps[symbol] = datetime.now().isoformat()

    def _execute_weight_delta(self, symbol, delta_weight, equity):
        """Execute a weight delta for a symbol."""
        try:
            # If configured for dry-run, log the intent and do not place orders.
            if self.is_dry_run():
                self.logger.log_event(
                    "DRY_RUN_ORDER",
                    {"symbol": symbol, "delta_weight": delta_weight, "equity": equity}
                )
                return

            # Get current price
            price = self.broker.get_latest_price(symbol)

            # Calculate dollar amount
            dollar_delta = delta_weight * equity

            # Calculate quantity
            qty = int(dollar_delta / price)

            if qty == 0:
                return

            side = 'sell' if qty < 0 else 'buy'
            qty = abs(qty)

            # Submit order
            self.broker.submit_order(symbol, qty, side)
        except Exception as e:
            self.logger.log_error(f"Execution failed for {symbol}: {e}")

    def process_intent(self, intent: dict):
        """
        Main entry point for strategy signals.
        intent: {
          "symbol": "QQQ",
          "signal": "TARGET_WEIGHT",
          "weight": 0.49,
          "attribution": {...},
          "reason": "..."
        }
        """
        symbol = intent['symbol']
        target_weight = intent.get('weight', 0.0)

        self.logger.log_event("INTENT_RECEIVED", intent)

        # 1. Kill Switch Check
        if self.safety.is_kill_switch_active:
            self.logger.log_event("EXECUTION_ABORTED", {"reason": "KillSwitch"})
            return

        # 2. Normal Execution Logic
        account = self.broker.get_account()
        equity = account['equity']

        # Max Position Safety — long-only clamping
        max_pct = self.config.get('execution', {}).get('max_position_pct', 0.5)
        original_weight = target_weight
        target_weight = clamp_weight(target_weight, max_pct)

        if original_weight != target_weight:
            self.logger.log_event("WEIGHT_CAPPED", {"original": original_weight, "capped": target_weight}, level="WARNING")

        target_value = equity * target_weight

        # Current Position
        pos = self.broker.get_position(symbol)
        current_qty = pos['qty'] if pos else 0.0

        try:
            price = self.broker.get_latest_price(symbol)
        except:
            self.logger.log_error("Could not fetch price for execution", {"symbol": symbol})
            return

        current_value = current_qty * price
        delta_value = target_value - current_value

        # Threshold to avoid noise
        if abs(delta_value) < 100:
            return

        qty_to_trade = int(delta_value / price)

        if qty_to_trade == 0:
            return

        side = 'buy' if qty_to_trade > 0 else 'sell'
        qty = abs(qty_to_trade)

        # Execute
        try:
            self.broker.submit_order(symbol, qty, side)
        except Exception as e:
            self.logger.log_error(f"Execution failed: {str(e)}")

    def panic_close_all(self):
        """Kill switch trigger action."""
        self.safety.activate_kill_switch()
        self.broker.cancel_all_orders()
        self.broker.close_all_positions()
        self.logger.log_event("PANIC_STOP_TRIGGERED", {"reason": "User/System Kill Switch"}, level="CRITICAL")
