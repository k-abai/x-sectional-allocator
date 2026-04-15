import os
import time
import alpaca_trade_api as tradeapi
from core.broker_interface import BrokerInterface
from core.load_keys import load_keys
from core.logger import SystemLogger

class AlpacaBroker(BrokerInterface):
    def __init__(self, mode: str = 'paper'):
        self.logger = SystemLogger()
        self.mode = mode.lower()
        
        # Load Keys
        keys = load_keys()
        self.api_key = keys['api_key']
        self.secret_key = keys['secret_key']
        env_base = keys.get('base_url') or ''
        self.base_url = env_base

        if self.mode == 'live':
            self.base_url = self.base_url.replace('paper-api', 'api')
            self.logger.log_event("BROKER_INIT", {"mode": "LIVE", "url": self.base_url}, level="WARNING")
        else:
            self.logger.log_event("BROKER_INIT", {"mode": "PAPER", "url": self.base_url})
        
        self.api = tradeapi.REST(self.api_key, self.secret_key, self.base_url, api_version='v2')
        
    def _api_call(self, func, *args, **kwargs):
        """Helper to invoke Alpaca API calls with rate-limit retries."""
        max_retries = 5
        base_delay = 1.0
        for retries in range(max_retries):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                err_str = str(e).lower()
                if '429' in err_str or 'rate limit' in err_str or 'too many requests' in err_str:
                    delay = base_delay * (2 ** retries)
                    self.logger.log_event("RATE_LIMIT", {"retrying_in": delay, "attempt": retries + 1}, level="WARNING")
                    time.sleep(delay)
                    if retries == max_retries - 1:
                        raise e
                else:
                    raise e
        
    def get_account(self):
        try:
            acc = self._api_call(self.api.get_account)
            return {
                "equity": float(acc.equity),
                "last_equity": float(acc.last_equity),
                "cash": float(acc.cash),
                "status": acc.status,
                "currency": acc.currency,
                "buying_power": float(acc.buying_power)
            }
        except Exception as e:
            self.logger.log_error(f"Alpaca get_account failed: {str(e)}")
            raise e

    def get_positions(self):
        try:
            positions = self._api_call(self.api.list_positions)
            return [{
                "symbol": p.symbol,
                "qty": float(p.qty),
                "side": "long" if float(p.qty) > 0 else "short",
                "current_price": float(p.current_price),
                "market_value": float(p.market_value),
                "unrealized_pl": float(p.unrealized_pl)
            } for p in positions]
        except Exception as e:
            self.logger.log_error(f"Alpaca get_positions failed: {str(e)}")
            return []

    def get_position(self, symbol: str):
        try:
            p = self._api_call(self.api.get_position, symbol)
            return {
                "symbol": p.symbol,
                "qty": float(p.qty),
                "side": "long" if float(p.qty) > 0 else "short",
                "current_price": float(p.current_price),
                "market_value": float(p.market_value),
                "unrealized_pl": float(p.unrealized_pl)
            }
        except Exception:
            return None

    def submit_order(self, symbol: str, qty: float, side: str, order_type: str = 'market', time_in_force: str = 'day'):
        self.logger.log_event("ORDER_SUBMIT_ATTEMPT", {
            "symbol": symbol, "qty": qty, "side": side, "type": order_type
        })
        try:
            order = self._api_call(
                self.api.submit_order,
                symbol=symbol,
                qty=qty,
                side=side,
                type=order_type,
                time_in_force=time_in_force
            )
            self.logger.log_event("ORDER_SUBMITTED", {
                "order_id": order.id, "symbol": symbol, "status": order.status
            })
            return order._raw
        except Exception as e:
            self.logger.log_error(f"Order failed: {str(e)}", {"symbol": symbol, "side": side})
            raise e

    def close_position(self, symbol: str):
        try:
            self._api_call(self.api.close_position, symbol)
            self.logger.log_event("POSITION_CLOSE_SUBMITTED", {"symbol": symbol})
            return True
        except Exception as e:
            self.logger.log_error(f"Close position failed: {str(e)}", {"symbol": symbol})
            return False

    def close_all_positions(self):
        try:
            self._api_call(self.api.close_all_positions, cancel_orders=True)
            self.logger.log_event("PANIC_CLOSE_ALL_SUBMITTED", {}, level="CRITICAL")
            return self.get_positions()
        except Exception as e:
            self.logger.log_error(f"Panic close failed: {str(e)}")
            raise e

    def cancel_all_orders(self):
        try:
            self._api_call(self.api.cancel_all_orders)
            self.logger.log_event("CANCEL_ALL_ORDERS", {})
            return []
        except Exception as e:
            self.logger.log_error(f"Cancel all orders failed: {str(e)}")
            raise e

    def has_open_order(self, symbol: str) -> bool:
        """Check if there are any open orders for a symbol."""
        try:
            orders = self._api_call(self.api.list_orders, status='open')
            return any(order.symbol == symbol for order in orders)
        except Exception as e:
            self.logger.log_error(f"Check open orders failed: {str(e)}", {"symbol": symbol})
            return False

    def get_latest_price(self, symbol: str) -> float:
        """Fetch latest price safely with retries."""
        try:
            trade = self._api_call(self.api.get_latest_trade, symbol)
            return float(trade.price)
        except Exception as e:
            self.logger.log_error(f"Could not fetch price for {symbol}: {str(e)}")
            raise e
