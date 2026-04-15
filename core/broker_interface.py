from abc import ABC, abstractmethod
from typing import Dict, Any, List

class BrokerInterface(ABC):
    @abstractmethod
    def get_account(self) -> Dict[str, Any]:
        """Returns account details (equity, cash, status)."""
        pass

    @abstractmethod
    def get_positions(self) -> List[Dict[str, Any]]:
        """Returns list of open positions."""
        pass

    @abstractmethod
    def get_position(self, symbol: str) -> Dict[str, Any]:
        """Returns specific position or None."""
        pass

    @abstractmethod
    def submit_order(self, symbol: str, qty: float, side: str, order_type: str = 'market', time_in_force: str = 'day') -> Dict[str, Any]:
        """Submits an order. Returns order details."""
        pass

    @abstractmethod
    def close_position(self, symbol: str) -> Dict[str, Any]:
        """Closes a specific position."""
        pass

    @abstractmethod
    def close_all_positions(self) -> List[Dict[str, Any]]:
        """Closes ALL positions (Panic button)."""
        pass

    @abstractmethod
    def cancel_all_orders(self) -> List[Dict[str, Any]]:
        """Cancels all open orders."""
        pass

    @abstractmethod
    def has_open_order(self, symbol: str) -> bool:
        """Check if there are open orders for a symbol."""
        pass
