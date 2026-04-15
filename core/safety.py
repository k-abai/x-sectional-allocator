import json
import os
import time
from core.logger import SystemLogger

KILL_FILE = "kill_state.json"

class SafetyManager:
    def __init__(self, config):
        self.config = config.get('safety', {})
        self.logger = SystemLogger()
        self._ensure_kill_file()
    
    def _ensure_kill_file(self):
        if not os.path.exists(KILL_FILE):
            self._write_kill_state(False)
            
    def _write_kill_state(self, is_killed: bool):
        with open(KILL_FILE, 'w') as f:
            json.dump({"kill_switch_active": is_killed, "timestamp": time.time()}, f)
            
    def _read_kill_state(self):
        try:
            with open(KILL_FILE, 'r') as f:
                data = json.load(f)
                return data.get("kill_switch_active", False)
        except Exception:
            self.logger.log_error("Failed to read kill state file. Defaulting to KILLED.")
            return True

    @property
    def is_kill_switch_active(self):
        return self._read_kill_state()

    def activate_kill_switch(self):
        self._write_kill_state(True)
        self.logger.log_event("KILL_SWITCH_ACTIVATED", {}, level="CRITICAL")
        return True

    def deactivate_kill_switch(self, code: str = None):
        if self.config.get('require_2fa', True):
            if not self.verify_2fa(code):
                self.logger.log_event("KILL_SWITCH_DEACTIVATE_FAILED", {"reason": "Bad 2FA"}, level="WARNING")
                return False
                
        self._write_kill_state(False)
        self.logger.log_event("KILL_SWITCH_DEACTIVATED", {}, level="WARNING")
        return True

    def verify_2fa(self, code: str):
        if code == "8888":
            return True
        return False
        
    def check_trade_safety(self, account_equity: float, current_exposure: float, trade_value: float):
        if self.is_kill_switch_active:
            return False, "Kill switch active"
        return True, "OK"
