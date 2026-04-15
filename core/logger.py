import logging
import json
import os
from datetime import datetime
import threading

class JsonFormatter(logging.Formatter):
    def format(self, record):
        log_record = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "event": record.msg if isinstance(record.msg, str) else record.msg.get('event', 'UNKNOWN'),
            "message": record.getMessage()
        }
        # Merge extra fields if msg is dict
        if isinstance(record.msg, dict):
            log_record.update(record.msg)
            
        return json.dumps(log_record)

class SystemLogger:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(SystemLogger, cls).__new__(cls)
                cls._instance._setup()
            return cls._instance

    def _setup(self):
        self.logger = logging.getLogger("SystemLogger")
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False
        
        # Directories
        self.log_dir = "logs"
        self.trade_dir = os.path.join(self.log_dir, "trades")
        self.daily_dir = os.path.join(self.log_dir, "daily")
        
        if not os.path.exists(os.path.join(self.log_dir, "errors")):
            os.makedirs(os.path.join(self.log_dir, "errors"))
            
        for d in [self.log_dir, self.trade_dir, self.daily_dir]:
            os.makedirs(d, exist_ok=True)
            
        today = datetime.now().strftime("%Y%m%d")
        
        # Operations Log
        op_handler = logging.FileHandler(os.path.join(self.log_dir, f"ops_{today}.json"))
        op_handler.setFormatter(JsonFormatter())
        self.logger.addHandler(op_handler)
        
        # Error Log
        err_handler = logging.FileHandler(os.path.join(self.log_dir, "errors", f"errors_{today}.json"))
        err_handler.setLevel(logging.ERROR)
        err_handler.setFormatter(JsonFormatter())
        self.logger.addHandler(err_handler)
        
    def log_event(self, event_type: str, data: dict = None, level: str = "INFO"):
        if data is None:
            data = {}
            
        payload = {
            "event": event_type,
            **data
        }
        
        if level == "INFO":
            self.logger.info(payload)
        elif level == "WARNING":
            self.logger.warning(payload)
        elif level == "ERROR":
            self.logger.error(payload)
        elif level == "CRITICAL":
            self.logger.critical(payload)
            
    def log_trade(self, trade_data: dict):
        self.log_event("TRADE_EXECUTED", trade_data)

    def log_error(self, error_msg: str, context: dict = None):
        if context is None: context = {}
        self.log_event("ERROR", {"message": error_msg, **context}, level="ERROR")
