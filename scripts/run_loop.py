import time
import sys
import os
import yaml
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from scripts.run_production import run_production
from core.logger import SystemLogger

def main_loop():
    logger = SystemLogger()

    # Load config for poll interval
    try:
        with open(PROJECT_ROOT / 'config' / 'system.yaml', 'r') as f:
            config = yaml.safe_load(f)
        poll_interval = config.get('execution', {}).get('poll_interval', 3600)
    except:
        poll_interval = 3600

    print("\n" + "!" * 50)
    print("  CROSS-SECTIONAL CONTINUOUS TRADING SERVICE")
    print(f"  Heartbeat: {poll_interval} seconds")
    print("!" * 50)

    logger.log_event("CS_SERVICE_START", {"poll_interval": poll_interval})

    try:
        while True:
            current_time = time.strftime("%Y-%m-%d %H:%M:%S")
            print(f"\n[{current_time}] Initiating CS Strategy Cycle...")

            try:
                run_production()
            except Exception as e:
                print(f"[CRITICAL] Loop error: {e}")
                logger.log_error(f"Execution Loop Error: {e}")

            print(f"Sleeping for {poll_interval} seconds...")
            time.sleep(poll_interval)

    except KeyboardInterrupt:
        print("\n[INFO] Continuous Service Stopped by User.")
        logger.log_event("CS_SERVICE_STOPPED", {"reason": "UserInterrupt"})

if __name__ == "__main__":
    main_loop()
