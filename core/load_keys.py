from dotenv import load_dotenv
from pathlib import Path
import os

def load_keys():
    # Always load .env from project root
    load_dotenv(override=True)

    api_key = os.getenv("APCA_API_KEY_ID")
    secret_key = os.getenv("APCA_API_SECRET_KEY")
    base_url = os.getenv("APCA_API_BASE_URL")   

    if not api_key or not secret_key:
        raise RuntimeError("Alpaca API keys not found. Check .env file.")

    # Normalize base_url: remove trailing slashes and any duplicate /v2 segments
    if base_url:
        base_url = base_url.rstrip('/')
        while base_url.lower().endswith('/v2'):
            base_url = base_url[:-3]
        # do not re-append '/v2'; REST client will handle versioning via api_version

    return {
        "api_key": api_key,
        "secret_key": secret_key,
        "base_url": base_url,
    }
