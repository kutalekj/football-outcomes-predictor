import os
from typing import Optional

def load_env():
    try:
        from dotenv import load_dotenv
    except Exception:
        return
    load_dotenv(override=False)

def env_str(key: str, default: Optional[str] = None) -> Optional[str]:
    return os.getenv(key, default)

def env_int(key: str, default: Optional[int] = None) -> Optional[int]:
    v = os.getenv(key, None)
    return int(v) if v is not None else default

def env_bool(key: str, default: bool = False) -> bool:
    v = os.getenv(key)
    if v is None: return default
    return v.lower() in {"1","true","yes","y","on"}
