"""Minimal .env loader (no dependency). Reads KEY=VALUE lines from a .env in the
project root and sets them in os.environ if not already present. Values are
never logged by this module."""

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_env(path: str | None = None) -> list[str]:
    """Load .env into os.environ. Returns the list of KEY NAMES loaded (not values)."""
    env_path = Path(path) if path else PROJECT_ROOT / ".env"
    loaded: list[str] = []
    if not env_path.exists():
        return loaded
    for raw in env_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, val = line.split("=", 1)
        key = key.strip()
        val = val.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = val
        loaded.append(key)
    return loaded
