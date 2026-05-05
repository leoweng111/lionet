from __future__ import annotations

import json
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = PROJECT_ROOT / "config"


def _ensure_config_dir() -> None:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)


def _atomic_write_json(path: Path, data: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, sort_keys=True)
    tmp.replace(path)


def load_json_config(file_name: str, default_data: Any) -> Any:
    """Load one config json under project-root config dir; create it with default when missing/invalid."""
    _ensure_config_dir()
    path = CONFIG_DIR / file_name
    if not path.exists():
        _atomic_write_json(path, default_data)
        return default_data

    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        _atomic_write_json(path, default_data)
        return default_data


def save_json_config(file_name: str, data: Any) -> Path:
    """Save one config json under project-root config dir."""
    _ensure_config_dir()
    path = CONFIG_DIR / file_name
    _atomic_write_json(path, data)
    return path

