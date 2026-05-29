from __future__ import annotations

import json
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
PROFILE_CONFIG_PATH = SCRIPT_DIR / "resolution_profile.json"


def _load_profile_config() -> dict:
    with PROFILE_CONFIG_PATH.open("r", encoding="utf-8") as f:
        data = json.load(f)

    active = data.get("active_profile")
    profiles = data.get("profiles")
    if not isinstance(active, str) or not active.strip():
        raise ValueError("resolution_profile.json must define non-empty 'active_profile'.")
    if not isinstance(profiles, dict) or not profiles:
        raise ValueError("resolution_profile.json must define non-empty 'profiles'.")
    if active not in profiles:
        raise ValueError(f"active_profile '{active}' is not present in profiles.")

    profile = profiles[active]
    if not isinstance(profile, dict):
        raise ValueError(f"Profile '{active}' must be a JSON object.")
    param_dir = profile.get("param_dir")
    if not isinstance(param_dir, str) or not param_dir.strip():
        raise ValueError(f"Profile '{active}' must include non-empty 'param_dir'.")

    return data


def get_active_param_dir() -> Path:
    cfg = _load_profile_config()
    active = str(cfg["active_profile"])
    profile = cfg["profiles"][active]
    param_dir = SCRIPT_DIR / str(profile["param_dir"])
    return param_dir.resolve()


def resolve_param_path(path_or_name: str | Path) -> Path:
    path = Path(path_or_name)
    if path.is_absolute():
        return path
    return (get_active_param_dir() / path).resolve()

