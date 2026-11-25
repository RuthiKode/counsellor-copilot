"""Load config.yaml and resolve paths relative to the repo root."""
from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "reports" / "figures"


def load_config(path: Path | str | None = None) -> dict:
    path = Path(path) if path else ROOT / "config.yaml"
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def data_path(cfg: dict | None = None) -> Path:
    cfg = cfg or load_config()
    return ROOT / cfg["data"]["path"]
