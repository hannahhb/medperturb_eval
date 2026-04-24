from __future__ import annotations
import json
import os
import glob
import logging
from threading import Lock
from typing import Any, Dict, List, Optional

import pandas as pd


def setup_logging(log_path: str = "./medperturb_eval.log") -> logging.Logger:
    os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_path), logging.StreamHandler()],
    )
    return logging.getLogger(__name__)


# ---------- detail cache IO ----------

def _detail_path(log_path: str, row_id: str) -> str:
    return os.path.join(log_path, f"detailed_{row_id}.json")


def discover_processed_ids(log_path: str) -> set:
    pattern = os.path.join(log_path, "detailed_*.json")
    return {
        os.path.basename(fp)[len("detailed_"):-len(".json")]
        for fp in glob.glob(pattern)
    }


def save_detail(log_path: str, row_id: str, data: Dict[str, Any], lock: Optional[Lock] = None) -> None:
    os.makedirs(log_path, exist_ok=True)
    path = _detail_path(log_path, row_id)
    if lock:
        with lock:
            with open(path, "w") as f:
                json.dump(data, f, indent=2, default=str)
    else:
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)


def try_load_detail(log_path: str, row_id: str) -> Optional[Dict[str, Any]]:
    path = _detail_path(log_path, row_id)
    if os.path.exists(path):
        try:
            with open(path) as f:
                return json.load(f)
        except Exception:
            return None
    return None


def load_cached_rows(processed_ids: set, log_path: str) -> Dict[str, Dict]:
    cached = {}
    for rid in processed_ids:
        detail = try_load_detail(log_path, rid)
        if detail:
            cached[rid] = detail
    return cached


# ---------- CSV helpers ----------

def save_csv(rows: List[Dict], output_path: str, name: str = "results.csv") -> str:
    if not rows:
        return ""
    df = pd.DataFrame(rows)
    os.makedirs(output_path, exist_ok=True)
    csv_path = os.path.join(output_path, name)
    df.to_csv(csv_path, index=False)
    return csv_path
