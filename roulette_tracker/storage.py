"""
storage.py — Sauvegarde et chargement des résultats (CSV + JSON).
"""

import json
import csv
from pathlib import Path
from dataclasses import asdict
from datetime import datetime
from typing import List

from detector import RouletteResult


# ── CSV ───────────────────────────────────────────────────────────────────────

CSV_FIELDS = ["turn", "number", "color", "parity", "dozen", "timestamp", "frame_idx"]


def save_csv(results: List[RouletteResult], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for i, r in enumerate(results, 1):
            row = asdict(r)
            row["turn"] = i
            writer.writerow(row)
    print(f"[Storage] CSV sauvegardé → {path}")
    return path


def load_csv(path: str | Path) -> List[RouletteResult]:
    path = Path(path)
    results = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            results.append(RouletteResult(
                number    = int(row["number"]),
                color     = row["color"],
                parity    = row["parity"],
                dozen     = row["dozen"],
                timestamp = float(row["timestamp"]),
                frame_idx = int(row["frame_idx"]),
            ))
    return results


# ── JSON ──────────────────────────────────────────────────────────────────────

def save_json(results: List[RouletteResult], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at": datetime.now().isoformat(),
        "total_turns":  len(results),
        "results": [
            {"turn": i, **asdict(r)}
            for i, r in enumerate(results, 1)
        ],
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"[Storage] JSON sauvegardé → {path}")
    return path


def load_json(path: str | Path) -> List[RouletteResult]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return [
        RouletteResult(
            number    = r["number"],
            color     = r["color"],
            parity    = r["parity"],
            dozen     = r["dozen"],
            timestamp = r["timestamp"],
            frame_idx = r["frame_idx"],
        )
        for r in data["results"]
    ]


# ── Sauvegarde combinée ───────────────────────────────────────────────────────

def save_all(results: List[RouletteResult], output_dir: str | Path, name: str = "results") -> dict:
    """Sauvegarde en CSV + JSON dans output_dir."""
    output_dir = Path(output_dir)
    csv_path  = save_csv(results,  output_dir / f"{name}.csv")
    json_path = save_json(results, output_dir / f"{name}.json")
    return {"csv": csv_path, "json": json_path}
