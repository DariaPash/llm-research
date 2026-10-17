"""Фиксация гипотез по сохранённому паспорту, без зависимости от Git."""

import hashlib
import json
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def hypothesis_commit(root=ROOT):
    root = Path(root)
    record = json.loads(
        (root / "config/hypotheses_provenance.json").read_text(encoding="utf-8")
    )
    manifest = json.loads(
        (root / "results/run_manifest.json").read_text(encoding="utf-8")
    )
    manifest_digest = hashlib.sha256(
        (root / "results/run_manifest.json").read_bytes()
    ).hexdigest()
    if manifest_digest != record["source_manifest_sha256"]:
        raise ValueError("Исходный паспорт прогонов изменён")
    original = manifest["hypotheses"]
    if (record["commit"], record["commit_time"]) != (
        original["commit"],
        original["commit_time"],
    ):
        raise ValueError("Метаданные гипотез расходятся с исходным паспортом")
    digest = hashlib.sha256((root / "hypotheses.md").read_bytes()).hexdigest()
    if digest != record["submitted_hypotheses_sha256"]:
        raise ValueError("hypotheses.md отличается от сохранённой версии")
    return record["commit"], datetime.fromisoformat(record["commit_time"])
