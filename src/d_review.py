"""Техническое разрешение 0/н/п в d1 без переписывания человеческих листов.

Это аудит конкретного сохранённого среза, не универсальная проверка произвольного
Python. Все 24 программы просмотрены по коду. Хэши не дают перенести эти решения
на новые ответы или оценки без нового аудита.
"""

import hashlib
import json
from collections import Counter
from pathlib import Path

from src import autograde, parse

ROOT = Path(__file__).resolve().parent.parent
COLUMN = "d1_preprocessing_before_split"
SHEET_SHA256 = {
    "sheet_r1.csv": "f8d6abb25d2d7d6c0a90166569d4bf04fe13041d3f6cc652e310197dd5d131c7",
    "sheet_r2.csv": "8ad8757f3469e43e7e5888e1a020349f2cb651f8aec5434780fb3ef249bc1696",
}


def agreement(a, b):
    n = len(a)
    if not n or n != len(b):
        raise ValueError("Нужны непустые парные оценки")
    ca, cb = Counter(a), Counter(b)
    po = sum(x == y for x, y in zip(a, b)) / n
    pe = sum(ca[k] * cb[k] for k in ca.keys() | cb.keys()) / n**2
    return {"n": n, "agreement": po, "kappa": (po - pe) / (1 - pe) if pe != 1 else None}


def decide(task_id, code):
    """Зафиксированное решение по просмотренному коду, с проверкой его признаков."""
    if not code:
        raise ValueError("В проверенном срезе у всех D есть код")
    if task_id == "D6":
        if "train_test_split(" in code or "scaler.fit_transform(X)" not in code:
            raise ValueError("Изменился проверенный протокол кластеризации D6")
        return (
            "н/п",
            "clustering_without_holdout",
            "Кластеризация по условию без train/test; масштабирование всех X допустимо.",
        )
    if task_id == "D2":
        if "PCA(" not in code or not any(
            s in code for s in ["pipeline.fit(X_train", "pca.fit_transform(X_train"]
        ):
            raise ValueError("Изменился проверенный PCA на train")
        return (
            "0",
            "preprocessing_on_train",
            "PCA обучается на train и применяется к test; утечки нет.",
        )
    if task_id == "D1" and "StandardScaler(" in code:
        if "Pipeline(" not in code or "grid_search.fit(X_train, y_train)" not in code:
            raise ValueError("Изменился проверенный Pipeline + CV")
        return (
            "0",
            "preprocessing_on_train",
            "StandardScaler внутри Pipeline, GridSearchCV получает только train.",
        )
    if task_id not in {"D1", "D3", "D4", "D5"}:
        raise ValueError("Неизвестная область технического решения")
    if any(
        s in code
        for s in ["StandardScaler(", "PCA(", "fit_transform(", "MinMaxScaler("]
    ):
        raise ValueError(
            "Найден препроцессинг в случае, проверенном как отсутствие преобразований"
        )
    return (
        "0",
        "no_learned_preprocessing",
        "В коде нет обучаемого препроцессинга: утечки через него нет. «0» и прежнее «н/п» означают одно отсутствие ошибки.",
    )


def review_rows(r1, r2, key):
    pairs = [{r["item_id"]: r for r in rows if r["type"] == "D"} for rows in (r1, r2)]
    if len(pairs[0]) != 24 or pairs[0].keys() != pairs[1].keys():
        raise ValueError("Изменился состав D")
    rows = []
    for item_id in sorted(pairs[0]):
        a, b = (p[item_id] for p in pairs)
        if a["answer_text"] != b["answer_text"]:
            raise ValueError("Проверяющие оценивали разный текст")
        k = key[item_id]
        code = parse.parse_answer(a["answer_text"], "code", "ru").value
        resolved, category, reason = decide(k["task_id"], code)
        values = [a[COLUMN], b[COLUMN]]
        accepted = (
            {"0", "н/п"} if category == "no_learned_preprocessing" else {resolved}
        )
        if any(v not in accepted for v in values):
            raise ValueError(
                f"{item_id}: найдено содержательное расхождение; нельзя автоматически нормировать"
            )
        evidence = "\n".join(
            f"{i}: {line}"
            for i, line in enumerate(code.splitlines(), 1)
            if any(
                s in line
                for s in [
                    "train_test_split(",
                    ".fit(",
                    ".fit_transform(",
                    "Pipeline(",
                    "make_pipeline(",
                ]
            )
        )
        rows.append(
            {
                "item_id": item_id,
                "task_id": k["task_id"],
                "call_id": k["call_id"],
                "original_r1": values[0],
                "original_r2": values[1],
                "technical_d1": resolved,
                "case": category,
                "encoding_disagreement": values[0] != values[1],
                "substantive_disagreement": False,
                "reviewer_kind": "ai_technical_adjudication",
                "code_sha256": hashlib.sha256(code.encode()).hexdigest(),
                "evidence": evidence,
                "reason": reason,
            }
        )
    return rows


def build(root=ROOT):
    root = Path(root)
    paths = [root / "grading" / name for name in SHEET_SHA256]
    for path in paths:
        if hashlib.sha256(path.read_bytes()).hexdigest() != SHEET_SHA256[path.name]:
            raise ValueError(
                "Листы изменились: техническое решение D требует нового аудита"
            )
    key = {
        r["item_id"]: r
        for r in autograde.read_csv(root / "grading/key_DO_NOT_OPEN.csv")
    }
    rows = review_rows(*(autograde.read_csv(p) for p in paths), key)
    output = root / "grading/d1_technical_resolution.csv"
    autograde.write_csv(output, rows, list(rows[0]))
    raw = agreement([r["original_r1"] for r in rows], [r["original_r2"] for r in rows])
    record = {
        "status": "applicability_ambiguity_resolved_technically",
        "reviewer_kind": "ai_technical_adjudication",
        "is_human": False,
        "authority": "Пользователь поручил ассистенту разобраться с D; grading/submission_confirmation.json",
        "raw_agreement": raw,
        "encoding_disagreements": sum(r["encoding_disagreement"] for r in rows),
        "substantive_disagreements": sum(r["substantive_disagreement"] for r in rows),
        "case_counts": dict(sorted(Counter(r["case"] for r in rows).items())),
        "human_scores_modified": False,
        "original_kappa_replaced": False,
        "new_human_rerating_claimed": False,
        "interpretation": "15 несовпадений вызваны кодировкой отсутствующего препроцессинга как 0 или н/п. "
        "Все 24 случая разрешены по коду и условию. Исходная κ=0,1 сохраняется. "
        "Согласие после технического решения не выдаётся за новую независимую каппу людей.",
        "sha256": {
            **{
                str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in paths
            },
            "grading/d1_technical_resolution.csv": hashlib.sha256(
                output.read_bytes()
            ).hexdigest(),
        },
    }
    (root / "grading/d1_technical_resolution.json").write_text(
        json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return record


if __name__ == "__main__":
    print(json.dumps(build(), ensure_ascii=False, indent=2))
