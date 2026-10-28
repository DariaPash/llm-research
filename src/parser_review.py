"""Техническая сверка фиксированной выборки; исходные человеческие отметки не меняет.

python -m src.parser_review. Это ИИ-аудит, не новая человеческая проверка G2.
Решения относятся только к сохранённому срезу с указанной SHA256.
"""

import hashlib
import json
from pathlib import Path

from src import autograde, bank, grade, parse

ROOT = Path(__file__).resolve().parent.parent
SOURCE_SHA256 = "ccb54c8c87cfbc845ef7853f1d18cb1b2c3fe1e4fae4431edef79e0f23bc98ae"
POLICY = {
    "reviewer_kind": "ai_technical_audit",
    "is_human": False,
    "label_policy": "Сохранена прежняя мягкая нормализация: «возможен/возможно отказ» → 1. "
    "Это неоднозначный ответ и нарушение точного формата, но не ошибка извлечения по действующему правилу. "
    "Строгая оценка таких ответов как неразобранных здесь не подменяет основной эксперимент.",
    "multiple_numbers_policy": "Сохранено правило первого числа для скалярного ответа (включая A2/F2). "
    "Для S14 извлечение 2.0 из 2;2;0.5 неполно и отмечено отдельно. "
    "Автооценка «неверно» остаётся верной: модель дала числа при недостатке данных.",
    "meaning": "Согласие означает корректность извлечения и автооценки, а не совпадение прогноза с эталоном.",
    "human_judgments_modified": False,
    "model_responses_or_scores_modified": False,
}


def review_rows(rows, tasks):
    if [r["spot_id"] for r in rows] != [f"S{i:02}" for i in range(1, 21)]:
        raise ValueError("Изменился состав выборки: нужен новый технический аудит")
    reviewed = []
    for row in rows:
        task = (
            {"type": "E", "answer_kind": "label", "reference": int(row["reference"])}
            if row["type"] == "E"
            else tasks[row["task_id"]]
        )
        parsed = parse.parse_answer(
            row["response_text"], task["answer_kind"], row["lang"]
        )
        graded = grade.grade_auto(task, parsed)
        if (
            autograde.fmt(parsed.value) != row["parsed_value"]
            or str(graded["correct"]) != row["auto_correct"]
        ):
            raise ValueError(
                f"{row['spot_id']}: текущий парсер/автооценка отличаются от проверенного среза"
            )
        sid = row["spot_id"]
        ambiguous = sid in {"S01", "S02", "S03", "S04"}
        extraction = sid != "S14"
        comment = (
            "Значение извлечено корректно; автооценка соответствует эталону и допуску."
        )
        if ambiguous:
            comment = (
                "По сохранённой мягкой политике фраза о возможном отказе нормализуется в 1. "
                "Автооценка False при эталоне 0 верна. Ответ неоднозначен и не соблюдает точную метку; "
                "это ограничение нормализации, не основание путать неверный прогноз с ошибкой парсера."
            )
        elif sid == "S13":
            comment = (
                "«Нет отказа» правильно извлечено как 0; при эталоне 1 автооценка False верна. "
                "Ошибка модели, не парсера."
            )
        elif sid == "S14":
            comment = (
                "В ответе 2;2;0.5, но извлечено только 2.0: неполное извлечение. "
                "Автооценка False верна независимо от потери чисел: эталон — нельзя определить."
            )
        elif sid in {"S07", "S09", "S10"}:
            comment = (
                "Метка «нет отказа» правильно извлечена эвристикой; parse_ok=False "
                "корректно отмечает отсутствие строки ОТВЕТ. Автооценка True верна."
            )
        reviewed.append(
            {
                **{
                    c: row[c]
                    for c in (
                        "spot_id",
                        "selection",
                        "call_id",
                        "raw_answer_line",
                        "parsed_value",
                        "reference",
                        "auto_correct",
                    )
                },
                "original_human_agrees": row["human_agrees"],
                "original_comment": row["comment"],
                "extraction_agrees": "да" if extraction else "нет",
                "grading_agrees": "да",
                "technical_agrees": "да" if extraction else "нет",
                "ambiguous_label": ambiguous,
                "reviewer_kind": POLICY["reviewer_kind"],
                "technical_comment": comment,
            }
        )
    return reviewed


def build(root=ROOT):
    source = root / "grading/parser_spotcheck.csv"
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    if digest != SOURCE_SHA256:
        raise ValueError(
            "Исходная выборка изменилась: нельзя автоматически переносить решения аудита"
        )
    rows = review_rows(
        autograde.read_csv(source),
        {t["id"]: t for t in bank.load_tasks(root / "data/tasks.jsonl")},
    )
    target = root / "grading/parser_technical_review.csv"
    autograde.write_csv(target, rows, list(rows[0]))
    record = {
        **POLICY,
        "source": "grading/parser_spotcheck.csv",
        "source_sha256": digest,
        "output": "grading/parser_technical_review.csv",
        "output_sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
        "n": len(rows),
        "n_agree": sum(r["technical_agrees"] == "да" for r in rows),
        "n_ambiguous_labels": sum(r["ambiguous_label"] for r in rows),
    }
    (root / "grading/parser_technical_review.json").write_text(
        json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return record


if __name__ == "__main__":
    print(json.dumps(build(), ensure_ascii=False, indent=2))
