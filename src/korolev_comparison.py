"""Offline supplemental five-platform comparison; never calls an LLM or executes D.

Run: python -m src.korolev_comparison
Historical files are read-only. Missing results/reviews are missing, not failures,
zero cost, or evidence of completion. JSON is the machine-readable source; CSV,
Markdown and PNG are derived, each replaced atomically (JSON is published last).
"""

import argparse
import hashlib
import io
import json
import math
import os
import tempfile
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

from src import analysis, autograde, bank, blind, run_bank, typeE, typeE_metrics

ROOT = run_bank.ROOT
MODEL = "korolev-k5"
MODELS = ["gigachat-2-max", "yandexgpt-pro-5.1", "deepseek-v4-pro", "qwen3-8b", MODEL]
AUTO_TYPES = ("A", "B", "F")
FILES = {s: f"{s}_ru.jsonl" for s in ("main", "cot", "followup_doubt")}
FILES["typeE"] = "typeE_llm.jsonl"
MODEL_CONFIG = {
    MODELS[0]: {"provider": "gigachat", "role": "ru1"},
    MODELS[1]: {"provider": "openai_compat", "role": "ru2"},
    MODELS[2]: {"provider": "openai_compat", "role": "big"},
    MODELS[3]: {"provider": "ollama", "role": "local"},
    MODEL: {"provider": "korolev_web", "role": "supplemental"},
}
LIMITATIONS = [
    "Разновременное дополнение: октябрьские ответы четырёх моделей не перезапускаются.",
    "Общая accuracy по 38 задачам не сопоставима, пока C/D не оценены; 0.5 сохраняется.",
    "Главный парный тест: только общие для ВСЕХ пяти моделей бинарные A/B/F (план 26).",
    "Техническая ошибка не является неверным ответом; неполные выборки явно отмечены.",
    "B: автопроверяется только вердикт; обоснование требует отдельной ручной оценки.",
    "Листы не доказывают независимость/человеческое происхождение оценок; AI-маркеры запрещены.",
    "Королёв: стоимость неизвестна, даже если веб-доступ бесплатный; пропуски usage не нули.",
    "Скрытые инструкции, RAG и backend сервиса могут отличаться от plain API других моделей.",
    "E: 200 строк с обогащённым классом отказа; перенос на естественную долю некорректен.",
    "Классические fixed-предсказания пересчитаны офлайн в текущем окружении; исторические агрегаты отдельно.",
    "Pooled H1/H6/H9 — описательные объединения; зависимость моделей на одной задаче ограничивает выводы.",
    "Дополнительный qwen3-8b-think исключён из пяти основных моделей; прежние выводы H6 не меняются.",
    "D: читаются только сохранённые исполнения; анализ не запускает сгенерированный код.",
    "Исторические выводы notebook, исходные summary, гипотезы и PDF не заменяются; новые ячейки notebook читают отдельное дополнение.",
]


def clean(value):
    """Strict JSON: no NaN/Infinity, NumPy scalars or invented zeroes."""
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [clean(v) for v in value]
    if value is None or value is pd.NA:
        return None
    if isinstance(value, np.generic):
        return clean(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def records(frame):
    return clean(frame.to_dict("records"))


def atomic_write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data.encode("utf-8") if isinstance(data, str) else data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def read_jsonl(path):
    # Fail closed on a malformed JSON file, rather than claiming an incomplete
    # read to be a finished run. Runner publishes tables with atomic replace.
    return run_bank.read_jsonl(Path(path))


def unique_rows(rows, stage):
    """First successful result per planned unit; never multiply sample sizes."""
    chosen = {}
    for row in rows:
        if row.get("model_key") not in MODELS or row.get("lang") != "ru":
            continue
        if row.get("experiment") != stage or row.get("repeat", 0) != 0:
            continue
        if stage == "main" and row.get("variant") != "base":
            continue
        key = (row["model_key"], row["task_id"])
        if stage == "typeE":
            key = (row["model_key"], row.get("shots"), row.get("row_idx"))
        old = chosen.get(key)
        if old is None or old.get("error") is not None:
            chosen[key] = row
    return list(chosen.values())


def load_grades(results_dir, tasks, *, supplemental=False):
    path = results_dir / autograde.GRADED_NAME
    inputs = [p for p in autograde.answer_tables(results_dir)]
    dex = results_dir / "d_exec.jsonl"
    if dex.exists():
        inputs.append(dex)
    stale = not path.exists() or any(
        p.stat().st_mtime_ns > path.stat().st_mtime_ns for p in inputs
    )
    if stale and inputs:
        graded, _ = autograde.build(results_dir, tasks)
        if supplemental:
            atomic_write(
                path,
                "".join(
                    json.dumps(clean(r), ensure_ascii=False, allow_nan=False) + "\n"
                    for r in graded
                ),
            )
    else:
        graded = read_jsonl(path)
    # A repeated call_id must not explode joins. Prefer the first record, as D
    # execution is pass@1, not best-of-retries.
    return {(r["call_id"], r["check"]): r for r in graded}


def prepare_worksheets(results_dir, grading_dir, tasks, seed):
    """Never overwrite a started sheet or its corresponding key.

    Pristine blank sheets may grow with a run. Once review starts, report
    uncovered call_ids instead of reshuffling IDs or replacing human work.
    """
    existing = [grading_dir / n for n in [*blind.SHEETS, blind.KEY_NAME]]
    desired = blind.collect(results_dir, ["main"], tasks)
    desired_ids = {r["call_id"] for r in desired if r.get("model_key") == MODEL}
    if not desired_ids:
        return {"status": "not_started", "n_answers": 0, "n_uncovered": 0}
    if any(
        autograde.human_started(p, [*blind.GRADE_COLUMNS, "comment"], "item_id")
        for p in existing[:-1]
    ):
        covered = (
            {r["call_id"] for r in autograde.read_csv(existing[-1])}
            if existing[-1].exists()
            else set()
        )
        return {
            "status": "preserved_existing",
            "n_answers": len(desired_ids),
            "n_uncovered": len(desired_ids - covered),
        }
    sheet, key = blind.build(results_dir, ["main"], tasks, seed)
    extra = blind.name_regex(["Королёв ИИ", "Королев ИИ", "Korolev AI", MODEL])
    for row, item in zip(sheet, key):
        row["answer_text"] = blind.anonymize(row["answer_text"], extra)
        item["answer_sha"] = blind.answer_sha(row["answer_text"])
    for path, rows, columns in [
        (existing[-1], key, blind.KEY_COLUMNS),
        *[(p, sheet, blind.SHEET_COLUMNS) for p in existing[:-1]],
    ]:
        import csv

        stream = io.StringIO(newline="")
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
        atomic_write(path, "\ufeff" + stream.getvalue())
    return {"status": "pending_human", "n_answers": len(key), "n_uncovered": 0}


def answer_frame(rows, graded, manual):
    """Join auto and genuine worksheet values without filling missing C/D."""
    manual_by_call = (
        manual.set_index("call_id").to_dict("index") if not manual.empty else {}
    )
    out = []
    for raw in rows:
        if raw.get("error") is not None:
            continue
        row = dict(raw)
        auto = graded.get((row["call_id"], "answer"))
        dex = graded.get((row["call_id"], "d_exec"), {})
        human = manual_by_call.get(row["call_id"], {})
        for col in ("value", "confidence", "refused", "parse_ok"):
            row[col] = auto.get(col) if auto else row.get(col)
        row["correct"] = (
            float(auto["correct"])
            if auto and auto.get("correct") is not None
            else np.nan
        )
        if row["type"] in ("C", "D"):
            row["correct"] = (
                human.get("correct_manual", np.nan)
                if human.get("complete", False)
                else np.nan
            )
        row["score_0_2"] = human.get("score_mean", np.nan)
        row["manual_complete"] = bool(human.get("complete", False))
        row["any_error_mean"] = human.get("any_error_mean", np.nan)
        for col in blind.CHECKLIST:
            row[f"{col}_mean"] = human.get(f"{col}_mean", np.nan)
        row["pass_at_1"] = dex.get("pass_at_1", np.nan)
        row["metric_ok_auto"] = dex.get("correct", np.nan)
        if row["metric_ok_auto"] is None:
            row["metric_ok_auto"] = np.nan
        row["parse_ok_auto"] = auto.get("parse_ok", np.nan) if auto else np.nan
        row["refusal"] = (
            bool(row.get("refused")) or row.get("value") == "cannot_determine"
        )
        row["abstain"] = row["refusal"] and row.get("answer_kind") != "cannot_determine"
        out.append(row)
    columns = [
        "model_key",
        "task_id",
        "type",
        "call_id",
        "correct",
        "confidence",
        "refusal",
        "abstain",
        "pair_id",
        "task_variant",
        "tokens_in",
        "tokens_out",
        "tokens_in_full",
        "latency_s",
        "first_call_id",
        "pass_at_1",
        "metric_ok_auto",
        "any_error_mean",
        "score_0_2",
        "manual_complete",
        "parse_ok_auto",
        *[f"{c}_mean" for c in blind.CHECKLIST],
    ]
    df = pd.DataFrame(out)
    for col in columns:
        if col not in df:
            df[col] = pd.Series(index=df.index, dtype=object)
    for col in (
        "correct",
        "confidence",
        "tokens_in",
        "tokens_out",
        "tokens_in_full",
        "latency_s",
        "pass_at_1",
        "metric_ok_auto",
        "any_error_mean",
    ):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in ("refusal", "abstain", "manual_complete"):
        df[col] = df[col].astype(bool)
    return df


def common_auto(main, task_ids):
    auto = main[main["type"].isin(AUTO_TYPES) & main["correct"].apply(analysis.binary)]
    sets = [set(auto.loc[auto["model_key"] == m, "task_id"]) for m in MODELS]
    common = set.intersection(set(task_ids), *sets)
    return auto[auto["task_id"].isin(common)].copy(), sorted(common)


def accuracy_table(main, tasks, n_boot, seed, *, by_type=False, expected_ids=None):
    table = (
        analysis.acc_table(
            main, ["model_key", "type"] if by_type else ["model_key"], n_boot, seed
        )
        if len(main)
        else pd.DataFrame()
    )
    lookup = {(r["model_key"], r.get("type")): r for r in records(table)}
    out = []
    for model in MODELS:
        for typ in analysis.BANK_TYPES if by_type else [None]:
            n_expected = (
                len(expected_ids)
                if expected_ids is not None
                else sum(typ is None or t["type"] == typ for t in tasks.values())
            )
            row = lookup.get(
                (model, typ),
                {
                    "model_key": model,
                    **({"type": typ} if typ else {}),
                    "n": 0,
                    "n_graded": 0,
                    "n_ungraded": 0,
                    "n_correct": None,
                    "accuracy": None,
                    "ci_low": None,
                    "ci_high": None,
                    "refusal_rate": None,
                    "n_abstain": 0,
                    "accuracy_answered": None,
                },
            )
            row.update(
                n_expected=n_expected,
                n_missing=n_expected - row["n"],
                complete=n_expected > 0 and row["n_graded"] == n_expected,
            )
            if row["n_graded"] == 0:
                row["n_correct"] = None
            out.append(row)
    return out


def paired_tests(main, models=MODELS):
    wide = analysis.pivot_correct(main) if len(main) else pd.DataFrame()
    rows = []
    for a, b in combinations(models, 2):
        test = (
            analysis.mcnemar_test(wide[a], wide[b])
            if a in wide and b in wide
            else analysis.mcnemar_test([], [])
        )
        rows.append({"model_a": a, "model_b": b, **test})
    return holm(rows)


def holm(rows):
    valid = [
        i
        for i, row in enumerate(rows)
        if row.get("p_value") is not None and np.isfinite(row["p_value"])
    ]
    for row in rows:
        row["p_holm"] = None
    for i, p in zip(
        valid,
        multipletests([rows[i]["p_value"] for i in valid], method="holm")[1]
        if valid
        else [],
    ):
        rows[i]["p_holm"] = float(p)
    return clean(rows)


def stage_coverage(raw, tasks, split):
    out = []
    for model in MODELS:
        main = [r for r in raw["main"] if r["model_key"] == model]
        main_ok = {r["task_id"] for r in main if r.get("error") is None}
        main_complete = set(tasks) <= main_ok
        eligible = {
            r["call_id"]
            for r in main
            if r.get("error") is None
            and r.get("correct") is True
            and r["type"] in AUTO_TYPES
        }
        specs = [
            ("main", set(tasks), None),
            ("cot", {i for i, t in tasks.items() if t["type"] in ("A", "B")}, None),
            ("followup_doubt", eligible if main_complete else None, None),
        ]
        specs += [("typeE", set(split["test"]), k) for k in (0, 5, 20)]
        for stage, expected, shots in specs:
            group = [
                r
                for r in raw[stage]
                if r["model_key"] == model
                and (shots is None or r.get("shots") == shots)
            ]
            field = (
                "row_idx"
                if stage == "typeE"
                else "first_call_id"
                if stage == "followup_doubt"
                else "task_id"
            )
            seen = {r.get(field) for r in group}
            ok = {r.get(field) for r in group if r.get("error") is None}
            good = ok & expected if expected is not None else ok
            if expected == set() and main_complete:
                status = "not_required"
            elif not group:
                status = "not_started"
            else:
                status = (
                    "complete" if expected is not None and expected <= ok else "partial"
                )
            out.append(
                {
                    "model_key": model,
                    "stage": stage,
                    "shots": shots,
                    "expected": len(expected) if expected is not None else None,
                    "received": len(seen),
                    "successful": len(good),
                    "technical_errors": sum(r.get("error") is not None for r in group),
                    "missing": len(expected - ok) if expected is not None else None,
                    "status": status,
                }
            )
    return out


def condition_rows(pairs, n_boot, seed, cluster="task_id"):
    valid = pairs.loc[
        pairs["correct_a"].apply(analysis.binary).astype(bool)
        & pairs["correct_b"].apply(analysis.binary).astype(bool)
    ]
    rows = (
        records(
            analysis.compare_conditions(
                valid, "model_key", n_boot, seed, cluster=cluster
            )
        )
        if len(valid)
        else []
    )
    have = {r["model_key"] for r in rows}
    rows += [empty_condition(model) for model in [*MODELS, "все"] if model not in have]
    return rows


def empty_condition(model):
    return {
        "model_key": model,
        "n_pairs": 0,
        **{
            col: None
            for col in [
                "acc_a",
                "acc_a_ci_low",
                "acc_a_ci_high",
                "acc_b",
                "acc_b_ci_low",
                "acc_b_ci_high",
                "diff_a_minus_b",
                "diff_ci_low",
                "diff_ci_high",
                "mcnemar_p",
            ]
        },
        "only_a_correct": 0,
        "only_b_correct": 0,
    }


def h6_tables(main, cot, n_boot, seed):
    rows, ratios = [], []
    pairs = cot.merge(main, on=["model_key", "task_id"], suffixes=("_cot", "_main"))
    pairs = pairs.rename(
        columns={"correct_cot": "correct_a", "correct_main": "correct_b"}
    )
    for suffix in ("cot", "main"):
        pairs[f"tokens_total_{suffix}"] = (
            pairs[f"tokens_in_full_{suffix}"] + pairs[f"tokens_out_{suffix}"]
        )
    for typ in ("A", "B"):
        p = pairs[pairs["type_cot"] == typ]
        for row in condition_rows(p, n_boot, seed):
            row.update(
                type=typ,
                pool="available_models",
                n_models=p["model_key"].nunique() if row["model_key"] == "все" else 1,
            )
            rows.append(row)
        sets = [set(p.loc[p["model_key"] == m, "task_id"]) for m in MODELS]
        common = set.intersection(*sets)
        cp = p[p["task_id"].isin(common)]
        pooled = (
            records(analysis.compare_conditions(cp, None, n_boot, seed))
            if len(cp)
            else [
                {
                    "group": "все",
                    **{
                        k: v
                        for k, v in empty_condition("все").items()
                        if k != "model_key"
                    },
                }
            ]
        )
        for row in pooled:
            row.update(type=typ, pool="common_all_five", n_models=5 if common else 0)
            rows.append(row)
    for model, g in pairs.groupby("model_key", sort=False):
        rec = {"model_key": model, "n_pairs": len(g)}
        for name, col in [
            ("tokens_out", "tokens_out"),
            ("tokens_total", "tokens_total"),
            ("latency", "latency_s"),
            ("cost", "cost_rub"),
        ]:
            a, b = g[f"{col}_cot"], g[f"{col}_main"]
            known = a.notna() & b.notna()
            rec[f"{name}_n_known_pairs"] = int(known.sum())
            rec[f"{name}_ratio"] = (
                a.mean() / b.mean() if known.all() and len(g) and b.mean() > 0 else None
            )
        ratios.append(rec)
    have = {r["model_key"] for r in ratios}
    ratios += [
        {
            "model_key": m,
            "n_pairs": 0,
            "tokens_out_ratio": None,
            "tokens_total_ratio": None,
            "latency_ratio": None,
            "cost_ratio": None,
        }
        for m in MODELS
        if m not in have
    ]
    return rows, clean(ratios)


def resources(frame, pricing):
    frame = analysis.add_cost(frame, pricing, MODEL_CONFIG)
    # Deliberately override even an accidentally added zero in a future price file.
    frame.loc[frame["model_key"] == MODEL, "cost_rub"] = np.nan
    rows = []
    for model in MODELS:
        g = frame[frame["model_key"] == model]
        rec = {"model_key": model, "n": len(g)}
        for col in ("latency_s", "tokens_in", "tokens_in_full", "tokens_out"):
            rec[f"{col}_n_known"] = int(g[col].notna().sum())
            rec[f"{col}_mean"] = g[col].mean()
        known = len(g) > 0 and g["cost_rub"].notna().all()
        rec["cost_rub_total"] = g["cost_rub"].sum() if known else None
        corr = g[g["correct"].notna()]
        rec["rub_per_correct"] = (
            corr["cost_rub"].sum() / corr["correct"].sum()
            if known and len(corr) and corr["correct"].sum() > 0
            else None
        )
        rec["cost_status"] = "known" if known else "unknown"
        rows.append(rec)
    return frame, clean(rows)


def e_tables(raw, root, split, n_boot, seed):
    rows = typeE_metrics.llm_rows(raw, split["test"], typeE.split_hash(split))
    table = typeE_metrics.llm_table(rows, split)
    table_rows = records(table)
    for rec in table_rows:
        group = [
            r
            for r in rows
            if r["model_key"] == rec["model"]
            and r["shots"] == rec["shots"]
            and r.get("error") is None
        ]
        for col in ("tokens_in", "tokens_out"):
            rec[f"{col}_n_known"] = sum(r.get(col) is not None for r in group)
            if any(r.get(col) is None for r in group):
                rec[f"{col}_sum"] = None
    have = {(r["model"], r["shots"]) for r in table_rows}
    for model in MODELS:
        for k in (0, 5, 20):
            if (model, k) not in have:
                table_rows.append(
                    {
                        "source": "llm",
                        "model": model,
                        "shots": k,
                        "n_eval": 0,
                        "complete": False,
                        "accuracy": None,
                        "f1_failure": None,
                        "balanced_accuracy": None,
                        "coverage": None,
                    }
                )
    pred = []
    for r in rows:
        if r.get("error") is None:
            pred.append(
                {
                    "source": "llm",
                    "model_key": r["model_key"],
                    "shots": r["shots"],
                    "row_idx": r["row_idx"],
                    "y_true": r["y_true"],
                    "pred": r["value"]
                    if typeE_metrics.is_label(r["value"])
                    else 1 - r["y_true"],
                }
            )
    historical = []
    path = root / "results/typeE_classical.csv"
    if path.exists():
        historical = records(typeE_metrics.classical_table(pd.read_csv(path)))
    data_path = root / "data/ai4i2020.csv"
    if data_path.exists():
        X, y = typeE.features(typeE.normalize_columns(pd.read_csv(data_path)))
        for r in rows:
            if r["y_true"] != y[r["row_idx"]]:
                raise ValueError("E: y_true differs from the fixed dataset")
        for k in (5, 20):
            idx = split.get("shots", {}).get(str(k))
            if not idx:
                continue
            for name, estimator in typeE.make_models(seed).items():
                estimator.fit(X.iloc[idx], y[idx])
                p = estimator.predict(X.iloc[split["test"]])
                pred += [
                    {
                        "source": "classical_offline_recomputed",
                        "model_key": name,
                        "shots": k,
                        "row_idx": i,
                        "y_true": int(y[i]),
                        "pred": int(v),
                    }
                    for i, v in zip(split["test"], p)
                ]
    preds = pd.DataFrame(
        pred, columns=["source", "model_key", "shots", "row_idx", "y_true", "pred"]
    )
    ci = records(analysis.typeE_ci_table(preds, n_boot, seed)) if len(preds) else []
    for rec in ci:
        g = preds[
            (preds["model_key"] == rec["model_key"]) & (preds["shots"] == rec["shots"])
        ]
        rec["source"] = g["source"].iloc[0]
        correct = (g["pred"].to_numpy() == g["y_true"].to_numpy()).astype(float)
        rec["accuracy"] = correct.mean()
        rec["accuracy_ci_low"], rec["accuracy_ci_high"] = analysis.bootstrap_stat(
            list(range(len(g))), lambda s: correct[s].mean(), n_boot, seed
        )
        rec["complete"] = len(g) == len(split["test"])
    tests, common_ci = [], []
    for k in (0, 5, 20):
        g = preds[preds["shots"] == k]
        sets = [set(g.loc[g["model_key"] == m, "row_idx"]) for m in MODELS]
        common = set.intersection(*sets)
        common_preds = g[g["model_key"].isin(MODELS) & g["row_idx"].isin(common)]
        common_ci += (
            records(analysis.typeE_ci_table(common_preds, n_boot, seed))
            if len(common_preds)
            else []
        )
        pool = MODELS + sorted(set(g["model_key"]) - set(MODELS))
        family = []
        for a, b in combinations(pool, 2):
            paired = g[g["model_key"] == a].merge(
                g[g["model_key"] == b], on="row_idx", suffixes=("_a", "_b")
            )
            if len(paired) and not (paired["y_true_a"] == paired["y_true_b"]).all():
                raise ValueError("E: inconsistent reference labels across models")
            test = analysis.mcnemar_test(
                (paired["pred_a"] == paired["y_true_a"]).astype(int),
                (paired["pred_b"] == paired["y_true_b"]).astype(int),
            )
            family.append(
                {
                    "shots": k,
                    "model_a": a,
                    "model_b": b,
                    "common_all_five_n": len(common),
                    "scope": "pairwise_common_rows",
                    **test,
                }
            )
        tests += holm(family)
    return {
        "typeE_llm": clean(table_rows),
        "typeE_ci": clean(ci),
        "typeE_common_five_ci": common_ci,
        "typeE_classical_historical": historical,
        "typeE_mcnemar": tests,
    }


def build(root=ROOT, *, n_boot=1000, seed=42, worksheets=True):
    root = Path(root)
    watch = [
        root / p
        for p in (
            "data/tasks.jsonl",
            "data/e_split.json",
            "data/ai4i2020.csv",
            "config/pricing.json",
            "logs/calls.jsonl",
        )
    ]
    watch += [
        root / "results" / sub / name
        for sub in ("", "korolev")
        for name in [*FILES.values(), "d_exec.jsonl"]
    ]

    def digest(path):
        return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None

    initial_hashes = {p: digest(p) for p in watch}
    tasks = {t["id"]: t for t in bank.load_tasks(root / "data/tasks.jsonl")}
    split = json.loads((root / "data/e_split.json").read_text(encoding="utf-8"))
    old, new = root / "results", root / "results/korolev"
    raw, frames, manual_info = {}, {}, {}
    parts = []
    inputs = [root / "data/tasks.jsonl", root / "data/e_split.json"]
    for directory, is_new in [(old, False), (new, True)]:
        grades = load_grades(directory, tasks, supplemental=is_new)
        gd = root / "grading" / "korolev" if is_new else root / "grading"
        if is_new and worksheets:
            manual_info = prepare_worksheets(directory, gd, tasks, seed)
        manual = analysis.load_manual(gd)
        subset = {}
        for stage, name in FILES.items():
            path = directory / name
            inputs.append(path)
            subset[stage] = unique_rows(read_jsonl(path), stage)
            # Only the assigned new model may enter the supplemental namespace.
            subset[stage] = [
                r for r in subset[stage] if (r["model_key"] == MODEL) == is_new
            ]
            if stage == "typeE":
                subset[stage] = typeE_metrics.llm_rows(
                    subset[stage], split["test"], typeE.split_hash(split)
                )
                for r in subset[stage]:
                    auto = grades.get((r["call_id"], "answer"))
                    if auto:
                        r.update(value=auto["value"])
            else:
                frame = answer_frame(subset[stage], grades, manual)
                # New provider reports full input usage; do not inspect any
                # supplemental authentication/session metadata or manifests.
                frame = (
                    frame.assign(tokens_in_full=frame["tokens_in"])
                    if is_new
                    else analysis.add_full_input(
                        frame, root / "logs/calls.jsonl", MODEL_CONFIG
                    )
                )
                frames.setdefault(stage, []).append(frame)
        parts.append(subset)
        inputs += [
            directory / "graded_auto.jsonl",
            directory / "d_exec.jsonl",
            *[gd / n for n in [blind.KEY_NAME, *blind.SHEETS]],
        ]
    for stage in FILES:
        raw[stage] = parts[0][stage] + parts[1][stage]
    main, cot, follow = [
        pd.concat(frames[s], ignore_index=True)
        for s in ("main", "cot", "followup_doubt")
    ]
    pricing_path = root / "config/pricing.json"
    inputs.append(pricing_path)
    pricing = analysis.load_pricing(pricing_path)
    main, costs = resources(main, pricing)
    cot, _ = resources(cot, pricing)
    auto_ids = {i for i, t in tasks.items() if t["type"] in AUTO_TYPES}
    common, common_ids = common_auto(main, auto_ids)
    coverage = stage_coverage(raw, tasks, split)
    new_coverage = [r for r in coverage if r["model_key"] == MODEL]
    generation = (
        "not_started"
        if not any(r["received"] for r in new_coverage)
        else "complete"
        if all(r["status"] in ("complete", "not_required") for r in new_coverage)
        else "partial"
    )
    new_main = main[main["model_key"] == MODEL]
    review = new_main[new_main["type"].isin(blind.TYPES)]
    pending = review[~review["manual_complete"].astype(bool)]
    d = new_main[new_main["type"] == "D"]
    execution_complete = (
        len(d) == sum(t["type"] == "D" for t in tasks.values())
        and d["pass_at_1"].notna().all()
    )
    review_complete = (
        len(review) == sum(t["type"] in blind.TYPES for t in tasks.values())
        and pending.empty
    )
    status = (
        generation
        if generation != "complete"
        else "finished"
        if review_complete and execution_complete
        else "pending_human_or_execution"
    )
    expected_auto = sum(t["type"] in AUTO_TYPES for t in tasks.values())
    auto_complete = (
        int(new_main[new_main["type"].isin(AUTO_TYPES)]["correct"].notna().sum())
        == expected_auto
    )
    if status == "finished" and not auto_complete:
        status = "pending_automatic_grading"
    h1 = analysis.h1_pairs(main)
    h1_rows = condition_rows(h1, n_boot, seed, "pair_id")
    h1_common_ids = set.intersection(
        *[set(h1.loc[h1["model_key"] == m, "pair_id"]) for m in MODELS]
    )
    for rec in h1_rows:
        rec["pool"] = "available_models"
        rec["n_models"] = (
            int(h1["model_key"].nunique())
            if rec["model_key"] == "все"
            else int(rec["n_pairs"] > 0)
        )
    h6, h6_ratios = h6_tables(main, cot, n_boot, seed)
    # Validate H9 linkage on model AND task in addition to the opaque call_id.
    ids = {(r["call_id"], r["model_key"], r["task_id"]) for r in records(main)}
    valid_follow = (
        follow[
            [
                ((r["first_call_id"], r["model_key"], r["task_id"]) in ids)
                for _, r in follow.iterrows()
            ]
        ]
        if len(follow)
        else follow
    )
    flips = records(analysis.h9_table(main, valid_follow)) if len(valid_follow) else []
    have_flips = {r["model_key"] for r in flips}
    for model in [*MODELS, "все"]:
        if model not in have_flips:
            flips.append(
                {
                    "model_key": model,
                    "n_first_correct": 0,
                    "n_flipped": 0,
                    "flip_rate": None,
                    "ci_low": None,
                    "ci_high": None,
                }
            )
    for rec in flips:
        eligible = main[main["type"].isin(AUTO_TYPES) & (main["correct"] == 1)]
        if rec["model_key"] != "все":
            eligible = eligible[eligible["model_key"] == rec["model_key"]]
        rec["n_eligible_main"] = len(eligible)
        rec["n_missing_eligible"] = max(0, len(eligible) - rec["n_first_correct"])
    calibration = main[main["type"].isin(AUTO_TYPES)].copy()
    calibration.loc[~calibration["confidence"].between(0, 100), "confidence"] = np.nan
    common.loc[~common["confidence"].between(0, 100), "confidence"] = np.nan
    reliability = []
    for model, group in calibration.groupby("model_key", sort=False):
        ok = group[
            group["confidence"].notna() & group["correct"].apply(analysis.binary)
        ]
        _, bins = analysis.ece(
            ok["confidence"].to_numpy(float), ok["correct"].to_numpy(float)
        )
        reliability += [{"model_key": model, **r} for r in records(bins)]
    dt = analysis.typeD_table(main, n_boot, seed)
    for idx, rec in dt.iterrows():
        g = main[
            (main["type"] == "D")
            & (
                (main["model_key"] == rec["model_key"])
                if rec["model_key"] != "все модели"
                else True
            )
        ]
        dt.loc[idx, "n_executed"] = int(g["pass_at_1"].notna().sum())
        dt.loc[idx, "metric_ok_auto"] = g["metric_ok_auto"].mean()
        lo, hi = analysis.cluster_bootstrap(
            g, "metric_ok_auto", n_boot=n_boot, seed=seed
        )
        dt.loc[idx, "metric_ok_auto_ci_low"], dt.loc[idx, "metric_ok_auto_ci_high"] = (
            lo,
            hi,
        )
    binary_main = main[main["correct"].apply(analysis.binary).astype(bool)]
    full_common_ids = set.intersection(
        *[
            set(binary_main.loc[binary_main["model_key"] == m, "task_id"])
            for m in MODELS
        ]
    )
    d_rows = records(dt)
    have_d = {r["model_key"] for r in d_rows}
    d_rows += [
        {
            "model_key": m,
            "n_programs": 0,
            "n_executed": 0,
            "pass_at_1": None,
            "correct": None,
            "metric_ok_auto": None,
        }
        for m in MODELS
        if m not in have_d
    ]
    for rec in d_rows:
        rec["n_expected"] = sum(t["type"] == "D" for t in tasks.values()) * (
            len(MODELS) if rec["model_key"] == "все модели" else 1
        )
        rec["n_execution_missing"] = rec["n_expected"] - rec.get("n_executed", 0)
    refusal_rows, manual_scores = [], []
    for model in MODELS:
        for typ in analysis.BANK_TYPES:
            g = main[(main["model_key"] == model) & (main["type"] == typ)]
            refusal_rows.append(
                {
                    "model_key": model,
                    "type": typ,
                    "n": len(g),
                    "n_refusal_or_cannot_determine": int(g["refusal"].sum()),
                    "n_abstain_on_answerable": int(g["abstain"].sum()),
                    "refusal_rate": g["refusal"].mean() if len(g) else None,
                }
            )
            if typ in ("B", "C"):
                manual_scores.append(
                    {
                        "model_key": model,
                        "type": typ,
                        "n_answers": len(g),
                        "n_both_reviewed": int(g["score_0_2"].notna().sum()),
                        "mean_score_0_2": g["score_0_2"].mean(),
                    }
                )
    tables = {
        "main_accuracy": accuracy_table(main, tasks, n_boot, seed),
        "main_by_type": accuracy_table(main, tasks, n_boot, seed, by_type=True),
        "common_auto_accuracy": accuracy_table(
            common, tasks, n_boot, seed, expected_ids=common_ids
        ),
        "mcnemar_common_auto": paired_tests(common),
        "mcnemar_main_common_binary": paired_tests(
            binary_main[binary_main["task_id"].isin(full_common_ids)]
        ),
        "h1": h1_rows,
        "h1_common_five": condition_rows(
            h1[h1["pair_id"].isin(h1_common_ids)], n_boot, seed, "pair_id"
        ),
        "h6": h6,
        "h6_resources": h6_ratios,
        "h9": flips,
        "calibration": records(analysis.calibration_table(calibration, n_boot, seed)),
        "calibration_common_auto": records(
            analysis.calibration_table(common, n_boot, seed)
        ),
        "calibration_bins": reliability,
        "refusals": refusal_rows,
        "manual_scores": clean(manual_scores),
        "resources_main": costs,
        "typeD": d_rows,
    }
    for stage, frame in [("cot", cot), ("followup_doubt", follow)]:
        _, tables[f"resources_{stage}"] = resources(frame, pricing)
    tables.update(e_tables(raw["typeE"], root, split, n_boot, seed))
    provenance = {}
    for path in inputs:
        if path.exists():
            provenance[str(path.relative_to(root))] = hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
    changed = [
        str(p.relative_to(root)) for p in watch if initial_hashes[p] != digest(p)
    ]
    if changed:
        status = "inputs_changed_rerun"
    runner_path = new / "run_manifest.json"
    runner_record = json.loads(runner_path.read_text()) if runner_path.exists() else {}
    runner_state = {
        key: runner_record.get(key)
        for key in ("status", "updated_utc", "resume_not_before_utc")
    }
    return clean(
        {
            "schema_version": 1,
            "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "status": status,
            "generation_status": generation,
            "runner": runner_state,
            "snapshot_stable": not changed,
            "inputs_changed_during_analysis": changed,
            "models": MODELS,
            "new_model": MODEL,
            "analysis_config": {
                "bootstrap_n": n_boot,
                "seed": seed,
                "ece_bins": analysis.ECE_BINS,
                "lang": "ru",
                "auto_types": AUTO_TYPES,
            },
            "scope": {
                "main_expected_per_model": len(tasks),
                "auto_expected_per_model": len(auto_ids),
                "common_auto_n": len(common_ids),
                "common_auto_task_ids": common_ids,
                "common_main_binary_n": len(full_common_ids),
                "typeE_expected_per_shots": len(split["test"]),
                "split_hash": typeE.split_hash(split),
            },
            "manual": {
                **manual_info,
                "n_received_BCD": len(review),
                "n_pending_BCD": len(pending),
                "reviews_complete": review_complete,
                "execution_complete": execution_complete,
                "automatic_grading_complete": auto_complete,
                "pending_types": pending["type"].value_counts().to_dict(),
            },
            "coverage": coverage,
            "tables": tables,
            "input_sha256": provenance,
            "limitations": LIMITATIONS,
        }
    )


def markdown_table(rows):
    if not rows:
        return "Нет данных.\n"
    columns = list(dict.fromkeys(k for row in rows for k in row))

    def fmt(value):
        if value is None:
            return "—"
        if isinstance(value, float):
            return f"{value:.5g}"
        return str(value).replace("|", "\\|").replace("\n", " ")

    return (
        "\n".join(
            [
                "| " + " | ".join(columns) + " |",
                "|" + "---|" * len(columns),
                *[
                    "| " + " | ".join(fmt(row.get(c)) for c in columns) + " |"
                    for row in rows
                ],
            ]
        )
        + "\n"
    )


def report_markdown(result):
    text = [
        "# Королёв ИИ: реальное дополнение к сравнению пяти моделей",
        "",
        f"Срез: {result['generated_utc']}. **Статус: `{result['status']}`; генерация: `{result['generation_status']}`.**",
        (
            "**Пауза сервиса после HTTP 429.** Повторные обращения разрешены не раньше "
            f"{result.get('runner', {}).get('resume_not_before_utc')}. Полученные ответы не повторяются."
            if result.get("runner", {}).get("status") == "paused_error"
            and result.get("runner", {}).get("resume_not_before_utc")
            else ""
        ),
        "",
        "Это офлайн-анализ сохранённых ответов, не запуск LLM. Пустые клетки — отсутствующие данные. `complete` в строке метрики относится только к её покрытию, а не ко всему эксперименту.",
        f"Стабильность входного среза: {result['snapshot_stable']}. Если данные изменились во время анализа (`inputs_changed_rerun`), повторите офлайн-анализ после завершения runner; это не ошибка модели.",
        "",
        f"Общий бинарный A/B/F-поднабор всех пяти: **{result['scope']['common_auto_n']} / {result['scope']['auto_expected_per_model']}**. До ручной оценки C/D общую accuracy по 38 задачам нельзя использовать как рейтинг пяти моделей. Стоимость Королёва неизвестна.",
        "",
        "## Реальное покрытие",
        markdown_table(result["coverage"]),
        "## Ручная проверка и исполнения D",
        f"Получено B/C/D: {result['manual']['n_received_BCD']}; ожидают ручной проверки: {result['manual']['n_pending_BCD']}. Ручная проверка завершена: {result['manual']['reviews_complete']}; исполнения D полны: {result['manual']['execution_complete']}. Начатые листы и их ключ не заменяются; незаполненные могут расширяться. Новые ответы вне начатого ключа требуют отдельного дополнения листов.",
        "",
    ]
    titles = {
        "main_accuracy": "Описательная точность на доступных оценках (не рейтинг 38 задач)",
        "main_by_type": "Точность по типам",
        "common_auto_accuracy": "Сопоставимая точность: общий A/B/F всех пяти",
        "mcnemar_common_auto": "Все 10 пар: Макнемар + Холм на общем A/B/F",
        "mcnemar_main_common_binary": "Макнемар по всем общим бинарным оценкам main (0.5 исключены)",
        "h1": "H1: оригиналы против изменённых условий",
        "h1_common_five": "H1: только полные пары, общие для всех пяти моделей",
        "h6": "H6: CoT против main, A/B, модели и описательное объединение",
        "h6_resources": "H6: ресурсы на совпадающих парах",
        "h9": "H9: смена правильного ответа",
        "calibration": "Калибровка/ECE на доступных A/B/F",
        "calibration_common_auto": "Калибровка на общем A/B/F",
        "calibration_bins": "Корзины надёжности (пустые корзины не нулевая точность)",
        "refusals": "Отказы и корректное признание недостатка данных",
        "manual_scores": "Ручные баллы B/C; обоснование B не заменяется автооценкой вердикта",
        "resources_main": "Ресурсы main; неизвестная стоимость не равна нулю",
        "typeD": "D: сохранённые исполнения и отдельные ручные оценки",
        "typeE_llm": "E: покрытие и метрики LLM, 0/5/20",
        "typeE_ci": "E: F1/BA/accuracy с 95% CI и офлайн-классические модели",
        "typeE_common_five_ci": "E: общий поднабор пяти LLM",
        "typeE_classical_historical": "E: исторические агрегаты классических моделей (SD не CI)",
        "typeE_mcnemar": "E: парные тесты на совпадающих строках, Холм отдельно по shots",
    }
    for name, rows in result["tables"].items():
        text += [f"## {titles.get(name, name)}", markdown_table(rows)]
    text += [
        "## Ограничения и интерпретация",
        *[f"- {s}" for s in result["limitations"]],
        "",
        "Исторические таблицы и вердикты H1/H6/H9 не изменены. Pooled-строки этого дополнения не заменяют исходные выводы. CI — перцентильный бутстреп по задачам (E — по строкам); H9 — Клоппер–Пирсон. Макнемар исключает 0.5 и пропуски; средняя точность сохраняет 0.5.",
        "",
        "Машиночитаемые данные: [comparison.json](../results/korolev/comparison.json). CSV находятся в той же папке; рисунки — `figures/korolev/`. PDF не пересобирались.",
        "",
    ]
    return "\n".join(text)


def figures(result, folder):
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    def save(fig, name):
        stream = io.BytesIO()
        fig.savefig(stream, format="png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        atomic_write(folder / name, stream.getvalue())

    fig, ax = plt.subplots(figsize=(9, 4))
    for i, row in enumerate(result["tables"]["common_auto_accuracy"]):
        if row["accuracy"] is not None:
            ax.errorbar(
                i,
                row["accuracy"],
                yerr=[
                    [row["accuracy"] - row["ci_low"]],
                    [row["ci_high"] - row["accuracy"]],
                ],
                fmt="o",
                capsize=4,
            )
    ax.set(
        xticks=range(5),
        xticklabels=MODELS,
        ylim=(0, 1.05),
        ylabel="Accuracy, 95% CI",
        title=f"Common A/B/F: n={result['scope']['common_auto_n']} · {result['status']}",
    )
    ax.tick_params(axis="x", rotation=15)
    save(fig, "common_accuracy.png")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for model in dict.fromkeys(r["model_key"] for r in result["tables"]["typeE_ci"]):
        rows = sorted(
            [r for r in result["tables"]["typeE_ci"] if r["model_key"] == model],
            key=lambda r: r["shots"],
        )
        for ax, col in zip(axes, ["f1_failure", "balanced_accuracy"]):
            good = [r for r in rows if r[col] is not None]
            if good:
                ax.plot(
                    [r["shots"] for r in good],
                    [r[col] for r in good],
                    "o-",
                    label=model,
                )
            ax.set(xlabel="Shots", ylabel=col, ylim=(0, 1.05), xticks=[0, 5, 20])
    if axes[0].lines:
        axes[0].legend(fontsize=7)
    fig.suptitle(f"Available rows (coverage in CSV) · {result['status']}")
    save(fig, "typeE.png")
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot([0, 1], [0, 1], "--", color="grey")
    for row in result["tables"]["calibration"]:
        bins = [
            b
            for b in result["tables"]["calibration_bins"]
            if b["model_key"] == row["model_key"] and b["n"] > 0
        ]
        if bins:
            ax.plot(
                [b["mean_confidence"] for b in bins],
                [b["accuracy"] for b in bins],
                "o-",
                label=f"{row['model_key']} (ECE={row['ece']:.3f})",
            )
    ax.set(
        xlim=(0, 1.05),
        ylim=(0, 1.05),
        xlabel="Mean confidence in bin",
        ylabel="Accuracy",
        title=f"A/B/F calibration · {result['status']}",
    )
    if len(ax.lines) > 1:
        ax.legend(fontsize=7)
    save(fig, "calibration.png")


def write_outputs(result, root=ROOT, *, plots=True):
    root = Path(root)
    output = root / "results/korolev"
    for name, rows in result["tables"].items():
        atomic_write(output / f"{name}.csv", pd.DataFrame(rows).to_csv(index=False))
    atomic_write(
        output / "coverage.csv", pd.DataFrame(result["coverage"]).to_csv(index=False)
    )
    if plots:
        figures(result, root / "report/figures/korolev")
    atomic_write(root / "report/korolev_comparison.md", report_markdown(result))
    atomic_write(
        output / "comparison.json",
        json.dumps(clean(result), ensure_ascii=False, indent=2, allow_nan=False) + "\n",
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bootstrap-n", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--no-worksheets", action="store_true")
    args = parser.parse_args(argv)
    if args.bootstrap_n < 1:
        parser.error("bootstrap-n must be positive")
    result = build(
        n_boot=args.bootstrap_n, seed=args.seed, worksheets=not args.no_worksheets
    )
    write_outputs(result, plots=not args.no_plots)
    print(
        f"{result['status']}: generation={result['generation_status']}; common_auto={result['scope']['common_auto_n']}/{result['scope']['auto_expected_per_model']}"
    )
    print("results/korolev/comparison.json; report/korolev_comparison.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
