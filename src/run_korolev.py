"""Реальное дополнение ЛР3: main, CoT, H9 и 600 классификаций одной модели.

python -m src.run_korolev --estimate не обращается к сервису.
Реквизиты читаются из окружения или getpass, но не сохраняются. Все ответы
сразу фиксируются; повторный запуск использует первый успешный ответ из кэша.
Исторические таблицы и паспорт четырёх моделей не перезаписываются.
"""

import argparse
import getpass
import hashlib
import json
import os
import platform
import sys
import time
from datetime import datetime, timedelta, timezone
from importlib import metadata
from pathlib import Path

from src import bank, llm, prompts, run_bank, run_classifier, run_extra, typeE

ROOT = llm.ROOT
OUT = ROOT / "results" / "korolev"
LOG = ROOT / "logs" / "korolev_calls.jsonl"
MANIFEST = OUT / "run_manifest.json"
MODEL = "korolev-k5"
STAGES = ("main", "cot", "followup_doubt", "typeE")


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temp.replace(path)


def stage_path(stage):
    return OUT / ("typeE_llm.jsonl" if stage == "typeE" else f"{stage}_ru.jsonl")


def followup_jobs(tasks, model, temperature, cfg):
    """Та же H9: второй ход только после правильного main на A/B/F."""
    by_id = {t["id"]: t for t in tasks}
    types = cfg["extra_runs"]["followup_doubt"]["types"]
    jobs = []
    for row in run_bank.read_jsonl(stage_path("main")):
        if (
            row["model_key"] != model
            or row["error"] is not None
            or row["correct"] is not True
            or row["variant"] != "base"
        ):
            continue
        task = by_id.get(row["task_id"])
        if task is None or (types and task["type"] not in types):
            continue
        jobs.append(
            run_bank.make_job(
                task,
                model,
                "followup_doubt",
                "ru",
                "followup_doubt",
                temperature,
                history=prompts.followup_history(task, "ru", row["response_text"]),
                extra={
                    "first_call_id": row["call_id"],
                    "first_value": row["value"],
                    "first_correct": row["correct"],
                },
            )
        )
    return jobs


def build_jobs(stage, tasks, model, cfg, df, split):
    temperature = cfg.get("temperature_main", 0)
    if stage == "main":
        return [
            run_bank.make_job(t, model, "main", "ru", "base", temperature)
            for t in tasks
        ]
    if stage == "cot":
        return run_extra.jobs_simple(
            tasks,
            [model],
            "cot",
            "ru",
            ["cot"],
            temperature,
            cfg["extra_runs"]["cot"]["types"],
        )
    if stage == "followup_doubt":
        return followup_jobs(tasks, model, temperature, cfg)
    return run_classifier.build_jobs(
        df,
        split,
        [model],
        cfg["typeE"]["shots"],
        split["test"],
        "ru",
        "typeE",
        temperature,
    )


def input_hashes():
    names = [
        "data/tasks.jsonl",
        "data/ai4i2020.csv",
        "data/e_split.json",
        "hypotheses.md",
        "config/experiment.json",
        "src/prompts.py",
    ]
    return {name: sha256(ROOT / name) for name in names}


def quota_resume_after(log_path, call_id):
    """Сохраняем серверный Retry-After вне сессии, чтобы рестарт не обходил паузу."""
    if not call_id or not Path(log_path).exists():
        return None
    with Path(log_path).open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            rec = json.loads(line)
            if rec.get("call_id") != call_id or not str(
                rec.get("error", "")
            ).startswith("HTTP 429"):
                continue
            delay = rec.get("retry_after_s")
            if delay is None:
                delay = 60.0
            # Пауза отсчитывается от получения ответа, не от начала запроса.
            received = datetime.fromisoformat(rec["ts_utc"]) + timedelta(
                seconds=rec.get("latency_s") or 0
            )
            return (received + timedelta(seconds=float(delay))).isoformat()
    return None


def remaining_quota_seconds(record, current=None):
    until = record.get("resume_not_before_utc")
    if not until:
        return 0.0
    return max(
        0.0,
        (
            datetime.fromisoformat(until) - (current or datetime.now(timezone.utc))
        ).total_seconds(),
    )


def initial_manifest(model, cfg, split):
    packages = {}
    for name in ("httpx", "numpy", "pandas", "scikit-learn", "scipy", "statsmodels"):
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            packages[name] = None
    model_cfg = llm.get_model(model)
    return {
        "created_utc": now(),
        "status": "prepared",
        "model_key": model,
        "model_requested": model_cfg["model"],
        "provider": model_cfg["provider"],
        "base_url": model_cfg["base_url"],
        "params": model_cfg.get("params", {}),
        "historical_comparison": True,
        "note": "Новый реальный прогон после исходного октябрьского эксперимента. "
        "Веб-протокол и скрытые инструкции сервиса могут отличаться от plain API.",
        "log": run_bank.show(LOG),
        "output_dir": run_bank.show(OUT),
        "prompt_version": prompts.PROMPT_VERSION,
        "prompt_hash": prompts.prompt_hash(),
        "temperature_requested": cfg.get("temperature_main", 0),
        "max_tokens": None,
        "typeE_split_hash": typeE.split_hash(split),
        "inputs_sha256": input_hashes(),
        "plan": {"main": 38, "cot": 16, "followup_doubt_max": 26, "typeE": 600},
        "python": sys.version,
        "platform": platform.platform(),
        "packages": packages,
        "historical_files_sha256": {
            name: sha256(ROOT / name)
            for name in (
                "logs/calls.jsonl",
                "results/run_manifest.json",
                "results/summary_metrics.json",
                "results/main_ru.jsonl",
                "results/cot_ru.jsonl",
                "results/followup_doubt_ru.jsonl",
                "results/typeE_llm.jsonl",
                "grading/sheet_r1.csv",
                "grading/sheet_r2.csv",
            )
        },
        "stages": {},
        "manual_grading": "B reasoning, C and D checklist await human review",
        "secrets_stored": False,
    }


def save_row(path, row, key):
    """Только производные таблицы нового прогона; журнал остаётся append-only."""
    rows = {key(r): r for r in run_bank.read_jsonl(path)}
    rows[key(row)] = row
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".jsonl.tmp")
    run_bank.write_jsonl(temp, list(rows.values()))
    temp.replace(path)


def validate(cfg, tasks, split, df):
    problems = run_bank.eval_problems("main", bank.BANK_PATH)
    if problems:
        raise ValueError("Проверка исходного протокола: " + "; ".join(problems))
    typeE.check_split(split, typeE.features(df)[1])
    original = json.loads((ROOT / "results/run_manifest.json").read_text())
    if typeE.split_hash(split) != original["typeE_split_hash"]:
        raise ValueError("Разбиение E не совпадает с историческим прогоном")
    if (
        len(tasks) != 38
        or len(split["test"]) != 200
        or cfg["typeE"]["shots"] != [0, 5, 20]
    ):
        raise ValueError("Банк/объём E изменён: нужен отдельный протокол")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--stages", default=",".join(STAGES))
    ap.add_argument("--estimate", action="store_true")
    ap.add_argument(
        "--delay", type=float, default=1.0, help="пауза между новыми запросами, с"
    )
    ap.add_argument("--max-new-calls", type=int, default=None)
    args = ap.parse_args(argv)
    stages = run_bank.parse_list(args.stages) or []
    if not stages or any(s not in STAGES for s in stages):
        ap.error("Неизвестные этапы")
    if args.delay < 0 or (args.max_new_calls is not None and args.max_new_calls < 1):
        ap.error("Пауза неотрицательна, max-new-calls >= 1")
    cfg = run_bank.load_experiment()
    tasks = bank.load_tasks()
    df, split = typeE.load_data(), typeE.load_split()
    validate(cfg, tasks, split, df)
    if args.estimate:
        for stage in stages:
            jobs = build_jobs(stage, tasks, args.model, cfg, df, split)
            text = str(len(jobs))
            if stage == "followup_doubt" and not stage_path("main").exists():
                text = "до 26 (только правильные A/B/F основного прогона)"
            print(f"{stage}: {text}")
        print("API не вызывался. Полный план: 654 + до 26 follow-up, без smoke.")
        return 0

    model_cfg = llm.get_model(args.model)
    if model_cfg["provider"] != "korolev_web":
        raise ValueError("Этот runner предназначен только для реального korolev_web")
    if MANIFEST.exists():
        record = json.loads(MANIFEST.read_text())
        if (
            record["inputs_sha256"] != input_hashes()
            or record["model_key"] != args.model
        ):
            raise ValueError("Входы или модель изменились: не смешивать разные прогоны")
        if (
            record["params"] != model_cfg.get("params", {})
            or record["base_url"] != model_cfg["base_url"]
        ):
            raise ValueError(
                "Параметры сервиса изменились: нужен отдельный ключ и журнал"
            )
    else:
        record = initial_manifest(args.model, cfg, split)
    # Для уже прерванного старой версией runner прогона восстанавливаем паузу из журнала.
    if record.get("status") == "paused_error" and not record.get(
        "resume_not_before_utc"
    ):
        for stage in record.get("stages", {}).values():
            deadline = quota_resume_after(LOG, stage.get("last_call_id"))
            if deadline:
                record["resume_not_before_utc"] = deadline
                atomic_json(MANIFEST, record)
    if remaining_quota_seconds(record) > 0:
        print(
            f"Пауза сервера: запросы разрешены не раньше {record['resume_not_before_utc']}. "
            "Ни вход, ни генерация не выполнялись.",
            flush=True,
        )
        return 4
    # При CLI-входе секреты вводятся без эха; для автоматического запуска — временное окружение.
    for name, prompt in (
        ("KOROLEV_LOGIN", "SSO login: "),
        ("KOROLEV_PASSWORD", "SSO password: "),
    ):
        if not os.getenv(name):
            if not sys.stdin.isatty():
                raise ValueError(
                    f"Нет {name}: передайте через временное окружение, не через файл"
                )
            os.environ[name] = getpass.getpass(prompt)
    record.update(status="running", last_started_utc=now())
    atomic_json(MANIFEST, record)
    new_calls = 0
    try:
        for stage in stages:
            if stage == "followup_doubt":
                main_rows = run_bank.read_jsonl(stage_path("main"))
                if sum(r["error"] is None for r in main_rows) != len(tasks):
                    raise ValueError("H9 требует все 38 успешных main-ответов")
            jobs = build_jobs(stage, tasks, args.model, cfg, df, split)
            path = stage_path(stage)
            key = run_classifier.row_key if stage == "typeE" else run_bank.row_key
            record["stages"][stage] = {"expected": len(jobs), "status": "running"}
            atomic_json(MANIFEST, record)
            for n, job in enumerate(jobs, 1):
                if args.max_new_calls is not None and new_calls >= args.max_new_calls:
                    record.update(status="paused_budget", updated_utc=now())
                    atomic_json(MANIFEST, record)
                    return 0
                res, row = run_bank.run_job(job, LOG)
                save_row(path, row, key)
                if not res.cached:
                    new_calls += 1
                print(
                    f"{stage} {n}/{len(jobs)} {job['task']['id']} "
                    f"{'cache' if res.cached else 'new'} "
                    f"{'ERROR' if res.error else 'ok'}",
                    flush=True,
                )
                if res.error:
                    # Не штурмуем квоту и не называем техошибку ответом модели.
                    record["stages"][stage].update(
                        status="paused_error", last_call_id=res.call_id
                    )
                    record.update(status="paused_error", updated_utc=now())
                    deadline = quota_resume_after(LOG, res.call_id)
                    if deadline:
                        record["resume_not_before_utc"] = deadline
                    atomic_json(MANIFEST, record)
                    return run_bank.EXIT_TECH_ERRORS
                if not res.cached and args.delay:
                    time.sleep(args.delay)
            rows = run_bank.read_jsonl(path)
            record["stages"][stage] = {
                "expected": len(jobs),
                "received": len(rows),
                "successful": sum(r["error"] is None for r in rows),
                "status": "complete",
            }
            atomic_json(MANIFEST, record)
        complete = all(
            record["stages"].get(s, {}).get("status") == "complete" for s in STAGES
        )
        record.update(status="complete" if complete else "partial", updated_utc=now())
        atomic_json(MANIFEST, record)
        print(
            f"Сохранено в {run_bank.show(OUT)}; новых вызовов: {new_calls}", flush=True
        )
        return 0
    except BaseException as exc:
        record.update(
            status="interrupted", updated_utc=now(), error_type=type(exc).__name__
        )
        atomic_json(MANIFEST, record)
        # Не печатаем сырые сетевые исключения с потенциальными SSO URL/токенами.
        print(
            f"Прогон остановлен: {type(exc).__name__}. Журнал и полученные ответы сохранены.",
            flush=True,
        )
        return 1


if __name__ == "__main__":
    sys.exit(main())
