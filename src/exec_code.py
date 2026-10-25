"""Однократный запуск кода D без исправлений; результаты — results/d_exec.jsonl.

pass@1 — завершение с кодом 0 без таймаута, а не правильность методологии.
Метрика берётся из последней числовой строки stdout и сверяется с эталоном.
Временная папка, очищенное окружение и таймаут не обеспечивают полной изоляции:
запускаемому коду доступны сеть и файлы вне папки.
"""

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

from src import bank, grade, parse, run_bank

ROOT = run_bank.ROOT
PYTHON = bank.PYTHON  # тот же Python, которым запущен конвейер
TIMEOUT_S = 300
MAX_OUTPUT_CHARS = (
    20000  # stdout/stderr длиннее обрезаются (начало и конец сохраняются)
)

# Передаём только разрешённые переменные окружения, без API-ключей.
ENV_WHITELIST = [
    "PATH",
    "HOME",
    "LANG",
    "LC_ALL",
    "LC_CTYPE",
    "TMPDIR",
    "USER",
    "LOGNAME",
    "SHELL",
    "TERM",
]


def clean_env() -> dict:
    env = {k: os.environ[k] for k in ENV_WHITELIST if k in os.environ}
    env["MPLBACKEND"] = "Agg"  # plt.show() не открывает окно и не ждёт закрытия
    return env


def cut(text: str) -> str:
    if len(text) <= MAX_OUTPUT_CHARS:
        return text
    half = MAX_OUTPUT_CHARS // 2
    return (
        text[:half]
        + f"\n…[обрезано {len(text) - MAX_OUTPUT_CHARS} символов]…\n"
        + text[-half:]
    )


def run_code(code_path: Path, timeout_s: int = TIMEOUT_S) -> dict:
    """Запускает файл во временной папке. Файлы, которые создаст код, удаляются вместе с папкой."""
    with tempfile.TemporaryDirectory(prefix="d_exec_") as tmp:
        script = Path(tmp) / code_path.name
        shutil.copyfile(code_path, script)
        t0 = time.perf_counter()
        # Отдельная группа позволяет завершить по таймауту и дочерние процессы CV.
        proc = subprocess.Popen(
            [str(PYTHON), script.name],
            cwd=tmp,
            env=clean_env(),
            text=True,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
        )
        timed_out = False
        try:
            stdout, stderr = proc.communicate(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            timed_out = True
            os.killpg(proc.pid, signal.SIGKILL)
            stdout, stderr = proc.communicate()
        finally:
            # Дочерние процессы, оставшиеся после выхода кода, тоже завершаем.
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        duration = time.perf_counter() - t0
    return {
        "returncode": proc.returncode,
        "timed_out": timed_out,
        "duration_s": round(duration, 2),
        "stdout": cut(stdout),
        "stderr": cut(stderr),
    }


def extract_metric(stdout: str) -> tuple[float | None, str | None]:
    """(метрика, строка, из которой она взята): последняя строка stdout с числом."""
    for line in reversed(stdout.splitlines()):
        picked = parse.pick_number(parse.clean(line), "en")
        if picked is not None:
            return picked[0], line.strip()
    return None, None


def code_path(
    out: Path, model_key: str, task_id: str, experiment: str, lang: str
) -> Path:
    name = (
        task_id
        if (experiment, lang) == ("main", "ru")
        else f"{task_id}__{experiment}_{lang}"
    )
    return out / "d_code" / model_key / f"{name}.py"


def exec_row(row: dict, task: dict | None, out: Path, timeout_s: int) -> dict:
    """Одна запись d_exec.jsonl для одного ответа типа D."""
    code, notes = parse.extract_code(row["response_text"] or "")
    rec = {
        "experiment": row["experiment"],
        "lang": row["lang"],
        "task_id": row["task_id"],
        "model_key": row["model_key"],
        "call_id": row["call_id"],
        "ts_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "code_found": code is not None,
        "extract_notes": notes,
        "code_path": None,
        "returncode": None,
        "timed_out": False,
        "duration_s": None,
        "stdout": None,
        "stderr": None,
        "metric": None,
        "metric_line": None,
        "reference_metric": task.get("reference_metric") if task else None,
        "tolerance": task.get("tolerance") if task else None,
        "metric_ok": None,
        "pass_at_1": False,
    }
    if code is None:
        return rec
    path = code_path(
        out, row["model_key"], row["task_id"], row["experiment"], row["lang"]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        code, encoding="utf-8"
    )  # без правок: ровно то, что извлечено из ответа
    rec["code_path"] = run_bank.show(path)
    rec.update(run_code(path, timeout_s))
    rec["pass_at_1"] = rec["returncode"] == 0 and not rec["timed_out"]
    if rec["pass_at_1"]:
        rec["metric"], rec["metric_line"] = extract_metric(rec["stdout"])
        ref, tol = rec["reference_metric"], rec["tolerance"]
        if rec["metric"] is not None and ref is not None:
            rec["metric_ok"] = grade.close(
                rec["metric"], ref, tol if tol is not None else grade.DEFAULT_TOLERANCE
            )
    return rec


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Запуск кода из ответов типа D")
    ap.add_argument(
        "--experiment",
        default="main",
        help="main или несколько через запятую, например main,paraphrase",
    )
    ap.add_argument(
        "--lang",
        default=None,
        help="ru, en или ru,en (по умолчанию все языки, для которых есть таблица)",
    )
    ap.add_argument(
        "--bank", default=None, help="файл задач (по умолчанию data/tasks.jsonl)"
    )
    ap.add_argument("--timeout", type=int, default=TIMEOUT_S)
    ap.add_argument(
        "--results-dir",
        default=None,
        help="отдельный каталог ответов и d_exec (например results/extra)",
    )
    ap.add_argument("--models", default=None, help="только указанные ключи моделей")
    ap.add_argument("--demo", action="store_true", help="читать и писать demo/results/")
    args = ap.parse_args(argv)

    if args.demo and args.results_dir:
        ap.error("--demo нельзя совмещать с --results-dir")
    out = (
        Path(args.results_dir).resolve()
        if args.results_dir
        else run_bank.out_dir(args.demo)
    )
    selected_models = set(run_bank.parse_list(args.models) or [])
    tables = [
        (exp, lang)
        for exp in run_bank.parse_list(args.experiment)
        for lang in (run_bank.parse_list(args.lang) or ["ru", "en"])
        if (out / f"{exp}_{lang}.jsonl").exists()
    ]
    tasks = {
        t["id"]: t
        for t in bank.load_tasks(Path(args.bank) if args.bank else bank.BANK_PATH)
    }
    exec_path = out / "d_exec.jsonl"
    done = {
        (r["experiment"], r["lang"], r["model_key"], r["task_id"], r["call_id"])
        for r in run_bank.read_jsonl(exec_path)
    }

    todo = []
    for exp, lang in tables:
        for row in run_bank.read_jsonl(out / f"{exp}_{lang}.jsonl"):
            key = (
                row["experiment"],
                row["lang"],
                row["model_key"],
                row["task_id"],
                row["call_id"],
            )
            if (
                row["answer_kind"] == "code"
                and (not selected_models or row["model_key"] in selected_models)
                and row["error"] is None
                and key not in done
            ):
                todo.append(row)
    print(f"Ответов типа D к запуску: {len(todo)} (уже запущено раньше: {len(done)})")

    exec_path.parent.mkdir(parents=True, exist_ok=True)
    for row in todo:
        rec = exec_row(row, tasks.get(row["task_id"]), out, args.timeout)
        # Журнал запусков только дописывается: каждый ответ запускается один раз (pass@1).
        with open(exec_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        status = (
            "код не найден"
            if not rec["code_found"]
            else "таймаут"
            if rec["timed_out"]
            else f"код возврата {rec['returncode']}, метрика {rec['metric']}"
        )
        print(f"  {rec['model_key']:22} {rec['task_id']:6} {status}")
    print(f"Результаты: {run_bank.show(exec_path)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
