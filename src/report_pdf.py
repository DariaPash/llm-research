"""Экспорт report/report.md в PDF через Chrome.

python -m src.report_pdf. Требует локальный Chrome.
Широкие таблицы разбиваются с повтором ключевых столбцов, без потери клеток.
"""

import copy
import hashlib
import html
import json
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

from bs4 import BeautifulSoup
import mistune

CHROME = (
    shutil.which("google-chrome")
    or "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
)

ROOT = Path(__file__).resolve().parent.parent
CSS = """
@page { size: A4; margin: 17mm 15mm 18mm;
  @bottom-right { content: counter(page); font: 9pt Arial; color: #667; } }
* { box-sizing: border-box; }
body { font: 10pt/1.45 Arial, sans-serif; color: #172331; margin: 0; }
h1 { font-size: 21pt; line-height: 1.18; color: #173c59; margin: 0 0 9mm; }
h2 { font-size: 16pt; color: #173c59; border-bottom: 1px solid #b7cbd9; margin-top: 10mm; }
h3 { font-size: 12pt; margin-top: 6mm; } h4 { font-size: 11pt; }
h1, h2, h3, h4 { break-after: avoid; }
p, li { orphans: 3; widows: 3; } p { margin: 2mm 0; }
table { border-collapse: collapse; width: 100%; table-layout: fixed; margin: 3mm 0 5mm;
  font-size: 7.6pt; line-height: 1.3; }
th, td { border: 0.5pt solid #c4d0d8; padding: 1.6mm; vertical-align: top; overflow-wrap: anywhere; }
th { background: #edf3f7; font-weight: bold; } thead { display: table-header-group; }
tr { break-inside: avoid; } tbody tr:nth-child(even) { background: #f7f9fb; }
img { display: block; max-width: 100%; max-height: 170mm; object-fit: contain; margin: 4mm auto; }
pre { white-space: pre-wrap; overflow-wrap: anywhere; background: #f2f5f7; padding: 3mm;
  font-size: 8pt; line-height: 1.3; }
code { font-family: Menlo, monospace; font-size: 0.87em; overflow-wrap: anywhere; }
blockquote { margin: 3mm 0; padding: 2mm 4mm; border-left: 2pt solid #6c94b0; background: #f5f8fa; }
a { color: #174f76; text-decoration: none; } .table-part { color: #667; font-size: 8pt; }
"""


def render_html(markdown, report_dir):
    # Служебные метки редактора не являются текстом готового отчёта.
    markdown = re.sub(
        r"^> Файл собирает.*$",
        lambda m: (
            "> Расчёт: `results/summary_metrics.json`, sha256 `"
            + re.search(r"sha256 `([^`]+)`", m[0])[1]
            + "`. Источник текста: `report/report.md`. Полный обновлённый отчёт."
        ),
        markdown,
        flags=re.M,
    )
    markdown = re.sub(r"<!--.*?-->", "", markdown, flags=re.S)
    md = mistune.create_markdown(escape=False, plugins=["table", "strikethrough"])
    soup = BeautifulSoup(md(markdown), "html.parser")
    for link in soup.find_all("a", href=True):
        if not re.match(r"https?://|#", link["href"]):
            link.unwrap()
    for table in list(soup.find_all("table")):
        rows = table.find_all("tr")
        ncols = len(rows[0].find_all(["th", "td"], recursive=False))
        if ncols <= 7:
            continue
        # Сохраняем модель и shots/type, иначе части E нельзя сопоставить.
        headers = [
            c.get_text(strip=True)
            for c in rows[0].find_all(["th", "td"], recursive=False)
        ]
        key_names = {
            "model",
            "model_key",
            "модель",
            "cot_model",
            "main_model",
            "модель 1",
            "модель 2",
            "type",
            "тип",
            "group",
            "experiment",
            "shots",
            "ключ",
        }
        keys = [i for i, header in enumerate(headers) if header in key_names][:3] or [0]
        values = [i for i in range(ncols) if i not in keys]
        width = 7 - len(keys)
        for start in range(0, len(values), width):
            indices = sorted([*keys, *values[start : start + width]])
            label = soup.new_tag("p", attrs={"class": "table-part"})
            label.string = f"Широкая таблица, часть {start // width + 1}; ключевые столбцы повторены."
            table.insert_before(label)
            part = copy.deepcopy(table)
            for row in part.find_all("tr"):
                for i, cell in enumerate(row.find_all(["th", "td"], recursive=False)):
                    if i not in indices:
                        cell.decompose()
            table.insert_before(part)
        table.decompose()
    base = html.escape(report_dir.resolve().as_uri() + "/", quote=True)
    return (
        f'<!doctype html><html lang="ru"><head><meta charset="utf-8">'
        f'<base href="{base}"><title>ЛР3 — LLM в классическом ML</title>'
        f"<style>{CSS}</style></head><body>{soup}</body></html>"
    )


def build(root=ROOT):
    report = root / "report/report.md"
    summary = root / "results/summary_metrics.json"
    text = report.read_text(encoding="utf-8")
    digest = hashlib.sha256(summary.read_bytes()).hexdigest()[:12]
    if f"sha256 `{digest}`" not in text:
        raise ValueError(
            "Отпечаток summary в отчёте устарел; сверьте расчёт и текст отчёта"
        )
    figures = [
        report.parent / name for name in re.findall(r"!\[[^\]]*\]\(([^)]+)\)", text)
    ]
    if any(not p.is_file() for p in figures):
        raise ValueError("В отчёте отсутствуют файлы рисунков")
    directory = report.parent
    directory.mkdir(exist_ok=True)
    target = directory / "report.pdf"
    # Печать во временный файл: при сбое прежний PDF не уничтожается.
    with tempfile.TemporaryDirectory(prefix="lab3_pdf_", dir=root) as tmp:
        source = Path(tmp) / "main.html"
        source.write_text(render_html(text, report.parent), encoding="utf-8")
        pdf = Path(tmp) / "report.pdf"
        subprocess.run(
            [
                CHROME,
                "--headless=new",
                "--disable-gpu",
                "--no-pdf-header-footer",
                f"--user-data-dir={Path(tmp) / 'chrome'}",
                f"--print-to-pdf={pdf}",
                source.resolve().as_uri(),
            ],
            check=True,
            capture_output=True,
            timeout=180,
        )
        if not pdf.exists() or not pdf.read_bytes().startswith(b"%PDF-"):
            raise ValueError("Chrome не создал PDF")
        target.write_bytes(pdf.read_bytes())
    paths = [report, summary, target, *sorted(set(figures))]
    record = {
        "status": "full_current_markdown_report_rendered",
        "source_report": "report/report.md",
        "output_pdf": "report/report.pdf",
        "renderer": "Google Chrome headless",
        "renderer_version": subprocess.check_output(
            [CHROME, "--version"], text=True
        ).strip(),
        "command": "python -m src.report_pdf",
        "note": "PDF из текста отчёта; широкие таблицы разбиты с повторением ключевых столбцов.",
        "sha256": {
            str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in paths
        },
    }
    (directory / "pdf_documentation.json").write_text(
        json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return target


if __name__ == "__main__":
    print(build())
