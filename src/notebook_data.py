"""Read-only combination of the original experiment and saved Korolev results.

Used by analysis.ipynb. Does not call models, run generated code, prepare review
worksheets, or overwrite the historical four-model summary.
"""

import hashlib
import numpy as np
import pandas as pd

from src import analysis as an, blind, manifest, run_bank

MODEL = "korolev-k5"


class NotebookData:
    def __init__(self, paths, models, demo=False):
        self.paths = paths
        self.models = models
        self.sources = [(paths["results"], paths["grading"], paths["log"])]
        extra = paths["results"] / "korolev"
        if not demo and (extra / "main_ru.jsonl").exists():
            self.sources.append(
                (
                    extra,
                    paths["grading"] / "korolev",
                    an.ROOT / "logs/korolev_calls.jsonl",
                )
            )
        self.manual = self._manual()

    def _manual(self):
        frames = []
        for _, grading, _ in self.sources:
            frame = an.load_manual(grading)
            if not frame.empty:
                # Checklist means normally ignore NA. For an unfinished second
                # sheet that would silently turn one review into a two-rater mean.
                cols = ["score_mean", "correct_manual", "any_error_mean"]
                cols += [f"{column}_mean" for column in blind.CHECKLIST]
                frame.loc[~frame["complete"], cols] = np.nan
                frames.append(frame)
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    def answers(self, stage, lang="ru"):
        frames, errors = [], 0
        for results, _, log in self.sources:
            frame, n_errors = an.answers(results, stage, lang, self.manual)
            errors += n_errors
            if not frame.empty:
                frames.append(an.add_full_input(frame, log, self.models))
        return (
            pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(),
            errors,
        )

    def raw_answers(self, stage, lang="ru"):
        return pd.concat(
            [
                an.read_table(results, f"{stage}_{lang}")
                for results, _, _ in self.sources
            ],
            ignore_index=True,
        )

    def runs(self, models):
        records = [
            row for _, _, log in self.sources for row in run_bank.read_jsonl(log)
        ]
        main_models = set(self.raw_answers("main")["model_key"])
        records = [row for row in records if row["model_key"] in main_models]
        return {
            "models": manifest.by_model(records, models),
            "calls_by_experiment": manifest.by_experiment(records),
        }

    def fingerprints(self):
        result = an.input_fingerprints(self.paths)
        files = [
            an.ROOT / "data/tasks.jsonl",
            an.ROOT / "data/e_split.json",
            an.ROOT / "config/models.json",
            an.ROOT / "src/notebook_data.py",
        ]
        for results, grading, log in self.sources[1:]:
            files += list(results.glob("*.jsonl")) + list(grading.glob("*.csv")) + [log]
        for path in files:
            if path.exists():
                result[run_bank.show(path)] = hashlib.sha256(
                    path.read_bytes()
                ).hexdigest()[:12]
        return result

    def type_e_table(self):
        frames = []
        for index, (results, _, _) in enumerate(self.sources):
            frame = an.typeE_table(results)
            # build() also includes the same historical classical baselines.
            if index:
                frame = frame[frame["source"] == "llm"]
            frames.append(frame)
        return pd.concat(frames, ignore_index=True)

    def type_e_predictions(self, split, lang="ru"):
        return pd.concat(
            [
                an.typeE_llm_preds(results, split, lang)
                for results, _, _ in self.sources
            ],
            ignore_index=True,
        )

    def review_status(self, models):
        rows = []
        for model in models:
            group = self.manual[self.manual["model_key"] == model]
            rows.append(
                {
                    "model_key": model,
                    "answers_BCD": len(group),
                    "r1_filled": int(
                        group["score_0_2_r1"].notna().sum()
                        + group.loc[group["type"] == "D", "d5_no_stratify_r1"]
                        .notna()
                        .sum()
                    ),
                    "r2_filled": int(
                        group["score_0_2_r2"].notna().sum()
                        + group.loc[group["type"] == "D", "d5_no_stratify_r2"]
                        .notna()
                        .sum()
                    ),
                    "complete_two_raters": int(group["complete"].sum()),
                    "review_protocol": "один автор, не вслепую; второй лист не заполнен"
                    if model == MODEL and not group["complete"].all()
                    else "два рабочих листа; происхождение см. grading/review_status.md",
                }
            )
        return pd.DataFrame(rows)


def common_answers(main, models):
    """Same automatically scored A/B/F tasks for every compared model."""
    frame = main[main["type"].isin(["A", "B", "F"]) & main["correct"].apply(an.binary)]
    sets = [set(frame.loc[frame["model_key"] == model, "task_id"]) for model in models]
    common = set.intersection(*sets) if sets else set()
    return frame[frame["task_id"].isin(common)].copy()


def rubric_scores(manual, models):
    rows = []
    for model in models:
        for typ in ("B", "C"):
            group = manual[(manual["model_key"] == model) & (manual["type"] == typ)]
            rows.append(
                {
                    "model_key": model,
                    "type": typ,
                    "n": len(group),
                    "n_two_raters": int(group["score_mean"].notna().sum()),
                    "mean_score": group["score_mean"].mean(),
                    "r1_mean": group["score_r1"].mean(),
                    "r2_mean": group["score_r2"].mean(),
                }
            )
    return pd.DataFrame(rows)


def resources(main, pricing, models):
    """Latency/tokens use ALL successful answers, not just scored A/B/F."""
    frame = an.add_cost(main, pricing, models)
    rows = []
    for model, group in frame.groupby("model_key", sort=False):
        rows.append(
            {
                "model_key": model,
                "n_answers": len(group),
                "latency_s_mean": group["latency_s"].mean(),
                "tokens_in_mean": group["tokens_in"].mean(),
                "tokens_in_full_mean": group["tokens_in_full"].mean(),
                "tokens_out_mean": group["tokens_out"].mean(),
                "cost_rub_total": group["cost_rub"].sum()
                if group["cost_rub"].notna().all()
                else np.nan,
            }
        )
    return pd.DataFrame(rows)


def h6_by_model(main, cot, n_boot, seed):
    direct = cot[cot["model_key"].isin(main["model_key"].unique())]
    pairs = an.paired_conditions(direct, main, ["model_key", "task_id", "type"])
    pairs["group"] = pairs["model_key"] + ": " + pairs["type"]
    return an.name_conditions(
        an.compare_conditions(pairs, "group", n_boot, seed, pooled=False), "cot", "main"
    )
