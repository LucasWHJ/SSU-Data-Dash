"""Bounded, shared-sample remarks screen: 600 history homes + 50 per month.

This is a small exploratory screen, not the official full-data validation.
No full extraction or full-data model training is launched by this script.
"""
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

from files import data, features, models
from experiments import compare_remarks as common
from experiments.prepare_remarks import extract
from run import TRACKS, VALIDATION_MONTHS

OUT = common.OUT / "fast"
METHODS = ["regex", "tfidf", "tfidf_direct", "embeddings", "llm"]


def sample_types(rows, n, seed):
    """Proportional type quotas; sampling never uses prices or remarks."""
    counts = rows.groupby("house_type_name").size().sort_index()
    exact = counts * n / len(rows)
    quota = np.floor(exact).astype(int)
    for name in (exact - quota).sort_values(ascending=False, kind="stable").index[:n - quota.sum()]:
        quota.loc[name] += 1
    return pd.concat([rows[rows.house_type_name == name].sample(int(count), random_state=seed)
                      for name, count in quota.items() if count])


def select_homes(listings):
    history = sample_types(listings[listings.date_sold < VALIDATION_MONTHS[0].start_time], 600, 0)
    parts = [history.assign(sample_role="history")]
    for i, month in enumerate(VALIDATION_MONTHS):
        homes = listings[listings.date_sold.dt.to_period("M") == month]
        parts.append(sample_types(homes, 50, i + 1).assign(sample_role=str(month)))
    selected = pd.concat(parts).sort_values(["date_sold", "ml_num"]).reset_index(drop=True)
    assert len(selected) == 900 and selected.ml_num.is_unique
    return selected


def update_status(state, stage):
    (OUT / "status.json").write_text(json.dumps({"state": state, "stage": stage}, indent=2))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    homes = select_homes(data.load("listings"))
    homes[["ml_num", "sample_role", "date_sold"]].to_csv(OUT / "selected_ids.csv", index=False)
    update_status("running", "extracting_at_most_900_descriptions")
    # Reuse matching records from the stopped full run, but write the subset's
    # parquet separately so it can never masquerade as a complete full cache.
    extract(homes, pilot=False, cache_dir=OUT / "cache",
            checkpoint_path=common.CACHE / "llm_condition.jsonl")
    llm = pd.read_parquet(OUT / "cache/llm.parquet").set_index("ml_num").loc[homes.ml_num]
    np.testing.assert_array_equal(llm.text_hash.to_numpy(), common.texts(homes).map(common.text_hash).to_numpy())
    cached = {
        "embeddings": common.read_cache(homes, "embeddings"),
        "llm": llm[common.LLM_COLUMNS].replace(0, np.nan).to_numpy(dtype=np.float32),
    }
    results = []
    for month in VALIDATION_MONTHS:
        # The same 600 initial histories plus earlier sampled validation sales.
        train = homes[homes.date_sold < month.start_time]
        validation = homes[homes.sample_role == str(month)]
        for method in METHODS:
            update_status("running", f"{method}_{month}")
            if method != "regex":
                a, b = common.fit_text_features(method, train, validation, cached.get(method))
                encoded = pd.DataFrame(np.vstack([a, b]),
                    index=np.concatenate([train.index, validation.index]),
                    columns=[f"text_{i}" for i in range(a.shape[1])])
            for track in TRACKS:
                model = models.make_model(track)
                if method != "regex":
                    base = getattr(features, track + "_features")

                    def make_features(rows):
                        x = base(rows)
                        return x.drop(columns=[c for c in x if c.startswith("remark_") and c != "remark_words"]).join(encoded.loc[rows.index])

                    model.make_features = make_features
                prediction = np.round(model.fit(train).predict(data.hide_answers(validation, track)))
                assert np.isfinite(prediction).all() and (prediction > 0).all()
                result = pd.DataFrame({"method": method, "track": track,
                    "ml_num": validation.ml_num, "month": str(month),
                    "sold_price": validation.sold_price, "predicted_price": prediction})
                results.append(result)
                print(method, track, month, common.score(result), flush=True)
        pd.concat(results).to_csv(OUT / "predictions.csv", index=False)
    predictions = pd.concat(results)
    summary = []
    for track in TRACKS:
        reference = predictions[(predictions.method == "regex") & (predictions.track == track)]
        for method in METHODS:
            candidate = predictions[(predictions.method == method) & (predictions.track == track)]
            paired = reference.merge(candidate, on=["ml_num", "month", "sold_price"], validate="one_to_one")
            assert len(paired) == len(reference) == len(candidate) == 300
            summary.append({"method": method, "track": track, "validation_homes": len(candidate), **common.score(candidate)})
    pd.DataFrame(summary).to_csv(OUT / "comparison.csv", index=False)
    lines = ["# Fast remarks screen — exploratory only", "",
        "600 initial history homes; each month adds earlier sampled sales. All methods",
        "use the same 50 validation homes per month (300 per track), HGB settings and",
        "structured features. Sampling uses date and property type, never price.", "",
        "These errors are NOT comparable to the full-data 8.89% / 4.11% results.", "",
        "| Method | Track | MdAPE % | MAPE % |", "|---|---|---:|---:|"]
    for row in summary:
        lines.append(f"| {row['method']} | {row['track']} | {row['mdape']:.3f} | {row['mape']:.3f} |")
    lines += ["", "The sample is small: differences may be noise and rankings may change at full scale.",
              "The active production model has not been changed based on this screen.",
              f"Runtime: {time.monotonic() - started:.1f} seconds."]
    (OUT / "comparison.md").write_text("\n".join(lines) + "\n")
    (OUT / "settings.json").write_text(json.dumps({"history_homes": 600, "validation_per_month": 50,
        "months": list(map(str, VALIDATION_MONTHS)), "methods": METHODS,
        "source_sha256": {p: common.text_hash(Path(p).read_text()) for p in ["files/models.py", "files/features.py"]}}, indent=2))
    update_status("complete", "fast_comparison_saved")
    print(pd.DataFrame(summary).to_string(index=False), flush=True)
    from experiments.summarize_remarks import summarize
    summarize()


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    try:
        main()
    except Exception as error:
        update_status("failed", str(error))
        raise
