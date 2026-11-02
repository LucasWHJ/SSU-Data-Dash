"""Summarize completed experiments; pending methods never get invented scores."""
import json
import pandas as pd

from experiments.compare_remarks import OUT, score
from run import TRACKS, VALIDATION_MONTHS


def summarize():
    baseline = pd.read_csv(OUT / "baseline_predictions.csv")
    rows = []
    methods = ["regex", "tfidf", "tfidf_direct", "embeddings", "llm"]
    for method in methods:
        path = OUT / f"{method}_scores.json"
        if method != "regex" and not path.exists():
            state = "pending"
            job = OUT / "job_status.json"
            if method == "llm" and job.exists() and json.loads(job.read_text()).get("state") == "stopped":
                state = "full_run_stopped"
            for track in TRACKS:
                rows.append({"method": method, "track": track, "status": state})
            continue
        predictions = baseline if method == "regex" else pd.read_csv(OUT / f"{method}_predictions.csv")
        for track in TRACKS:
            reference = baseline[baseline["track"] == track]
            candidate = predictions[predictions["track"] == track]
            keys = ["ml_num", "month", "sold_price"]
            paired = reference.merge(candidate, on=keys, validate="one_to_one")
            if len(paired) != len(reference) or len(candidate) != len(reference):
                raise ValueError(f"Incomplete or inconsistent validation homes for {method}/{track}")
            if set(candidate["month"]) != set(map(str, VALIDATION_MONTHS)):
                raise ValueError("Incomplete monthly validation")
            rows.append({"method": method, "track": track, "status": "complete",
                         **score(candidate)})
    results = pd.DataFrame(rows)
    results.to_csv(OUT / "comparison.csv", index=False)
    lines = ["# Remarks comparison", "", "Same structured features, HGB settings and six validation months.",
             "Keyword flags are replaced in each candidate; word count is retained.", "",
             "| Method | Track | MdAPE % | MAPE % | Status |", "|---|---|---:|---:|---|"]
    for row in rows:
        error = f"{row['mdape']:.4f}" if "mdape" in row else "—"
        mean = f"{row['mape']:.4f}" if "mape" in row else "—"
        lines.append(f"| {row['method']} | {row['track']} | {error} | {mean} | {row['status']} |")
    if (results["status"] == "complete").all():
        winners = results.loc[results.groupby("track")["mdape"].idxmin(), ["track", "method", "mdape"]]
        lines += ["", "Lowest observed validation MdAPE (not a hidden-test score):", "",
                  winners.to_string(index=False)]
    else:
        lines += ["", "The full three-method comparison is incomplete. No overall winner is declared yet."]
    if (OUT / "fast/comparison.md").exists():
        lines += ["", "A separate [fast shared-sample screen](fast/comparison.md) is available.",
                  "Its small-sample errors must not be mixed with the full-data scores above."]
    lines += ["", "Repeated selection on these months can overfit validation. Results do not prove",
              "one representation is universally better. Pretrained model revisions and the LLM",
              "prompt are recorded under cache/ and models/."]
    (OUT / "comparison.md").write_text("\n".join(lines) + "\n")
    print(results.to_string(index=False))


if __name__ == "__main__":
    summarize()
