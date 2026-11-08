"""Finish the long local experiment even when the chat turn ends.

The LLM extractor checkpoints each batch. On a resumed run completed scores are
kept; missing methods are evaluated before a final comparison is written.
"""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys

OUT = Path("outputs/remarks_comparison")


def status(state, stage):
    (OUT / "job_status.json").write_text(json.dumps({
        "state": state, "stage": stage,
        "updated_utc": datetime.now(timezone.utc).isoformat()}, indent=2))


def run(*args):
    print("Running", " ".join(args), flush=True)
    subprocess.run([sys.executable, "-u", "-m", *args], check=True)


def main():
    manifest = json.loads((OUT / "baseline_manifest.json").read_text())
    for path, expected in manifest["sha256"].items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Baseline input changed: {path}")
    if not (OUT / "llm_scores.json").exists():
        status("running", "local_llm_extraction_and_validation")
        run("experiments.prepare_remarks", "llm", "--evaluate")
    for method in ["tfidf", "tfidf_direct", "embeddings"]:
        if not (OUT / f"{method}_scores.json").exists():
            status("running", method)
            if method == "embeddings" and not (OUT / "cache/embedding_index.parquet").exists():
                run("experiments.prepare_remarks", "embeddings")
            run("experiments.compare_remarks", method)
    run("experiments.summarize_remarks")
    status("complete", "comparison_saved")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    try:
        main()
    except Exception as error:
        status("failed", str(error))
        raise
