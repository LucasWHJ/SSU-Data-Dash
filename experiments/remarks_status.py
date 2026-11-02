"""Read the actual long-job progress and any completed validation results."""
import json
from pathlib import Path
import re

from experiments.summarize_remarks import summarize


def main():
    out = Path("outputs/remarks_comparison")
    status = json.loads((out / "job_status.json").read_text())
    print("Job:", status["state"], "Stage:", status["stage"])
    log = (out / "llm_run.log").read_text()
    progress = re.findall(r"LLM (\d+) / (\d+) new rows/sec ([0-9.]+)", log)
    if progress and status["state"] != "stopped":
        completed, total, rate = map(float, progress[-1])
        print(f"LLM descriptions cached: {int(completed):,}/{int(total):,} ({100 * completed / total:.1f}%)")
        if rate > 0 and completed < total:
            print(f"Estimated extraction time remaining at the last batch rate: {(total - completed) / rate / 3600:.1f} hours")
    fast = out / "fast/status.json"
    if fast.exists():
        print("Fast screen:", json.loads(fast.read_text()))
    summarize()


if __name__ == "__main__":
    main()
