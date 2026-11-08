# Remarks experiments

These scripts compare text representations on the Lucas branch. They do not
change the active production model. Keep the current best version until the
complete comparison supports a replacement.

All candidates use the existing track's HGB hyperparameters, log target and loss.
Structured property features are unchanged. Original `remark_*` keyword flags
are replaced by each candidate representation; `remark_words` is retained.
The six validation months are December 2025 through May 2026. Each model trains
only on sales before that month. No hidden-test prices are used.

| Method | Representation |
|---|---|
| Existing baseline | Current regex flags and word count |
| `tfidf` | Word/bigram TF-IDF followed by 32-component TruncatedSVD |
| `tfidf_direct` | The same frequent-word TF-IDF weights, without dimensionality reduction |
| `embeddings` | Frozen MiniLM sentence embeddings, reduced to 32 components |
| `llm` | Local Qwen2.5 extraction of current property condition |

TF-IDF retains terms appearing in at least 5% and at most 95% of that fold's
training descriptions. Its vocabulary and SVD are fit on training homes only.
The embedding SVD is also fit on training homes only. Frozen pretrained encoders
can process each text independently and cache the result without fitting on
validation. Model revisions are pinned to commits before June 2026.

The local LLM maps descriptions to condition codes: 0 = unclear, 1 = original,
dated or repairs needed, 2 = partly updated, 3 = fully renovated or new. Unknown
and invalid extraction values become missing features rather than fabricated
facts. A four-field extraction pilot produced errors and was discarded before
any price validation; the final single-field prompt is recorded in cache metadata.
Check `llm_pilot.parquet` against the remarks before the full extraction.

## Running

TF-IDF uses the normal project environment. Embeddings and local LLM extraction
need a separate Python 3.11 environment with `requirements-remarks.txt`; the LLM
runtime requires an Apple Silicon Mac. The experiment currently uses an isolated
environment under `/tmp/ssu_remarks_env`. Downloads and caches stay in ignored
`outputs/remarks_comparison/`, and full LLM extraction can take many hours.

From the repository root, using the appropriate Python executable:

```bash
export DATA_DIR="$PWD/datasets"
python -m experiments.compare_remarks init  # once on a fresh checkout; runs baseline validation
python -m experiments.compare_remarks tfidf
python -m experiments.compare_remarks tfidf_direct
python -m experiments.prepare_remarks embeddings
python -m experiments.compare_remarks embeddings
python -m experiments.prepare_remarks llm --pilot 64
python -m experiments.prepare_remarks llm
python -m experiments.compare_remarks llm
python -m experiments.summarize_remarks
```

`python -m experiments.run_remarks_job` runs/resumes the long local LLM job,
evaluates it and fills in any remaining methods before writing the final table.
`job_status.json` reports running, complete or failed. No notifications or remote
jobs are scheduled; the Python process must keep running on this computer.
`python -m experiments.remarks_status` displays the cached row count, a rough
remaining-time estimate from the last batch, and completed validation scores.

The `init` command creates `baseline_predictions.csv` and `baseline_manifest.json`
from a fresh baseline validation. It refuses to overwrite an existing comparison.
Comparison checks the exact
IDs, months and targets against this baseline. A method gets a final score only
after all six folds complete. `comparison.md` and `comparison.csv` show pending
methods explicitly.

Embedding writes use temporary files and an incomplete-build marker. A stopped
rebuild cannot publish half-written vectors under an old complete index. Readers
verify encoder provenance, row positions, text hashes and finite, nonzero vectors.

LLM extraction appends a cache after every batch and resumes from completed IDs.
The cache verifies model revision, prompt hash and text hash. Keep the computer
running for extraction; rerun the command after an interruption to resume it.

Source models: [MiniLM](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
and [the local quantized Qwen model](https://huggingface.co/mlx-community/Qwen2.5-1.5B-Instruct-4bit).

These are validation results used for selection, not independent test results.
One configuration losing does not establish that its whole method is inferior.

## Fast screen

The full LLM run was stopped at the user's request because of its runtime.
`python -m experiments.fast_remarks` uses a fixed 600-home history sample plus
50 validation homes per month, at most 900 descriptions total. It reuses already
computed embeddings and any matching LLM checkpoints. All methods use exactly
the same homes and original HGB settings. Results go to `fast/comparison.md` and
are exploratory only; they cannot be compared to the full-data scores or used to
replace the active model without stronger evidence. This command never resumes
the full 150,000-description extraction.
