"""Compare remarks representations with fixed HGB settings and monthly folds.

Run from the repository root: python -m experiments.compare_remarks tfidf
Other methods: embeddings, llm. Their caches are made by prepare_remarks.py.
No vocabulary, dimensionality reduction, or category mapping is fit on a
validation month. Original keyword flags are replaced; word count is retained.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

from files import data, features, models
from run import VALIDATION_MONTHS, TRACKS

OUT = Path("outputs/remarks_comparison")
CACHE = OUT / "cache"
DIMENSIONS = 32
LLM_COLUMNS = ["condition"]


def text_hash(text):
    return hashlib.sha256(str(text).encode("utf-8")).hexdigest()


def texts(homes):
    return homes["remarks"].fillna("").astype(str)


def read_cache(homes, method):
    """Verify exact text matches; never silently use stale or missing features."""
    if method == "embeddings":
        index = pd.read_parquet(CACHE / "embedding_index.parquet").set_index("ml_num")
        values = np.load(CACHE / "embeddings.npy", mmap_mode="r")
    else:
        index = pd.read_parquet(CACHE / "llm.parquet").set_index("ml_num")
        values = index[LLM_COLUMNS].replace(0, np.nan).to_numpy(dtype=np.float32)
        index["position"] = np.arange(len(index))
    assert index.index.is_unique, "Duplicate IDs in text cache"
    selected = index.reindex(homes["ml_num"])
    if selected["text_hash"].isna().any():
        raise ValueError("Text cache is incomplete; finish prepare_remarks first")
    expected = texts(homes).map(text_hash).to_numpy()
    if not np.array_equal(selected["text_hash"].to_numpy(), expected):
        raise ValueError("Cached features do not match the current remarks")
    return np.asarray(values[selected["position"].to_numpy(dtype=int)])


def fit_text_features(method, train, homes, cached=None):
    if method in ("tfidf", "tfidf_direct"):
        vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=0.05,
                                   max_df=0.95, max_features=3000,
                                   sublinear_tf=True, dtype=np.float32)
        a = vectorizer.fit_transform(texts(train))
        b = vectorizer.transform(texts(homes))
        print("training vocabulary", len(vectorizer.vocabulary_), flush=True)
        if method == "tfidf_direct":
            return a.toarray(), b.toarray()
    else:
        a, b = cached[train.index], cached[homes.index]
        if method == "llm":
            return a, b
    # One fitted reducer per month, learned exclusively from earlier sales.
    reducer = TruncatedSVD(n_components=min(DIMENSIONS, a.shape[1] - 1), random_state=0)
    a = reducer.fit_transform(a)
    b = reducer.transform(b)
    return a.astype(np.float32), b.astype(np.float32)


def score(rows):
    error = 100 * (rows["predicted_price"] - rows["sold_price"]).abs() / rows["sold_price"]
    return {"mdape": float(error.median()), "mape": float(error.mean())}


def compare(method):
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((OUT / "baseline_manifest.json").read_text())
    for path, expected in manifest["sha256"].items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Baseline input changed: {path}. Restore it or rerun the full comparison.")
    listings = data.load("listings").reset_index(drop=True)
    cached = None if method in ("tfidf", "tfidf_direct") else read_cache(listings, method)
    results = []
    dimensions = {}
    for month in VALIDATION_MONTHS:
        started = time.monotonic()
        train = listings[listings["date_sold"] < month.start_time]
        homes = listings[listings["date_sold"].dt.to_period("M") == month]
        a, b = fit_text_features(method, train, homes, cached)
        dimensions[str(month)] = a.shape[1]
        columns = (["text_" + name for name in LLM_COLUMNS] if method == "llm"
                   else [f"text_{i}" for i in range(a.shape[1])])
        encoded = pd.DataFrame(np.vstack([a, b]), columns=columns,
                               index=np.concatenate([train.index, homes.index]))
        for track in TRACKS:
            base = getattr(features, track + "_features")

            def make_features(rows):
                x = base(rows)
                x = x.drop(columns=[c for c in x if c.startswith("remark_") and c != "remark_words"])
                return x.join(encoded.loc[rows.index])

            # Obtain exactly the existing track's target, loss, and tree settings.
            model = models.make_model(track)
            if type(model) is not models.FeatureModel:
                raise ValueError("Baseline changed: review the experiment before rerunning")
            model.make_features = make_features
            prediction = np.round(model.fit(train).predict(data.hide_answers(homes, track)))
            if not (np.isfinite(prediction).all() and (prediction > 0).all()):
                raise ValueError("Model returned invalid prices")
            rows = pd.DataFrame({"ml_num": homes["ml_num"], "track": track,
                                 "month": str(month), "sold_price": homes["sold_price"],
                                 "predicted_price": prediction})
            results.append(rows)
            print(method, track, month, score(rows), flush=True)
        pd.concat(results).to_csv(OUT / f"{method}_predictions.csv", index=False)
        print("month seconds", round(time.monotonic() - started, 1), flush=True)
    predictions = pd.concat(results)
    summary = {track: score(predictions[predictions["track"] == track]) for track in TRACKS}
    metadata = {
        "method": method, "summary": summary, "fold_dimensions": dimensions,
        "tfidf_min_df": 0.05, "tfidf_max_df": 0.95, "tfidf_ngram_range": [1, 2],
        "months": list(map(str, VALIDATION_MONTHS)),
        "experiment_source_sha256": text_hash(Path(__file__).read_text()),
        "feature_source_sha256": text_hash(Path("files/features.py").read_text()),
        "model_source_sha256": text_hash(Path("files/models.py").read_text()),
    }
    if method in ("embeddings", "llm"):
        filename = "embedding_metadata.json" if method == "embeddings" else "llm_metadata.json"
        metadata["encoder"] = json.loads((CACHE / filename).read_text())
    (OUT / f"{method}_scores.json").write_text(json.dumps(metadata, indent=2))
    print("RESULT", json.dumps(summary), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("method", choices=["tfidf", "tfidf_direct", "embeddings", "llm"])
    compare(parser.parse_args().method)
