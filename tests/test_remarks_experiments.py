"""Checks for train-only text fitting and cache integrity."""
import numpy as np
import pandas as pd
import pytest

from experiments import compare_remarks as comparison
from experiments.prepare_remarks import load_checkpoint, parse_output


def test_text_fit_is_independent_of_validation_vocabulary():
    train = pd.DataFrame({"remarks": ["renovated kitchen " + word for word in
        ["bright", "modern", "updated", "original", "dated", "large"]]})
    a, _ = comparison.fit_text_features("tfidf", train,
                                       pd.DataFrame({"remarks": ["bright kitchen"]}))
    b, _ = comparison.fit_text_features("tfidf", train,
                                       pd.DataFrame({"remarks": ["futureonlyword"]}))
    np.testing.assert_allclose(a, b)


def test_embedding_reduction_does_not_fit_on_validation_vectors():
    rng = np.random.default_rng(0)
    cached = rng.normal(size=(20, 12)).astype(np.float32)
    train = pd.DataFrame(index=range(15))
    validation = pd.DataFrame(index=range(15, 20))
    a, _ = comparison.fit_text_features("embeddings", train, validation, cached)
    changed = cached.copy()
    changed[15:] *= 1000
    b, _ = comparison.fit_text_features("embeddings", train, validation, changed)
    np.testing.assert_allclose(a, b)


def test_cached_embeddings_are_aligned_by_id_and_reject_stale_text(tmp_path, monkeypatch):
    import json
    monkeypatch.setattr(comparison, "CACHE", tmp_path)
    rows = pd.DataFrame({"ml_num": ["a", "b"], "remarks": ["first", "second"]})
    pd.DataFrame({"ml_num": rows.ml_num, "position": [0, 1],
                  "text_hash": rows.remarks.map(comparison.text_hash)}).to_parquet(
                      tmp_path / "embedding_index.parquet")
    np.save(tmp_path / "embeddings.npy", np.array([[1, 2], [3, 4]], dtype=np.float32))
    (tmp_path / "embedding_metadata.json").write_text(json.dumps({
        "model": comparison.EMBEDDING_MODEL, "revision": "test", "rows": 2}))
    np.testing.assert_array_equal(comparison.read_cache(rows.iloc[::-1], "embeddings"), [[3, 4], [1, 2]])
    with pytest.raises(ValueError, match="do not match"):
        comparison.read_cache(rows.assign(remarks="changed"), "embeddings")
    with pytest.raises(ValueError, match="incomplete"):
        comparison.read_cache(rows.assign(ml_num="missing"), "embeddings")
    (tmp_path / "embedding_building.json").write_text("{}")
    with pytest.raises(ValueError, match="build is incomplete"):
        comparison.read_cache(rows, "embeddings")
    (tmp_path / "embedding_building.json").unlink()
    np.save(tmp_path / "embeddings.npy", np.array([[0, 0], [3, 4]], dtype=np.float32))
    with pytest.raises(ValueError, match="unwritten zero"):
        comparison.read_cache(rows, "embeddings")


def test_baseline_init_does_not_overwrite_an_existing_comparison(tmp_path, monkeypatch):
    monkeypatch.setattr(comparison, "OUT", tmp_path)
    manifest = tmp_path / "baseline_manifest.json"
    manifest.write_text('{"keep": true}')
    with pytest.raises(FileExistsError, match="already exists"):
        comparison.initialize_baseline()
    assert manifest.read_text() == '{"keep": true}'


def test_baseline_init_records_a_fresh_validation_and_inputs(tmp_path, monkeypatch):
    import hashlib
    import json
    import run
    monkeypatch.setattr(comparison, "OUT", tmp_path / "comparison")
    monkeypatch.setattr(run, "ROOT", tmp_path)
    dataset = tmp_path / "listings.parquet"
    dataset.write_bytes(b"test dataset")
    monkeypatch.setattr(comparison.data, "find_file", lambda name: dataset)
    calls = []

    def validate():
        calls.append(True)
        (tmp_path / "outputs").mkdir()
        (tmp_path / "outputs/validation_predictions.csv").write_text("fresh predictions\n")

    monkeypatch.setattr(run, "validate", validate)
    comparison.initialize_baseline()
    assert calls == [True]
    baseline = comparison.OUT / "baseline_predictions.csv"
    assert baseline.read_text() == "fresh predictions\n"
    manifest = json.loads((comparison.OUT / "baseline_manifest.json").read_text())
    assert manifest["sha256"][str(dataset)] == hashlib.sha256(dataset.read_bytes()).hexdigest()


def test_interrupted_embedding_rebuild_preserves_the_previous_vectors(tmp_path, monkeypatch):
    import sys
    from types import SimpleNamespace
    from experiments import prepare_remarks as preparation

    class FailingEncoder:
        calls = 0

        def get_sentence_embedding_dimension(self):
            return 2

        def encode(self, batch, **kwargs):
            self.calls += 1
            if self.calls == 2:
                raise RuntimeError("simulated interruption")
            return np.tile(np.array([1.0, 0.0], dtype=np.float32), (len(batch), 1))

    monkeypatch.setattr(preparation, "CACHE", tmp_path)
    monkeypatch.setattr(preparation, "model_path", lambda repo: (tmp_path, "test"))
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(
        backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: False)),
        set_num_threads=lambda n: None))
    monkeypatch.setitem(sys.modules, "sentence_transformers", SimpleNamespace(
        SentenceTransformer=lambda *args, **kwargs: FailingEncoder()))
    destination = tmp_path / "embeddings.npy"
    np.save(destination, np.array([[42.0, 43.0]], dtype=np.float32))
    previous = destination.read_bytes()
    homes = pd.DataFrame({"ml_num": [str(i) for i in range(1025)], "remarks": "A home."})
    with pytest.raises(RuntimeError, match="simulated interruption"):
        preparation.embeddings(homes)
    assert destination.read_bytes() == previous
    assert (tmp_path / "embedding_building.json").exists()


@pytest.mark.parametrize("output,expected", [("1", [1]), (" 3\n", [3]),
    ("0", [0]), ("4", [None]), ("not stated", [None]), ("13", [None])])
def test_llm_output_must_follow_the_schema(output, expected):
    assert parse_output(output) == expected


def test_checkpoint_recovers_only_an_unfinished_final_record(tmp_path):
    import json
    path = tmp_path / "cache.jsonl"
    path.write_bytes(b'{"ml_num":"a"}\n{"ml_num":')
    assert load_checkpoint(path) == [{"ml_num": "a"}]
    assert path.read_bytes() == b'{"ml_num":"a"}\n'
    path.write_bytes(b'{"ml_num":"a"}\ncorrupt\n')
    with pytest.raises(json.JSONDecodeError):
        load_checkpoint(path)


def test_fast_screen_is_bounded_time_ordered_and_price_independent():
    from experiments.fast_remarks import select_homes
    from run import VALIDATION_MONTHS
    dates = [pd.Timestamp("2025-01-01")] * 800
    for month in VALIDATION_MONTHS:
        dates += [month.start_time] * 120
    rows = pd.DataFrame({"ml_num": [str(i) for i in range(len(dates))],
                         "date_sold": dates,
                         "house_type_name": ["Condo Apt", "Detached"] * (len(dates) // 2),
                         "sold_price": np.arange(len(dates)) + 1})
    selected = select_homes(rows)
    changed = select_homes(rows.assign(sold_price=rows.sold_price.iloc[::-1].to_numpy()))
    assert len(selected) == 900
    assert selected.ml_num.equals(changed.ml_num)
    for i, month in enumerate(VALIDATION_MONTHS):
        train = selected[selected.date_sold < month.start_time]
        validation = selected[selected.sample_role == str(month)]
        assert len(train) == 600 + 50 * i
        assert len(validation) == 50
        assert set(train.ml_num).isdisjoint(validation.ml_num)
