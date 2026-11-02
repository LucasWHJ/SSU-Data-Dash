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


def test_cached_embeddings_are_aligned_by_id_and_reject_stale_text(tmp_path, monkeypatch):
    monkeypatch.setattr(comparison, "CACHE", tmp_path)
    rows = pd.DataFrame({"ml_num": ["a", "b"], "remarks": ["first", "second"]})
    pd.DataFrame({"ml_num": rows.ml_num, "position": [0, 1],
                  "text_hash": rows.remarks.map(comparison.text_hash)}).to_parquet(
                      tmp_path / "embedding_index.parquet")
    np.save(tmp_path / "embeddings.npy", np.array([[1, 2], [3, 4]], dtype=np.float32))
    np.testing.assert_array_equal(comparison.read_cache(rows.iloc[::-1], "embeddings"), [[3, 4], [1, 2]])
    with pytest.raises(ValueError, match="do not match"):
        comparison.read_cache(rows.assign(remarks="changed"), "embeddings")
    with pytest.raises(ValueError, match="incomplete"):
        comparison.read_cache(rows.assign(ml_num="missing"), "embeddings")


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
