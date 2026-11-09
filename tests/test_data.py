"""Data and leakage checks (run:  pytest)."""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
P = ROOT / "data" / "processed"
LEAKAGE = {"cd420", "cd820", "cd496", "r", "offtrt", "pidnum"}


def test_raw_shapes():
    assert pd.read_csv(ROOT / "data" / "raw" / "ACTG175_original.csv").shape == (2139, 27)
    assert pd.read_csv(ROOT / "data" / "raw" / "GBSG2_original.csv").shape == (686, 10)


def test_clean_file_has_no_leakage_columns_and_no_missing():
    d = pd.read_csv(P / "actg175_clean.csv")
    assert not (LEAKAGE & set(d.columns))
    assert int(d.isna().sum().sum()) == 0


def test_split_has_no_overlap_and_covers_everyone():
    ids = pd.read_csv(P / "split_ids.csv")
    tr, te = set(ids.loc[ids["set"] == "train", "row"]), set(ids.loc[ids["set"] == "test", "row"])
    assert not tr & te
    assert len(tr) + len(te) == 2139
    g = pd.read_csv(P / "gbsg2_split_ids.csv")
    gtr, gte = set(g.loc[g["set"] == "train", "row"]), set(g.loc[g["set"] == "test", "row"])
    assert not gtr & gte and len(gtr) + len(gte) == 686


def test_train_test_event_rates_are_similar():
    ytr, yte = pd.read_csv(P / "y_train.csv"), pd.read_csv(P / "y_test.csv")
    assert abs(ytr["cens"].mean() - yte["cens"].mean()) < 0.02


def test_features_contain_no_outcome_or_treatment_columns():
    cols = set(pd.read_csv(P / "X_train_trees.csv").columns)
    assert not ({"days", "cens", "arms", "treat"} & cols)
