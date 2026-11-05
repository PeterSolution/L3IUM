import numpy as np
import pandas as pd
import pytest

from data_prep import load_data, prepare_data


def make_df(n=100):
    rng = np.random.default_rng(0)
    df = pd.DataFrame(rng.normal(size=(n, 4)), columns=list("abcd"))
    df["target"] = [i % 2 for i in range(n)]
    return df


def test_split_sizes():
    df = make_df()
    X_train, X_test, y_train, y_test = prepare_data(df, test_size=0.2, seed=1)
    assert len(X_test) == pytest.approx(len(df) * 0.2, abs=1)


def test_sizes_sum_to_total():
    df = make_df()
    X_train, X_test, y_train, y_test = prepare_data(df, test_size=0.3, seed=1)
    assert len(X_train) + len(X_test) == len(df)
    assert len(y_train) + len(y_test) == len(df)


def test_empty_dataframe_raises():
    with pytest.raises(ValueError):
        prepare_data(pd.DataFrame())


def test_invalid_test_size_raises():
    with pytest.raises(ValueError):
        prepare_data(make_df(), test_size=1.5)


def test_missing_target_column_raises():
    with pytest.raises(ValueError):
        prepare_data(make_df().drop(columns=["target"]))


def test_same_seed_gives_same_split():
    df = make_df()
    first = prepare_data(df, test_size=0.2, seed=7)
    second = prepare_data(df, test_size=0.2, seed=7)
    assert first[1].index.equals(second[1].index)


def test_load_data_returns_dataframe(tmp_path):
    path = tmp_path / "dane.csv"
    make_df(10).to_csv(path, index=False)
    result = load_data(str(path))
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 10