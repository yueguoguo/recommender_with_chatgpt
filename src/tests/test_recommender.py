import pytest
import pandas as pd
from main import (
    download_movielens_data,
    split_dataframe,
    train_sar_model,
    evaluate_sar_model,
)


def test_download_movielens_data():
    df = download_movielens_data()
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 100000
    assert set(df.columns) == set(["user_id", "item_id", "rating", "timestamp"])


def test_split_dataframe():
    df = pd.DataFrame(
        {
            "user_id": [1, 2, 3, 4, 5],
            "item_id": [10, 20, 30, 40, 50],
            "rating": [3, 4, 5, 3, 2],
        }
    )
    train_df, test_df = split_dataframe(df, split_ratio=0.8)
    assert isinstance(train_df, pd.DataFrame)
    assert isinstance(test_df, pd.DataFrame)
    assert len(train_df) + len(test_df) == len(df)
    assert set(train_df.columns) == set(test_df.columns) == set(df.columns)


def test_train_sar_model():
    train_df = pd.DataFrame(
        {
            "user_id": [1, 1, 1, 2, 2, 3, 3, 3, 3],
            "item_id": [10, 20, 30, 10, 20, 30, 40, 50, 60],
            "rating": [3, 4, 5, 3, 2, 4, 5, 4, 3],
        }
    )
    test_df = pd.DataFrame({"user_id": [1, 2, 3], "item_id": [40, 50, 60]})
    model = train_sar_model(train_df, test_df)
    assert isinstance(model, SARSingleNode)


def test_evaluate_sar_model():
    train_df = pd.DataFrame(
        {
            "user_id": [1, 1, 1, 2, 2, 3, 3, 3, 3],
            "item_id": [10, 20, 30, 10, 20, 30, 40, 50, 60],
            "rating": [3, 4, 5, 3, 2, 4, 5, 4, 3],
        }
    )
    test_df = pd.DataFrame({"user_id": [1, 2, 3], "item_id": [40, 50, 60]})
    model = train_sar_model(train_df, test_df)
    precision, recall = evaluate_sar_model(model, test_df)
    assert isinstance(precision, float)
    assert isinstance(recall, float)
    assert 0.0 <= precision <= 1.0
    assert 0.0 <= recall <= 1.0
