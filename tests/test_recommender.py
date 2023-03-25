import pytest
import pandas as pd
from src.recommender import (
    download_data,
    split_data,
    train_model,
    evaluate_model,
)
from recommenders.models.sar import SAR


def test_download_movielens_data():
    df = download_data()
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 100000
    assert set(df.columns) == set(["userID", "itemID", "rating", "timestamp"])


def test_split_dataframe():
    df = pd.DataFrame(
        {
            "userID": [1, 2, 3, 4, 5],
            "itemID": [10, 20, 30, 40, 50],
            "rating": [3, 4, 5, 3, 2],
        }
    )
    train_df, test_df = split_data(df, ratio=0.8)
    assert isinstance(train_df, pd.DataFrame)
    assert isinstance(test_df, pd.DataFrame)
    assert len(train_df) + len(test_df) == len(df)
    assert set(train_df.columns) == set(test_df.columns) == set(df.columns)


def test_train_sar_model():
    train_df = pd.DataFrame(
        {
            "userID": [1, 1, 1, 2, 2, 3, 3, 3, 3],
            "itemID": [10, 20, 30, 10, 20, 30, 40, 50, 60],
            "rating": [3, 4, 5, 3, 2, 4, 5, 4, 3],
            "timestamp": [
                881250949,
                881250949,
                881250949,
                881250949,
                881250949,
                881250949,
                881250949,
                881250949,
                881250949,
            ],
        }
    )
    test_df = pd.DataFrame({"userID": [1, 2, 3], "itemID": [40, 50, 60]})
    model = train_model(train_df)
    assert isinstance(model, SAR)


def test_evaluate_sar_model():
    train_df = pd.DataFrame(
        {
            "userID": [1, 1, 1, 2, 2, 3, 3, 3, 3],
            "itemID": [10, 20, 30, 10, 20, 30, 40, 50, 60],
            "rating": [3, 4, 5, 3, 2, 4, 5, 4, 3],
            "timestamp": [
                881250949,
                881250949,
                881250949,
                881250949,
                881250949,
                881250949,
                881250949,
                881250949,
                881250949,
            ],
        }
    )
    test_df = pd.DataFrame(
        {"userID": [1, 2, 3], "itemID": [40, 50, 60], "rating": [1, 1, 1]}
    )
    model = train_model(train_df)
    precision, recall = evaluate_model(model, test_df, 10)
    assert isinstance(precision, float)
    assert isinstance(recall, float)
    assert 0.0 <= precision <= 1.0
    assert 0.0 <= recall <= 1.0
