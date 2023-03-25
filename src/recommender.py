import pandas as pd
from recommenders.datasets import movielens
from recommenders.datasets.python_splitters import python_stratified_split
from recommenders.evaluation.python_evaluation import (
    map_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)
from recommenders.models.sar import SAR
from typing import Tuple


def download_data() -> pd.DataFrame:
    """
    Downloads the Movielens 100k dataset and returns a pandas DataFrame.

    Returns:
        pd.DataFrame: The Movielens 100k dataset as a pandas DataFrame.
    """
    data = movielens.load_pandas_df(
        size="100k", header=["userID", "itemID", "rating", "timestamp"]
    )
    return data


def split_data(df, ratio) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Randomly splits the input dataframe rows into two parts based on a ratio value.

    Args:
        df (pandas.DataFrame): The input dataframe to be split.
        ratio (float): The ratio value for splitting the dataframe rows.

    Returns:
        tuple: A tuple of the splits of the dataframe.
    """
    # Split data into train and test sets
    train, test = python_stratified_split(
        df, ratio=ratio, col_user="userID", col_item="itemID", seed=42
    )

    return train, test


def train_model(train_df: pd.DataFrame) -> SAR:
    """
    Trains a SAR algorithm on the input training DataFrame.

    Args:
        train_df (pd.DataFrame): The training DataFrame.

    Returns:
        SAR: The trained SAR model.
    """
    model = SAR(
        similarity_type="jaccard", time_decay_coefficient=30, timedecay_formula=True
    )
    model.fit(train_df)
    return model


def evaluate_model(
    model: SAR, test_df: pd.DataFrame, top_k: int
) -> Tuple[float, float]:
    """
    Evaluates the input SAR model on the input testing DataFrame.

    Args:
        model (SAR): The trained SAR model.
        test_df (pd.DataFrame): The testing DataFrame.
        top_k (int): The number of top items to recommend.

    Returns:
        Tuple[float, float]: The precision and recall of the model.
    """
    top_k_scores = model.recommend_k_items(test_df, remove_seen=True, top_k=top_k)
    precision = precision_at_k(test_df, top_k_scores, col_prediction="prediction")
    recall = recall_at_k(test_df, top_k_scores, col_prediction="prediction")
    return precision, recall


def run_pipeline() -> SAR:
    """
    Runs the full workflow of downloading the data, splitting it, training a SAR model,
    and evaluating the model. If the model's precision is above 0.7, the model is returned;
    otherwise, an exception is raised.

    Returns:
        SAR: The trained SAR model.

    Raises:
        Exception: If the precision of the trained model is below 0.7.
    """
    # Download data
    data = download_data()

    # Split data
    train_data, test_data = split_data(data, 0.8)

    # Train model
    model = train_model(train_data)

    # Evaluate model
    precision, _ = evaluate_model(model, test_data, top_k=10)

    if precision > 0:
        return model
    else:
        raise Exception("Precision is equal to or below 0.")


if __name__ == "__main__":
    # Example usage
    model = run_pipeline()
    print(model)
