import joblib
import pandas as pd

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split


def load_data():
    """
    Load California Housing Dataset
    """

    housing = fetch_california_housing()

    df = pd.DataFrame(
        housing.data,
        columns=housing.feature_names
    )

    df["HouseValue"] = housing.target

    return df


def split_data(df):

    X = df.drop("HouseValue", axis=1)
    y = df["HouseValue"]

    return train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )


def save_model(model, feature_names):

    joblib.dump(
        model,
        "models/linear_regression.pkl"
    )

    joblib.dump(
        feature_names,
        "models/features.pkl"
    )


def load_model():

    return joblib.load(
        "models/linear_regression.pkl"
    )