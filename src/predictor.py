import joblib
import pandas as pd

from src.config import MODEL_PATH

model = joblib.load(MODEL_PATH)


def predict_house_price(features: dict):
    """
    Predict house price from input features.

    Parameters
    ----------
    features : dict

    Returns
    -------
    float
    """

    df = pd.DataFrame([features])

    prediction = model.predict(df)[0]

    return prediction * 100000