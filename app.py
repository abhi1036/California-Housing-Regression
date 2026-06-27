from flask import Flask, render_template, request
import joblib

from src.predictor import predict_house_price
from src.config import METRICS_PATH

app = Flask(__name__)

# Load metrics once when the app starts
metrics = joblib.load(METRICS_PATH)

MODEL_NAME = "Linear Regression"


@app.route("/")
def home():
    return render_template(
        "index.html",
        model_name=MODEL_NAME,
        metrics=metrics
    )


@app.route("/predict", methods=["POST"])
def predict():

    features = {
        "MedInc": float(request.form["MedInc"]),
        "HouseAge": float(request.form["HouseAge"]),
        "AveRooms": float(request.form["AveRooms"]),
        "AveBedrms": float(request.form["AveBedrms"]),
        "Population": float(request.form["Population"]),
        "AveOccup": float(request.form["AveOccup"]),
        "Latitude": float(request.form["Latitude"]),
        "Longitude": float(request.form["Longitude"]),
    }

    prediction = predict_house_price(features)

    return render_template(
        "index.html",
        prediction=f"${prediction:,.2f}",
        model_name=MODEL_NAME,
        metrics=metrics
    )


if __name__ == "__main__":
    app.run(debug=True)