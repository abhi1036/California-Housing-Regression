import pandas as pd

from utils import load_model

model = load_model()

sample = pd.DataFrame([{
    "MedInc": 8.3252,
    "HouseAge": 41,
    "AveRooms": 6.984127,
    "AveBedrms": 1.02381,
    "Population": 322,
    "AveOccup": 2.555556,
    "Latitude": 37.88,
    "Longitude": -122.23
}])

prediction = model.predict(sample)

print(f"Predicted House Value: {prediction[0]:.3f}")