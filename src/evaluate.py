from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from utils import (
    load_data,
    split_data,
    load_model
)

df = load_data()

X_train, X_test, y_train, y_test = split_data(df)

model = load_model()

predictions = model.predict(X_test)

print(f"MAE  : {mean_absolute_error(y_test, predictions):.4f}")
print(f"MSE  : {mean_squared_error(y_test, predictions):.4f}")
print(f"RMSE : {mean_squared_error(y_test, predictions, squared=False):.4f}")
print(f"R²   : {r2_score(y_test, predictions):.4f}")