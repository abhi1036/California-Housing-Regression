from sklearn.linear_model import LinearRegression

from utils import (
    load_data,
    split_data,
    save_model
)

# Load Dataset
df = load_data()

# Split
X_train, X_test, y_train, y_test = split_data(df)

# Train Model
model = LinearRegression()

model.fit(X_train, y_train)

# Save
save_model(
    model,
    list(X_train.columns)
)

print("Model trained successfully.")