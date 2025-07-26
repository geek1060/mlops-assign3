import joblib
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

model = joblib.load("model.joblib")

X, y = fetch_california_housing(return_X_y=True)
_, X_test, _, _ = train_test_split(X, y, test_size=0.2, random_state=42)

y_pred = model.predict(X_test[:5])
print("Sample Predictions:", y_pred)
