import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib

# Sample dummy data
data = pd.DataFrame({
    "Disease_Index": [0.3, 0.4, 0.5],
    "Infra_Gap": [1, 2, 3],
    "Population": [5, 8, 12],
    "Budget": [1000, 2000, 3000]
})

X = data[["Disease_Index", "Infra_Gap", "Population"]]
y = data["Budget"]

model = LinearRegression()
model.fit(X, y)

joblib.dump(model, "budget_predictor.pkl")

print("Budget model saved.")
