"""
Predictive Analytics: Forecasting Future Sales Trends using Historical Data
Uses feature-engineered regression (trend + seasonality) to forecast sales,
and evaluates with train/test split + accuracy metrics.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ---------- Load & clean data ----------
df = pd.read_csv("data/historical_sales.csv", parse_dates=["Date"])
df = df.dropna().sort_values("Date").reset_index(drop=True)

# ---------- Feature engineering ----------
df["t"] = np.arange(len(df))                       # linear trend
df["dayofweek"] = df["Date"].dt.dayofweek
df["month"] = df["Date"].dt.month
df["dayofyear"] = df["Date"].dt.dayofyear
df["sin_year"] = np.sin(2 * np.pi * df["dayofyear"] / 365.25)
df["cos_year"] = np.cos(2 * np.pi * df["dayofyear"] / 365.25)
df["sin_week"] = np.sin(2 * np.pi * df["dayofweek"] / 7)
df["cos_week"] = np.cos(2 * np.pi * df["dayofweek"] / 7)

features = ["t", "dayofweek", "month", "sin_year", "cos_year", "sin_week", "cos_week"]
X = df[features]
y = df["Sales"]

# ---------- Train/test split (last 90 days = test) ----------
split = len(df) - 90
X_train, X_test = X.iloc[:split], X.iloc[split:]
y_train, y_test = y.iloc[:split], y.iloc[split:]
dates_test = df["Date"].iloc[split:]

# ---------- Models ----------
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=200, random_state=42),
}

results = {}
preds = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    preds[name] = pred
    results[name] = {
        "MAE": mean_absolute_error(y_test, pred),
        "RMSE": mean_squared_error(y_test, pred) ** 0.5,
        "R2": r2_score(y_test, pred),
    }

results_df = pd.DataFrame(results).T.round(2)
print("Model Evaluation:\n", results_df)
results_df.to_csv("outputs/model_evaluation.csv")

best_model_name = results_df["RMSE"].idxmin()
print(f"\nBest model: {best_model_name}")

# ---------- Plot actual vs predicted (test period) ----------
plt.figure(figsize=(10, 5))
plt.plot(dates_test, y_test, label="Actual", linewidth=2)
for name, pred in preds.items():
    plt.plot(dates_test, pred, label=f"Predicted ({name})", linestyle="--")
plt.title("Actual vs Predicted Sales (Last 90 Days)")
plt.xlabel("Date")
plt.ylabel("Sales")
plt.legend()
plt.tight_layout()
plt.savefig("outputs/actual_vs_predicted.png")
plt.close()

# ---------- Forecast next 30 days with best model ----------
best_model = models[best_model_name]
future_dates = pd.date_range(df["Date"].max() + pd.Timedelta(days=1), periods=30)
future = pd.DataFrame({"Date": future_dates})
future["t"] = np.arange(len(df), len(df) + 30)
future["dayofweek"] = future["Date"].dt.dayofweek
future["month"] = future["Date"].dt.month
future["dayofyear"] = future["Date"].dt.dayofyear
future["sin_year"] = np.sin(2 * np.pi * future["dayofyear"] / 365.25)
future["cos_year"] = np.cos(2 * np.pi * future["dayofyear"] / 365.25)
future["sin_week"] = np.sin(2 * np.pi * future["dayofweek"] / 7)
future["cos_week"] = np.cos(2 * np.pi * future["dayofweek"] / 7)

future["ForecastSales"] = best_model.predict(future[features])
future[["Date", "ForecastSales"]].to_csv("outputs/next_30_day_forecast.csv", index=False)

plt.figure(figsize=(10, 5))
plt.plot(df["Date"].iloc[-180:], df["Sales"].iloc[-180:], label="Historical (last 180 days)")
plt.plot(future["Date"], future["ForecastSales"], label="Forecast (next 30 days)", linestyle="--", color="red")
plt.title(f"30-Day Sales Forecast ({best_model_name})")
plt.xlabel("Date")
plt.ylabel("Sales")
plt.legend()
plt.tight_layout()
plt.savefig("outputs/forecast_next_30_days.png")
plt.close()

print("\nDone. Outputs saved in outputs/")
