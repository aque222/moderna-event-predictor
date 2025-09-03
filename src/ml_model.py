import yfinance as yf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# -------------------------------
# 1. Fetch Data
# -------------------------------
ticker = "MRNA"

df = yf.download(ticker, start="2015-01-01", end="2024-12-31")

print("Shape of dataframe:", df.shape)
print("First 5 rows before renaming:")
print(df.head())

# Flatten MultiIndex if needed
if isinstance(df.columns, pd.MultiIndex):
    df.columns = ["_".join(col).strip().lower().replace(" ", "_") for col in df.columns.values]
else:
    df.columns = [col.lower().replace(" ", "_") for col in df.columns]

print("Renamed columns:", df.columns)

# Pick a price column
price_candidates = [c for c in df.columns if "adj_close" in c or "close" in c]
if not price_candidates:
    raise ValueError("No price column found in dataframe!")
prices = df[price_candidates[0]]

df["return"] = prices.pct_change()
df["5d_ma"] = prices.rolling(window=5).mean()
df["10d_ma"] = prices.rolling(window=10).mean()
df["20d_vol"] = df["return"].rolling(window=20).std()

df = df.dropna()

# Target = 5-day forward return
df["target"] = prices.shift(-5) / prices - 1
df = df.dropna()

# -------------------------------
# 3. Train/Test Split
# -------------------------------
features = ["return", "5d_ma", "10d_ma", "20d_vol"]
X = df[features]
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False
)

# -------------------------------
# 4. Model Training
# -------------------------------
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# -------------------------------
# 5. Evaluation
# -------------------------------
y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse:.6f}")
print(f"R² Score: {r2:.6f}")

# -------------------------------
# 6. Plot Predicted vs Actual
# -------------------------------
plt.figure(figsize=(10, 6))
plt.plot(y_test.index, y_test, label="Actual", alpha=0.7)
plt.plot(y_test.index, y_pred, label="Predicted", alpha=0.7)
plt.legend()
plt.title(f"{ticker} – Predicted vs Actual 5-Day Returns")
plt.xlabel("Date")
plt.ylabel("Return")
plt.savefig("data/ml_predictions.png")
plt.close()

print("✅ Plot saved as data/ml_predictions.png")
