import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor

# -----------------------------
# 1. Load dataset
# -----------------------------
df = pd.read_csv("ParisHousing.csv")

# CHANGE THIS if your target column is different
TARGET_COLUMN = "price"

X = df.drop(columns=[TARGET_COLUMN])
y = df[TARGET_COLUMN]

# -----------------------------
# 2. Train / test split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# 3. Fit scaler
# -----------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# -----------------------------
# 4. Train Random Forest
# -----------------------------
model = RandomForestRegressor(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train_scaled, y_train)

# -----------------------------
# 5. Save artifacts
# -----------------------------
joblib.dump(model, "random_forest_model.joblib")
joblib.dump(scaler, "scaler.joblib")

print("✅ random_forest_model.joblib and scaler.joblib saved successfully")
