# =========================
# Smart HR - Model Training
# Features: years_experience + skill_count + job_state (One-Hot)
# =========================

import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# =========================
# 1️⃣ Load processed data
# =========================
df = pd.read_csv("data/processed/processed_jobs.csv")

# =========================
# 2️⃣ Feature Engineering
# =========================

# ---- Use real years of experience ----
# Make sure 'years_experience' column exists in CSV
# If not, you can compute from seniority as a fallback
if 'years_experience' not in df.columns:
    df['years_experience'] = df['seniority'].map({0: 1, 1: 6})

# ---- Skill count ----
skill_cols = ['python', 'excel', 'hadoop', 'spark', 'aws', 'tableau', 'big_data']
df['skill_count'] = df[skill_cols].sum(axis=1)

# ---- One-Hot for job_state ----
df = pd.get_dummies(df, columns=['job_state'], drop_first=False)

# ---- Features & Target ----
feature_cols = ['years_experience', 'skill_count'] + [c for c in df.columns if c.startswith('job_state_')]
X = df[feature_cols]
y = df['avg_salary']

# =========================
# 3️⃣ Train / Test Split
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =========================
# 4️⃣ Train Linear Regression
# =========================
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

y_pred_lr = lr_model.predict(X_test)

mae_lr = mean_absolute_error(y_test, y_pred_lr)
rmse_lr = np.sqrt(mean_squared_error(y_test, y_pred_lr))
r2_lr = r2_score(y_test, y_pred_lr)

print("Linear Regression Performance:")
print(f"MAE: {mae_lr:.2f}")
print(f"RMSE: {rmse_lr:.2f}")
print(f"R2: {r2_lr:.2f}")

# Save feature names for preprocessing in API
lr_model.feature_names_in_ = X_train.columns

# =========================
# 5️⃣ Train Random Forest
# =========================
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

y_pred_rf = rf_model.predict(X_test)

mae_rf = mean_absolute_error(y_test, y_pred_rf)
rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))
r2_rf = r2_score(y_test, y_pred_rf)

print("\nRandom Forest Performance:")
print(f"MAE: {mae_rf:.2f}")
print(f"RMSE: {rmse_rf:.2f}")
print(f"R2: {r2_rf:.2f}")

# Save feature names
rf_model.feature_names_in_ = X_train.columns

# =========================
# 6️⃣ Save Best Model
# =========================
best_model = rf_model if r2_rf > r2_lr else lr_model
joblib.dump(best_model, "backend/models/model.pkl")

print("\n Best model saved as 'model.pkl'")
