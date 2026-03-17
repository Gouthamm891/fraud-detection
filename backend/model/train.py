import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import xgboost as xgb
import joblib
import json
import os

# Generate synthetic data
np.random.seed(42)
n_samples = 10000

# Create features
data = {
    'amount': np.random.exponential(100, n_samples),
    'hour': np.random.randint(0, 24, n_samples),
    'merchant_category': np.random.randint(0, 10, n_samples),
    'distance_from_home': np.random.exponential(20, n_samples),
    'num_transactions_last_24h': np.random.poisson(5, n_samples)
}

df = pd.DataFrame(data)

# Create synthetic target (fraud)
# Fraud is more likely with high amount, late hour, high-risk category, etc.
score = (
    (df['amount'] > 200) * 0.3 +
    ((df['hour'] < 6) | (df['hour'] > 22)) * 0.2 +
    (df['merchant_category'] > 7) * 0.3 +
    (df['distance_from_home'] > 50) * 0.2 +
    (df['num_transactions_last_24h'] > 10) * 0.2
)
prob_fraud = 1 / (1 + np.exp(-(score - 0.5)))
df['fraud'] = np.random.binomial(1, prob_fraud)

# Split data
X = df.drop('fraud', axis=1)
y = df['fraud']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    random_state=42
)
model.fit(X_train, y_train)

# Evaluate
y_pred_proba = model.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test, y_pred_proba)
print(f"Test AUC: {auc:.4f}")

# Save model
os.makedirs('model', exist_ok=True)
joblib.dump(model, 'model/fraud_model.pkl')

# Save baseline statistics for drift detection
baseline_stats = {
    'amount_mean': float(df['amount'].mean()),
    'amount_std': float(df['amount'].std()),
    'hour_mean': float(df['hour'].mean()),
    'merchant_category_mean': float(df['merchant_category'].mean()),
    'distance_from_home_mean': float(df['distance_from_home'].mean()),
    'num_transactions_last_24h_mean': float(df['num_transactions_last_24h'].mean()),
    'fraud_rate': float(df['fraud'].mean())
}

with open('model/baseline_stats.json', 'w') as f:
    json.dump(baseline_stats, f, indent=2)

print("Model and baseline statistics saved successfully!")