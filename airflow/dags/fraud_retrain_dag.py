from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.dummy import DummyOperator
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import xgboost as xgb
import joblib
import json
import os
from scipy import stats

# Default arguments for the DAG
default_args = {
    'owner': 'fraud_team',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5)
}

# Paths (will be mounted as volumes in Docker)
DATA_PATH = '/opt/airflow/data/transactions.csv'
MODEL_PATH = '/opt/airflow/model/fraud_model.pkl'
BASELINE_PATH = '/opt/airflow/model/baseline_stats.json'

def extract_transactions(**context):
    """Extract today's transactions (simulated)"""
    np.random.seed(int(datetime.now().timestamp()))
    
    n_transactions = np.random.randint(8000, 12000)
    
    # Generate legitimate transactions
    n_legit = int(n_transactions * 0.98)
    legit_data = {
        'amount': np.random.exponential(60, n_legit),
        'hour': np.random.normal(14, 4, n_legit),
        'merchant_category': np.random.choice([0,1,2,3,4], n_legit),
        'distance_from_home': np.random.exponential(10, n_legit),
        'num_transactions_last_24h': np.random.poisson(3, n_legit),
        'fraud': 0
    }
    
    # Generate fraudulent transactions
    n_fraud = n_transactions - n_legit
    fraud_data = {
        'amount': np.random.exponential(300, n_fraud),
        'hour': np.random.choice([0,1,2,3,4,22,23], n_fraud),
        'merchant_category': np.random.choice([5,6,7,8,9], n_fraud),
        'distance_from_home': np.random.exponential(100, n_fraud),
        'num_transactions_last_24h': np.random.poisson(10, n_fraud),
        'fraud': 1
    }
    
    # Combine
    df_legit = pd.DataFrame(legit_data)
    df_fraud = pd.DataFrame(fraud_data)
    df = pd.concat([df_legit, df_fraud], ignore_index=True)
    
    # Shuffle
    df = df.sample(frac=1).reset_index(drop=True)
    
    # Save
    os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
    df.to_csv(DATA_PATH, index=False)
    
    print(f"Extracted {len(df)} transactions ({n_fraud} fraudulent)")
    return len(df)

def validate_data(**context):
    """Validate that we have enough data and it's clean"""
    df = pd.read_csv(DATA_PATH)
    
    # Check minimum rows
    if len(df) < 1000:
        raise ValueError(f"Not enough data: {len(df)} rows (minimum 1000)")
    
    # Check for missing values
    if df.isnull().any().any():
        raise ValueError("Missing values detected")
    
    # Check value ranges
    assert df['hour'].between(0, 23).all(), "Hour out of range"
    assert df['merchant_category'].between(0, 9).all(), "Category out of range"
    assert (df['amount'] >= 0).all(), "Negative amount"
    
    print(f"Data validation passed: {len(df)} rows")
    return len(df)

def check_data_drift(**context):
    """Check if data has drifted significantly from baseline"""
    
    # Load current data
    df_current = pd.read_csv(DATA_PATH)
    
    # Check if baseline exists
    if not os.path.exists(BASELINE_PATH):
        print("No baseline found - will train initial model")
        return 'preprocess'
    
    # Load baseline
    with open(BASELINE_PATH, 'r') as f:
        baseline = json.load(f)
    
    # Compare distributions for key features
    p_values = []
    
    # Kolmogorov-Smirnov test for continuous features
    for feature in ['amount', 'distance_from_home', 'num_transactions_last_24h']:
        if feature in baseline:
            # Generate baseline distribution parameters
            baseline_mean = baseline[f'{feature}_mean']
            baseline_std = baseline.get(f'{feature}_std', baseline_mean * 0.5)
            
            # Sample from current data
            current_sample = df_current[feature].sample(min(1000, len(df_current))).values
            
            # Generate baseline sample
            baseline_sample = np.random.normal(baseline_mean, baseline_std, 1000)
            
            # KS test
            ks_stat, p_value = stats.ks_2samp(current_sample, baseline_sample)
            p_values.append(p_value)
    
    # Check fraud rate
    current_fraud_rate = df_current['fraud'].mean()
    baseline_fraud_rate = baseline.get('fraud_rate', 0.02)
    
    fraud_rate_change = abs(current_fraud_rate - baseline_fraud_rate) / baseline_fraud_rate
    
    # Decision: drift if any p-value < 0.05 or fraud rate changed by > 20%
    DRIFT_P_VALUE = 0.05
    DRIFT_FRAUD_RATE = 0.2
    
    if any(p < DRIFT_P_VALUE for p in p_values) or fraud_rate_change > DRIFT_FRAUD_RATE:
        print(f"Data drift detected - retraining needed")
        print(f"Min p-value: {min(p_values):.4f}, Fraud rate change: {fraud_rate_change:.2%}")
        return 'preprocess'
    else:
        print(f"No significant drift - skipping retraining")
        print(f"Min p-value: {min(p_values):.4f}, Fraud rate change: {fraud_rate_change:.2%}")
        return 'skip_retrain'

def preprocess_data(**context):
    """Prepare data for training"""
    df = pd.read_csv(DATA_PATH)
    
    X = df.drop('fraud', axis=1)
    y = df['fraud']
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Push to XCom for next tasks
    context['ti'].xcom_push(key='X_train_shape', value=X_train.shape)
    context['ti'].xcom_push(key='X_test_shape', value=X_test.shape)
    
    # Save splits for later tasks
    os.makedirs('/opt/airflow/data/splits', exist_ok=True)
    X_train.to_csv('/opt/airflow/data/splits/X_train.csv', index=False)
    X_test.to_csv('/opt/airflow/data/splits/X_test.csv', index=False)
    pd.Series(y_train).to_csv('/opt/airflow/data/splits/y_train.csv', index=False)
    pd.Series(y_test).to_csv('/opt/airflow/data/splits/y_test.csv', index=False)
    
    return X_train.shape[0]

def train_model(**context):
    """Train the XGBoost model"""
    # Load splits
    X_train = pd.read_csv('/opt/airflow/data/splits/X_train.csv')
    y_train = pd.read_csv('/opt/airflow/data/splits/y_train.csv').squeeze()
    
    # Train
    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=42
    )
    model.fit(X_train, y_train)
    
    # Save model
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(model, MODEL_PATH + '.new')
    
    print(f"Model trained on {X_train.shape[0]} samples")

def evaluate_model(**context):
    """Evaluate the new model"""
    # Load splits
    X_test = pd.read_csv('/opt/airflow/data/splits/X_test.csv')
    y_test = pd.read_csv('/opt/airflow/data/splits/y_test.csv').squeeze()
    
    # Load model
    model = joblib.load(MODEL_PATH + '.new')
    
    # Predict
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_pred_proba)
    
    print(f"Test AUC: {auc:.4f}")
    
    # Check if model is good enough
    if auc < 0.9:
        raise ValueError(f"Model performance too low: AUC={auc:.4f}")
    
    context['ti'].xcom_push(key='auc', value=float(auc))
    return float(auc)

def update_baseline(**context):
    """Update baseline statistics with new data"""
    df = pd.read_csv(DATA_PATH)
    
    baseline_stats = {
        'amount_mean': float(df['amount'].mean()),
        'amount_std': float(df['amount'].std()),
        'hour_mean': float(df['hour'].mean()),
        'merchant_category_mean': float(df['merchant_category'].mean()),
        'distance_from_home_mean': float(df['distance_from_home'].mean()),
        'num_transactions_last_24h_mean': float(df['num_transactions_last_24h'].mean()),
        'fraud_rate': float(df['fraud'].mean())
    }
    
    with open(BASELINE_PATH, 'w') as f:
        json.dump(baseline_stats, f, indent=2)
    
    print("Baseline statistics updated")

def save_to_registry(**context):
    """Promote the new model to production"""
    # Replace old model with new one
    import shutil
    shutil.copy2(MODEL_PATH + '.new', MODEL_PATH)
    
    # Get metrics
    auc = context['ti'].xcom_pull(key='auc')
    
    print(f"Model promoted to production with AUC: {auc:.4f}")
    
    # In a real system, you would save to a model registry here
    # with metadata about the run

# Create the DAG
dag = DAG(
    'fraud_model_nightly_retrain',
    default_args=default_args,
    description='Nightly retraining pipeline with drift detection',
    schedule_interval=None,  # You'll change this later
    catchup=False,
    tags=['fraud', 'ml'],
)

# Define tasks
extract = PythonOperator(
    task_id='extract_transactions',
    python_callable=extract_transactions,
    dag=dag,
)

validate = PythonOperator(
    task_id='validate_data',
    python_callable=validate_data,
    dag=dag,
)

check_drift = BranchPythonOperator(
    task_id='check_data_drift',
    python_callable=check_data_drift,
    dag=dag,
)

preprocess = PythonOperator(
    task_id='preprocess',
    python_callable=preprocess_data,
    dag=dag,
)

train = PythonOperator(
    task_id='train_model',
    python_callable=train_model,
    dag=dag,
)

evaluate = PythonOperator(
    task_id='evaluate_model',
    python_callable=evaluate_model,
    dag=dag,
)

update_stats = PythonOperator(
    task_id='update_baseline',
    python_callable=update_baseline,
    dag=dag,
)

save = PythonOperator(
    task_id='save_to_registry',
    python_callable=save_to_registry,
    dag=dag,
)

skip = DummyOperator(
    task_id='skip_retrain',
    dag=dag,
)

# Set dependencies
extract >> validate >> check_drift
check_drift >> preprocess >> train >> evaluate >> update_stats >> save
check_drift >> skip