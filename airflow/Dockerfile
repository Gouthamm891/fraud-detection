FROM apache/airflow:2.7.1-python3.11

USER root
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    gcc \
    && apt-get autoremove -y \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

USER airflow
RUN pip install --no-cache-dir \
    scikit-learn==1.3.2 \
    xgboost==2.0.0 \
    pandas==2.1.3 \
    numpy==1.24.3 \
    joblib==1.3.2 \
    scipy==1.11.4