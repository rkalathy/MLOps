import time
import threading
from prometheus_api_client import PrometheusConnect
from prometheus_client import Gauge, start_http_server
from scipy.stats import wasserstein_distance
from sklearn.datasets import load_iris
import numpy as np

# Expose drift gauge on port 8002
drift_gauge = Gauge('iris_data_drift_score', 'Data drift score')
start_http_server(8002)

# Initialize Prometheus client
prom = PrometheusConnect(url="http://prometheus:9090", disable_ssl=True)

# Load Iris dataset for baseline
data = load_iris()
features = data.data
feature_names = data.feature_names

# Define bucket edges matching app.py instrumentation
FEATURE_BUCKETS = {
    'sepal length (cm)': [i*0.5 for i in range(8,19)],
    'sepal width (cm)':  [i*0.25 for i in range(8,21)],
    'petal length (cm)': [i*0.5 for i in range(2,17)],
    'petal width (cm)':  [i*0.25 for i in range(2,21)]
}

# Compute baseline histograms and normalize weights
BASELINE = {}
for idx, name in enumerate(feature_names):
    vals = features[:, idx]
    edges = FEATURE_BUCKETS[name]
    counts, bin_edges = np.histogram(vals, bins=[*edges, np.inf])
    bucket_rights = bin_edges[1:].tolist()
    weights = (counts / counts.sum()).tolist()
    BASELINE[name] = {'buckets': bucket_rights, 'weights': weights}

# Map metric names to dataset feature names
METRIC_MAP = {
    'sepal_length': 'iris_sepal_length_bucket',
    'sepal_width':  'iris_sepal_width_bucket',
    'petal_length': 'iris_petal_length_bucket',
    'petal_width':  'iris_petal_width_bucket'
}

# Fetch current histogram and normalize
def fetch_histogram(metric_name):
    results = prom.get_current_metric_value(metric_name)
    pairs = [(float(s['metric']['le']), float(s['value'][1])) for s in results]
    pairs.sort(key=lambda x: x[0])
    buckets, counts = zip(*pairs)
    weights = (np.array(counts) / np.sum(counts)).tolist()
    return list(buckets), weights

# Compute average Wasserstein distance across features
def compute_drift_score():
    scores = []
    for key, metric in METRIC_MAP.items():
        feature_key = key.replace('_', ' ')
        base = BASELINE[feature_key]
        curr_buckets, curr_weights = fetch_histogram(metric)
        score = wasserstein_distance(
            u_values=base['buckets'],
            v_values=curr_buckets,
            u_weights=base['weights'],
            v_weights=curr_weights
        )
        scores.append(score)
    return sum(scores) / len(scores)

# Background loop to update gauge every minute
def drift_loop():
    while True:
        try:
            score = compute_drift_score()
            drift_gauge.set(score)
            print(f"Drift score: {score}")
        except Exception as e:
            print(f"Error computing drift: {e}")
        time.sleep(60)

threading.Thread(target=drift_loop, daemon=True).start()

# Keep container alive
while True:
    time.sleep(60)