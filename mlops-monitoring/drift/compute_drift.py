import time
import threading
from prometheus_api_client import PrometheusConnect
from prometheus_client import Gauge, start_http_server
from scipy.stats import wasserstein_distance

# Expose drift gauge on port 8002
drift_gauge = Gauge('iris_data_drift_score', 'Data drift score')
start_http_server(8002)

# Initialize Prometheus client
prom = PrometheusConnect(url="http://prometheus:9090", disable_ssl=True)

# Mapping feature names to Prometheus histogram metrics
FEATURES = {
    'sepal_length': 'iris_sepal_length_bucket',
    'sepal_width':  'iris_sepal_width_bucket',
    'petal_length': 'iris_petal_length_bucket',
    'petal_width':  'iris_petal_width_bucket'
}

# Baseline distributions (bucket edges and counts) from training data
# Replace these with your actual train-time histograms
BASELINE = {
    feat: {
        'buckets': [i*0.5 for i in range(8, 19)],  # example edges: 4.0, 4.5, …, 9.0
        'counts':  [1.0] * 11                      # uniform counts for example
    }
    for feat in FEATURES
}

def fetch_histogram(metric_name):
    """Fetch current Prometheus histogram buckets and counts."""
    results = prom.get_current_metric_value(metric_name)
    buckets, counts = [], []
    for sample in results:
        le  = float(sample['metric']['le'])
        val = float(sample['value'][1])
        buckets.append(le)
        counts.append(val)
    return buckets, counts

def compute_drift_score():
    """Compute average Wasserstein distance across all features."""
    scores = []
    for feat, metric in FEATURES.items():
        base_buckets = BASELINE[feat]['buckets']
        base_counts  = BASELINE[feat]['counts']
        curr_buckets, curr_counts = fetch_histogram(metric)
        score = wasserstein_distance(
            u_values=base_buckets,
            v_values=curr_buckets,
            u_weights=base_counts,
            v_weights=curr_counts
        )
        scores.append(score)
    return sum(scores) / len(scores)

def drift_loop():
    """Background loop: compute & set drift score every 5 minutes."""
    while True:
        try:
            score = compute_drift_score()
            drift_gauge.set(score)
            print(f"Set drift score: {score}")
        except Exception as e:
            print(f"Error computing drift: {e}")
        time.sleep(300)  # wait 5 minutes

# Start drift computation thread
threading.Thread(target=drift_loop, daemon=True).start()

# Keep the main thread alive
while True:
    time.sleep(60)
