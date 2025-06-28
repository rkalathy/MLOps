from flask import Flask, request, jsonify
import pickle
import time
from prometheus_client import start_http_server, Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST

# Load model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

app = Flask(__name__)

# Metrics
total_requests = Counter('iris_requests_total', 'Total inference requests')
latency = Histogram('iris_inference_seconds', 'Inference latency')
predictions = Counter('iris_predictions_total', 'Prediction counts', ['predicted_class'])

# Histograms for feature distributions (for drift monitoring)
sepal_length = Histogram('iris_sepal_length', 'Sepal length', buckets=[i*0.5 for i in range(8, 19)])
sepal_width = Histogram('iris_sepal_width', 'Sepal width', buckets=[i*0.25 for i in range(8, 21)])
petal_length = Histogram('iris_petal_length', 'Petal length', buckets=[i*0.5 for i in range(2, 17)])
petal_width = Histogram('iris_petal_width', 'Petal width', buckets=[i*0.25 for i in range(2, 21)])

drift_score = Gauge('iris_data_drift_score', 'Data drift score')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json['features']  # expects list of 4 floats
    total_requests.inc()
    start = time.time()
    pred = int(model.predict([data])[0])
    latency.observe(time.time() - start)
    predictions.labels(predicted_class=str(pred)).inc()
    sepal_length.observe(data[0])
    sepal_width.observe(data[1])
    petal_length.observe(data[2])
    petal_width.observe(data[3])
    return jsonify({'prediction': pred})

@app.route('/metrics')
def metrics():
    return generate_latest(), 200, {'Content-Type': CONTENT_TYPE_LATEST}

if __name__ == '__main__':
    # Expose Prometheus metrics on port 8001
    start_http_server(8001)
    app.run(host='0.0.0.0', port=5000)