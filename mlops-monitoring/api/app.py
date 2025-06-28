from flask import Flask, request, jsonify
import pickle
import time
from prometheus_client import start_http_server, Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST

# Load model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

app = Flask(__name__)

# Prometheus metrics
total_requests = Counter('iris_requests_total', 'Total inference requests')
latency_hist = Histogram('iris_inference_seconds', 'Inference latency (seconds)')
class_counter = Counter('iris_predictions_total', 'Prediction counts', ['predicted_class'])
sepal_length_hist = Histogram('iris_sepal_length', 'Sepal length distribution', buckets=[i*0.5 for i in range(8, 19)])
sepal_width_hist = Histogram('iris_sepal_width', 'Sepal width distribution', buckets=[i*0.25 for i in range(12, 25)])

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json.get('features')
    total_requests.inc()
    start = time.time()
    pred = model.predict([data])[0]
    latency_hist.observe(time.time() - start)
    class_counter.labels(predicted_class=str(pred)).inc()
    sepal_length_hist.observe(data[0])
    sepal_width_hist.observe(data[1])
    return jsonify({'prediction': int(pred)})

@app.route('/metrics')
def metrics():
    return generate_latest(), 200, {'Content-Type': CONTENT_TYPE_LATEST}

if __name__ == '__main__':
    start_http_server(8001)
    app.run(host='0.0.0.0', port=5000)
