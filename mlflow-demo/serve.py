import os
import sys
import json
import flask
import logging
import pandas as pd
import mlflow.pyfunc
from flask import Flask, request, jsonify

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

model = None

def load_model():
    """Load the MLflow model"""
    global model
    try:
        model_path = "/opt/ml/model"
        
        if os.path.exists(model_path):
            logger.info(f"Loading model from {model_path}")
            model = mlflow.pyfunc.load_model(model_path)
            logger.info("Model loaded successfully")
        else:
            logger.error(f"Model path {model_path} does not exist")
            raise Exception(f"Model not found at {model_path}")
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        raise e

@app.route('/ping', methods=['GET'])
def ping():
    try:
        if model is not None:
            return flask.Response(response=json.dumps({"status": "healthy"}), 
                                status=200, 
                                mimetype='application/json')
        else:
            return flask.Response(response=json.dumps({"status": "unhealthy", "reason": "model not loaded"}), 
                                status=503, 
                                mimetype='application/json')
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return flask.Response(response=json.dumps({"status": "unhealthy", "reason": str(e)}), 
                            status=503, 
                            mimetype='application/json')

@app.route('/invocations', methods=['POST'])
def invocations():
    try:
        content_type = request.content_type
        
        if content_type == 'application/json':
            data = request.get_json(force=True)
            
            if 'instances' in data:
                input_data = pd.DataFrame(data['instances'])
            elif 'data' in data:
                input_data = pd.DataFrame(data['data'])
            elif 'columns' in data and 'data' in data:
                input_data = pd.DataFrame(data['data'], columns=data['columns'])
            else:
                input_data = pd.DataFrame(data)
                
        elif content_type == 'text/csv':
            csv_data = request.data.decode('utf-8')
            from io import StringIO
            input_data = pd.read_csv(StringIO(csv_data))
        else:
            raise ValueError(f"Unsupported content type: {content_type}")
        
        logger.info(f"Making prediction on data shape: {input_data.shape}")
        prediction = model.predict(input_data)
        
        if hasattr(prediction, 'tolist'):
            result = prediction.tolist()
        else:
            result = list(prediction)
            
        response = {"predictions": result}
        
        return flask.Response(response=json.dumps(response), 
                            status=200, 
                            mimetype='application/json')
        
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        error_response = {"error": str(e)}
        return flask.Response(response=json.dumps(error_response), 
                            status=400, 
                            mimetype='application/json')

if __name__ == '__main__':
    load_model()
    
    port = int(os.environ.get('SAGEMAKER_BIND_TO_PORT', 8080))
    
    logger.info(f"Starting server on port {port}")
    app.run(host='0.0.0.0', port=port, debug=False)