import os, json
import numpy as np
from mlflow.pyfunc import load_model

def init():
    global model
    # AzureML mounts your MLflow model under AZUREML_MODEL_DIR
    model_path = os.path.join(os.environ["AZUREML_MODEL_DIR"], "rf_model")
    model = load_model(model_path)

def run(raw_data):
    data = np.array(json.loads(raw_data)["instances"])
    preds = model.predict(data).tolist()
    return json.dumps({"predictions": preds})
