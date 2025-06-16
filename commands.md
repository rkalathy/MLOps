Hi,

Due to updates required in the libraries and the demo code, more time is needed. Please check back on Wednesday, 18 June, at 2 pm IST.

Apologies for the delay.


--- Install MLFLow

pip install mlflow

pip freeze > requirements.txt

python .\ml-loan-demo\loan-model.py

--- Run MLFLow UI 

mlflow ui

Navigate to - http://127.0.0.1:5000/


mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root s3://mlflow-artifacts-RANDOMTEXT --host 0.0.0.0 --port 5000