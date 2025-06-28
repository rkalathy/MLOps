from azureml.core import Workspace, Model, Environment
from azureml.core.model import InferenceConfig
from azureml.core.webservice import AciWebservice

# 1. Load (or create) your workspace
ws = Workspace.from_config()  # expects config in ./aml_config.json

# 2. Register the MLflow model
model_uri = "runs:/<run_id>/rf_model"
mlflow_model = Model.register(
    workspace=ws,
    model_name="iris-rf-mlflow",
    model_path=mlflow.artifacts.download_artifacts(model_uri),
)

# 3. Define environment
env = Environment("iris-env")
deps = env.python.conda_dependencies
deps.add_conda_package("python=3.9")
deps.add_conda_package("scikit-learn")
deps.add_pip_package("mlflow")
deps.add_pip_package("pandas")
deps.add_pip_package("numpy")
deps.add_pip_package("cloudpickle")

# 4. Inference configuration
inference_config = InferenceConfig(
    entry_script="score.py",
    environment=env
)

# 5. Deployment config (ACI)
aci_conf = AciWebservice.deploy_configuration(
    cpu_cores=1, memory_gb=1
)

# 6. Deploy
service = Model.deploy(
    workspace=ws,
    name="iris-aci-service",
    models=[mlflow_model],
    inference_config=inference_config,
    deployment_config=aci_conf,
    overwrite=True
)
service.wait_for_deployment(show_output=True)
print("Scoring URI:", service.scoring_uri)
