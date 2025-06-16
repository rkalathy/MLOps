from mlflow.deployments import get_deploy_client
from mlflow.exceptions import MlflowException

# 1. Initialize the SageMaker client (region + role inferred from the URI)
client = get_deploy_client(
    "sagemaker:/eu-west-2/"
    "arn:aws:iam::750952118292:role/SageMakerExecutionRole"
)  # :contentReference[oaicite:0]{index=0}

# 2. Define your endpoint config
deployment_name = "ridge-e"
model_uri      = "models:/ridge-model-demo/5"
config = {
    "instance_type": "ml.m5.xlarge",
    "instance_count": 1,
    # (optional) you can also pass execution_role_arn here if you prefer:
    # "execution_role_arn": "arn:aws:iam::750952118292:role/SageMakerExecutionRole"
}

# 3. Create or update the SageMaker endpoint
try:
    client.create_deployment(
        name=model_name,
        model_uri=model_uri,
        config=config
    )
    print(f"✅ Created new endpoint: {model_name}")
except MlflowException as e:
    if "AlreadyExistsException" in str(e):
        client.update_deployment(
            name=deployment_name,
            model_uri=model_uri,
            config=config
        )
        print(f"🔄 Updated existing endpoint: {deployment_name}")
    else:
        raise
