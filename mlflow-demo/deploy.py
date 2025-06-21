import os
import mlflow
import mlflow.deployments
import dagshub

dagshub.init(repo_owner='frontenddevtrainer', repo_name='MLOps', mlflow=True)

def main():
    # 1. Point at your MLflow model
    model_uri = os.environ.get("MODEL_URI", "runs:/5604241595be4a7f9080cb40a321d946/rf_model")
    
    # 2. Your SageMaker execution role
    role = "arn:aws:iam::750952118292:role/service-role/AmazonSageMaker-ExecutionRole-20250615T204995"
    
    # 3. Create deployment client
    client = mlflow.deployments.get_deploy_client("sagemaker:/eu-west-2")
    # Change to update_deployment deployment.

    deployment = client.create_deployment(
        name="iris-demo-endpoint-demo",
        model_uri=model_uri,
        flavor="python_function",
        config={
            "execution_role_arn": role,
            "instance_type": "ml.t2.medium",
            "instance_count": 1,
            "region_name": "eu-west-2",
            "model_data_bucket": "my-mlflow-bucket"
        }
    )

    print("Endpoint deployed:", deployment)

if __name__ == "__main__":
    main()

