# run_pipeline.py

import sagemaker
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.steps import ProcessingStep, TrainingStep
from sagemaker.workflow.model_step import ModelStep
from sagemaker.processing import ScriptProcessor, ProcessingOutput
from sagemaker.sklearn.estimator import SKLearn
from sagemaker.workflow.parameters import ParameterString
from sagemaker.workflow.pipeline_context import PipelineSession
from sagemaker.model import Model
from sagemaker import image_uris
from pprint import pprint
import time
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Setup session and role
pipeline_session = PipelineSession()
role = 'arn:aws:iam::750952118292:role/service-role/AmazonSageMaker-ExecutionRole-20250615T204995'
region = pipeline_session.boto_region_name

# Parameters
model_package_group_name = ParameterString(name="ModelPackageGroupName", default_value="sklearn-demo-model-group")

# Ensure scripts exist
assert os.path.exists("generate_data.py"), "❌ generate_data.py not found"
assert os.path.exists("train.py"), "❌ train.py not found"

# Get image URI
image_uri = image_uris.retrieve("sklearn", region, version="1.0-1")

# Step 1: Processing (generate data)
logger.info("Setting up Processing step...")
processor = ScriptProcessor(
    image_uri=image_uri,
    command=["python3"],
    role=role,
    instance_count=1,
    instance_type="ml.t3.medium"
)

step_process = ProcessingStep(
    name="GenerateData",
    processor=processor,
    outputs=[
        ProcessingOutput(output_name="train", source="/opt/ml/processing/train"),
        ProcessingOutput(output_name="test", source="/opt/ml/processing/test"),
    ],
    code="generate_data.py"
)

# Step 2: Training
logger.info("Setting up Training step...")
sklearn_estimator = SKLearn(
    entry_point="train.py",
    role=role,
    instance_type="ml.m5.large",
    framework_version="1.0-1",
    sagemaker_session=pipeline_session,
)

step_train = TrainingStep(
    name="TrainModel",
    estimator=sklearn_estimator,
    inputs={
        "train": step_process.properties.ProcessingOutputConfig.Outputs["train"].S3Output.S3Uri
    },
)

# Step 3: Register model
logger.info("Setting up Model step...")
model = Model(
    image_uri=image_uri,
    model_data=step_train.properties.ModelArtifacts.S3ModelArtifacts,
    sagemaker_session=pipeline_session,
    role=role
)

step_model = ModelStep(
    name="RegisterModel",
    step_args=model.create(instance_type="ml.m5.large")
)

# Define pipeline
logger.info("Defining pipeline...")
pipeline = Pipeline(
    name="InlineTrainingDeploymentPipeline",
    parameters=[model_package_group_name],
    steps=[step_process, step_train, step_model],
    sagemaker_session=pipeline_session
)

# Deploy pipeline
logger.info("Uploading or updating pipeline...")
pipeline.upsert(role_arn=role)

logger.info("Starting pipeline execution...")
execution = pipeline.start()

logger.info("Waiting for pipeline to complete...")
execution.wait()

# Print pipeline status summary
logger.info("Pipeline execution summary:")
for step in execution.list_steps():
    print(f"🧱 Step: {step['StepName']}")
    print(f"   └ Status: {step['StepStatus']}")
    if 'FailureReason' in step:
        print(f"   ❌ Reason: {step['FailureReason']}\n")

# If it failed, exit early
status = execution.describe()['PipelineExecutionStatus']
if status == "Failed":
    raise RuntimeError("Pipeline failed. Check the logs above and AWS Console for more info.")

# Deploy the trained model
model_artifact_uri = execution.steps["TrainModel"]["Metadata"]["TrainingJob"]["ModelArtifacts"]["S3ModelArtifacts"]
logger.info(f"✅ Model artifact S3 URI: {model_artifact_uri}")

logger.info("Deploying model to a real-time endpoint...")
final_model = Model(
    model_data=model_artifact_uri,
    image_uri=image_uri,
    role=role,
    sagemaker_session=sagemaker.Session()
)

endpoint_name = f"sklearn-endpoint-{int(time.time())}"
predictor = final_model.deploy(
    initial_instance_count=1,
    instance_type="ml.m5.large",
    endpoint_name=endpoint_name
)

print(f"🚀 Model deployed successfully to endpoint: {endpoint_name}")
