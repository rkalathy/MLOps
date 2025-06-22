import boto3
import sagemaker
from sagemaker.session import Session
import pandas as pd

from sagemaker.feature_store.feature_definition import FeatureDefinition, FeatureTypeEnum
from sagemaker.feature_store.feature_group import FeatureGroup
import time


region = "eu-west-2"
boto_session = boto3.Session(region_name=region)
sagemaker_client = boto_session.client("sagemaker")
sagemaker_session = sagemaker.Session(boto_session=boto_session)
role = "arn:aws:iam::750952118292:role/service-role/AmazonSageMaker-ExecutionRole-20250615T204995"
bucket_name = "sagemaker-features-demo" 
feature_group_name = "customers-feature-group-88"
record_identifier_name = "customer_id"
event_time_feature_name = "signup_date"


df = pd.read_csv("sagemaker-feature-store/customers.csv")


feature_definitions = [
    FeatureDefinition(feature_name="customer_id", feature_type=FeatureTypeEnum.INTEGRAL),
    FeatureDefinition(feature_name="first_name", feature_type=FeatureTypeEnum.STRING),
    FeatureDefinition(feature_name="last_name", feature_type=FeatureTypeEnum.STRING),
    FeatureDefinition(feature_name="signup_date", feature_type=FeatureTypeEnum.STRING),
    FeatureDefinition(feature_name="age", feature_type=FeatureTypeEnum.INTEGRAL),
]

feature_group = FeatureGroup(
    name=feature_group_name,
    sagemaker_session=sagemaker_session
)

try:
    sagemaker_client.describe_feature_group(FeatureGroupName=feature_group_name)
    print(f" Feature group '{feature_group_name}' already exists. Skipping creation.")
except sagemaker_client.exceptions.ResourceNotFound:
    sagemaker_client.create_feature_group(
        FeatureGroupName=feature_group_name,
        RecordIdentifierFeatureName=record_identifier_name,
        EventTimeFeatureName=event_time_feature_name,
        FeatureDefinitions=[fd.to_dict() for fd in feature_definitions],
        RoleArn=role,
        OnlineStoreConfig={"EnableOnlineStore": True},
        OfflineStoreConfig={
            "S3StorageConfig": {
                "S3Uri": f"s3://{bucket_name}/feature-store/{feature_group_name}/"
            },
            "DisableGlueTableCreation": False
        }
    )


if __name__ == "__main__":
    while True:
        status = sagemaker_client.describe_feature_group(FeatureGroupName=feature_group_name)["FeatureGroupStatus"]
        if status == "Created":
            print("Feature group ready.")
            break
        elif status == "CreateFailed":
            raise Exception("Feature group creation failed.")
        time.sleep(5)

    df["signup_date"] = pd.to_datetime(df["signup_date"]).dt.strftime("%Y-%m-%dT%H:%M:%S.%fZ")

    print("Ingesting to feature store...")
    feature_group.ingest(data_frame=df, max_workers=3, wait=True)
    status = feature_group.ingest(data_frame=df, max_workers=3, wait=True)
    print("Ingestion result:", status)
    print("Ingestion complete.")

