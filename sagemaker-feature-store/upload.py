import boto3
import sagemaker
from sagemaker.session import Session
import pandas as pd

from sagemaker.feature_store.feature_definition import FeatureDefinition, FeatureTypeEnum
from sagemaker.feature_store.feature_group import FeatureGroup


region = "eu-west-2"
boto_session = boto3.Session(region_name=region)
sagemaker_client = boto_session.client("sagemaker")
sagemaker_session = sagemaker.Session(boto_session=boto_session)
role = "arn:aws:iam::750952118292:role/service-role/AmazonSageMaker-ExecutionRole-20250615T204995"
bucket_name = "sagemaker-features-demo" 
feature_group_name = "customers-feature-group"
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

