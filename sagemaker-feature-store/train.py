import boto3
import pandas as pd
import sagemaker
import joblib
import time
from sagemaker.sklearn.estimator import SKLearn
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


region = "eu-west-2"
bucket_name = "sagemaker-features-demo" 
database_name = "sagemaker_featurestore"
feature_group_name = "customers_feature_group_1750610356"
s3_output = f"s3://{bucket_name}/athena-results/"
role = "arn:aws:iam::750952118292:role/service-role/AmazonSageMaker-ExecutionRole-20250615T204995"

boto_session = boto3.Session(region_name=region)
athena = boto_session.client("athena")
sagemaker_session = sagemaker.Session(boto_session=boto_session)

# glue = boto3.client("glue", region_name="eu-west-2")
# response = glue.get_tables(DatabaseName="sagemaker_featurestore")

# print("Available tables:")
# for table in response["TableList"]:
#     print(table["Name"])

query = f"""
SELECT * FROM "{database_name}"."{feature_group_name}"
"""

# SELECT *FROM "sagemaker_featurestore"."customers_feature_group_1750610356" LIMIT 1000;

print("Athena query...")
response = athena.start_query_execution(
    QueryString=query,
    QueryExecutionContext={"Database": database_name},
    ResultConfiguration={"OutputLocation": s3_output}
)

query_exec_id = response["QueryExecutionId"]

while True:
    result = athena.get_query_execution(QueryExecutionId=query_exec_id)
    status = result["QueryExecution"]["Status"]["State"]

    if status == "SUCCEEDED":
        break
    elif status == "FAILED":
        reason = result["QueryExecution"]["Status"]["StateChangeReason"]
        print(f"Athena query failed: {reason}")
        raise Exception("Athena query failed.")
    time.sleep(2)

result_path = f"{s3_output}{query_exec_id}.csv"
print(f"query succeeded. Result: {result_path}")

df = pd.read_csv(result_path)

print(df.head())

df = df.dropna()

X = df[["age"]] 
y = df["customer_id"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)


joblib.dump(model, "model.joblib")
print("📦 Model saved to model.joblib")