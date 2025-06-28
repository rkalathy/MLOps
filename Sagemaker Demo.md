aws codecommit create-repository --repository-name sagemaker-demo --repository-description "Demo SageMaker & Feature Store" --region eu-west-2

{
    "repositoryMetadata": {
        "accountId": "750952118292",
        "repositoryId": "8deabb35-9bbc-4b11-a1f5-2f68daf90ddb",
        "repositoryName": "sagemaker-demo",
        "repositoryDescription": "Demo SageMaker & Feature Store",
        "lastModifiedDate": "2025-06-22T15:29:16.961000+01:00",
        "creationDate": "2025-06-22T15:29:16.961000+01:00",
        "cloneUrlHttp": "https://git-codecommit.eu-west-2.amazonaws.com/v1/repos/sagemaker-demo",
        "cloneUrlSsh": "ssh://git-codecommit.eu-west-2.amazonaws.com/v1/repos/sagemaker-demo",
        "Arn": "arn:aws:codecommit:eu-west-2:750952118292:sagemaker-demo",
        "kmsKeyId": "arn:aws:kms:eu-west-2:750952118292:key/d4b26d02-a348-4928-a844-a0798346b933"
    }
}


git clone https://git-codecommit.eu-west-2.amazonaws.com/v1/repos/sagemaker-demo
cd sagemaker-demo
