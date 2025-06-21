import mlflow
import os
import dagshub

dagshub.init(repo_owner='frontenddevtrainer', repo_name='MLOps', mlflow=True)

# List recent runs
runs = mlflow.search_runs()
print(runs[['run_id', 'status', 'start_time']])


# mlflow models serve -m runs:/your-run-id/model --port 5000
# $env:MLFLOW_TRACKING_URI="https://dagshub.com/your-username/your-repo.mlflow"
# https://dagshub.com/frontenddevtrainer/MLOps.mlflow
# mlflow models serve -m runs:/59d4b68a2fcd4477b7b61e532d5e6aa7/model --port 5000