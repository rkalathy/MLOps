# Use a slim Python base
FROM python:3.8-slim

# Install MLflow and AWS SDK
RUN pip install --no-cache-dir \
    mlflow==3.1.0 \
    boto3 \
    gunicorn

# Create model directory
WORKDIR /opt/ml/model

# Copy your logged model into the image
COPY model/ /opt/ml/model

# Expose the port SageMaker will call
EXPOSE 8080

# Launch the MLflow PyFunc server via Gunicorn for production readiness
ENTRYPOINT ["gunicorn", "--bind", "0.0.0.0:8080", \
            "mlflow.pyfunc.scoring_server:app"]
