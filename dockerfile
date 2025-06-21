FROM python:3.9-slim

# Install required packages
RUN pip install mlflow==2.22.0 gunicorn flask scikit-learn pandas numpy

# Set working directory
WORKDIR /opt/ml/code

# Copy model and serving code
COPY . .

# Expose port for SageMaker
EXPOSE 8080

# Set environment variables
ENV PYTHONUNBUFFERED=TRUE
ENV PYTHONDONTWRITEBYTECODE=TRUE
ENV PATH="/opt/ml/code:${PATH}"

# Set the entrypoint for SageMaker serving
ENTRYPOINT ["python", "-m", "mlflow.models.container", "--enable-mlserver"]