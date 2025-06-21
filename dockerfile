FROM python:3.9-slim

RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    mlflow==2.22.0 \
    flask==2.3.3 \
    gunicorn==21.2.0 \
    pandas==2.2.3 \
    numpy==2.0.2 \
    scikit-learn==1.6.1 \
    cloudpickle==3.0.0

WORKDIR /opt/ml

COPY mlflow-demo/serve.py /opt/ml/serve.py
COPY requirements.txt /opt/ml/requirements.txt

ENV PYTHONUNBUFFERED=TRUE
ENV PYTHONDONTWRITEBYTECODE=TRUE
ENV PATH="/opt/ml:${PATH}"

EXPOSE 8080

ENTRYPOINT ["python", "/opt/ml/serve.py"]