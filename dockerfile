FROM public.ecr.aws/lambda/python:3.9
RUN pip install mlflow==2.22.0 && pip install cloudpickle==2.2.1
