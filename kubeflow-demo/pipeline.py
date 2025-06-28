import kfp
from kfp import dsl

def train_op():
    return dsl.ContainerOp(
        name='Train Linear Model',
        image='python:3.9',
        command=['sh', '-c'],
        arguments=[
            'pip install scikit-learn pandas joblib && '
            'python -c "'
            'import numpy as np; '
            'import joblib; '
            'from sklearn.linear_model import LinearRegression; '
            'X = np.array([[1], [2], [3], [4]]); '
            'y = np.array([2, 4, 6, 8]); '
            'model = LinearRegression().fit(X, y); '
            'joblib.dump(model, \'model.pkl\'); '
            'print(\'Model saved to model.pkl\')"'
        ],
        file_outputs={
            'model': '/model.pkl'  # optional: define if you want to track outputs
        }
    )

@dsl.pipeline(
    name='Linear Model with Pickle Output',
    description='Trains and saves a model to model.pkl'
)
def simple_pipeline():
    train_op()

if __name__ == '__main__':
    kfp.compiler.Compiler().compile(simple_pipeline, 'model_pipeline.yaml')
