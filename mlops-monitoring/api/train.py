from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import pickle

iris = load_iris()
X, y = iris.data, iris.target
model = RandomForestClassifier(n_estimators=100)
model.fit(X, y)

# Save model
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)
