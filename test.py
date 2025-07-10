import os
import mlflow
import mlflow.sklearn
from dagshub import dagshub_logger
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load DagsHub creds from environment
os.environ["MLFLOW_TRACKING_URI"] = os.getenv("MLFLOW_TRACKING_URI")
os.environ["MLFLOW_TRACKING_USERNAME"] = os.getenv("MLFLOW_TRACKING_USERNAME")
os.environ["MLFLOW_TRACKING_PASSWORD"] = os.getenv("MLFLOW_TRACKING_PASSWORD")

# Load data
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2)

# Set experiment (will be auto-created if doesn't exist)
mlflow.set_experiment("dagshub-iris-example")

with mlflow.start_run():
    clf = RandomForestClassifier(n_estimators=100)
    clf.fit(X_train, y_train)
    acc = accuracy_score(y_test, clf.predict(X_test))

    mlflow.sklearn.log_model(clf, "model")
    mlflow.log_metric("accuracy", acc)

    with dagshub_logger() as logger:
        logger.log_metrics({"accuracy": acc})
