import os
import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import dagshub

# Set MLflow Tracking URI to DagsHub
os.environ["MLFLOW_TRACKING_URI"] = (
    "https://dagshub.com/YOUR_USERNAME/YOUR_REPO_NAME/mlflow"
)
os.environ["MLFLOW_TRACKING_USERNAME"] = "YOUR_USERNAME"
os.environ["MLFLOW_TRACKING_PASSWORD"] = os.getenv("DAGSHUB_PAT")
# Load data
data = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    data.data, data.target, test_size=0.2, random_state=42
)

# Start MLflow experiment
mlflow.set_experiment("simple-iris-model")

with mlflow.start_run():
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    preds = clf.predict(X_test)
    acc = accuracy_score(y_test, preds)

    # Log model and metrics
    mlflow.log_param("n_estimators", 100)
    mlflow.log_metric("accuracy", acc)
    mlflow.sklearn.log_model(clf, "model")

    print(f"Accuracy: {acc:.4f}")
