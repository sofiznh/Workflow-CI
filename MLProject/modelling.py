import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
import argparse
import warnings

parser = argparse.ArgumentParser()
parser.add_argument("--n_estimators", type=int, default=150)
parser.add_argument("--max_depth", type=int, default=15)
parser.add_argument("--min_samples_split", type=int, default=8)
parser.add_argument("--min_samples_leaf", type=int, default=2)
parser.add_argument("--bootstrap", type=bool, default=True)
parser.add_argument("--max_features", type=str, default="log2")
parser.add_argument("--random_state", type=int, default=42)
parser.add_argument("--dataset", type=str, default="preprocessed_dataset")
args = parser.parse_args()

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(args.random_state)

    # Load dataset
    X_train = pd.read_csv(f"{args.dataset}/X_train.csv")
    X_test = pd.read_csv(f"{args.dataset}/X_test.csv")
    Y_train = pd.read_csv(f"{args.dataset}/Y_train.csv").values.ravel()
    Y_test = pd.read_csv(f"{args.dataset}/Y_test.csv").values.ravel()

    with mlflow.start_run():
        # Logging parameters
        mlflow.log_param("n_estimators", args.n_estimators)
        mlflow.log_param("max_depth", args.max_depth)
        mlflow.log_param("min_samples_split", args.min_samples_split)
        mlflow.log_param("min_samples_leaf", args.min_samples_leaf)
        mlflow.log_param("bootstrap", args.bootstrap)
        mlflow.log_param("max_features", args.max_features)
        mlflow.log_param("random_state", args.random_state)

        # Model
        model = RandomForestClassifier(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            min_samples_split=args.min_samples_split,
            min_samples_leaf=args.min_samples_leaf,
            bootstrap=args.bootstrap,
            max_features=args.max_features,
            random_state=args.random_state
        )

        model.fit(X_train, Y_train)

        # Log model
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            code_paths=["MLProject"]
        )

        # Predict
        Y_train_pred = model.predict(X_train)
        Y_test_pred = model.predict(X_test)

        # Train metrics
        train_acc = accuracy_score(Y_train, Y_train_pred)
        train_bal_acc = balanced_accuracy_score(Y_train, Y_train_pred)
        train_f1 = f1_score(Y_train, Y_train_pred, average="macro")

        mlflow.log_metric("train_accuracy", train_acc)
        mlflow.log_metric("train_balanced_accuracy", train_bal_acc)
        mlflow.log_metric("train_f1_macro", train_f1)

        # Test metrics
        test_acc = accuracy_score(Y_test, Y_test_pred)
        test_bal_acc = balanced_accuracy_score(Y_test, Y_test_pred)
        test_f1 = f1_score(Y_test, Y_test_pred, average="macro")

        mlflow.log_metric("test_accuracy", test_acc)
        mlflow.log_metric("test_balanced_accuracy", test_bal_acc)
        mlflow.log_metric("test_f1_macro", test_f1)
