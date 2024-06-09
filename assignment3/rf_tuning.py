import ray
import time
from ray import tune
from sklearn.datasets import fetch_covtype
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def train_random_forest(config):
    # Load dataset
    data = fetch_covtype()
    dsl = 1000 # Dataset size limit
    X_train, X_test, y_train, y_test = train_test_split(data.data[:dsl], data.target[:dsl], test_size=0.2, random_state=42)

    # Initialize and train the Random Forest classifier
    clf = RandomForestClassifier(
        max_depth=config["max_depth"],
        n_estimators=config["n_estimators"],
        ccp_alpha=config["ccp_alpha"],
        random_state=42
    )
    clf.fit(X_train, y_train)
    
    # Predict and calculate accuracy
    predictions = clf.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    
    # Report the accuracy to Ray Tune
    tune.report(accuracy=accuracy)

# Define the search space for hyperparameters
search_space = {
    "max_depth": tune.choice([10, 20, 30]),
    "n_estimators": tune.choice([50, 100, 150, 200]),
    "ccp_alpha": tune.loguniform(1e-6, 1e-2)
}

if __name__ == "__main__":
    # Initialize Ray
    ray.init(address='auto')  # Automatically detect the Ray cluster
    start_time = time.time()
    # Run hyperparameter tuning
    analysis = tune.run(
        train_random_forest,
        config=search_space,
        resources_per_trial={"cpu": 1, "gpu": 0},  # Adjust based on your VM configuration
        num_samples=10,  # Number of hyperparameter configurations to try
        metric="accuracy",
        mode="max"
    )
    end_time = time.time()
    # Print the best hyperparameter configuration
    print("Best hyperparameter configuration found:")
    best_config = analysis.best_config
    print(best_config)

    print(f"Time spent tuning: {round(end_time-start_time, 3)}")
    
    # Evaluate the best configuration on the test set
    data = fetch_covtype()
    X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2, random_state=42)

    best_clf = RandomForestClassifier(
        max_depth=best_config["max_depth"],
        n_estimators=best_config["n_estimators"],
        ccp_alpha=best_config["ccp_alpha"],
        random_state=42
    )
    best_clf.fit(X_train, y_train)
    best_predictions = best_clf.predict(X_test)
    best_accuracy = accuracy_score(y_test, best_predictions)

    print(f"Accuracy with the best configuration: {best_accuracy}")
