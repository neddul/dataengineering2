import ray
from ray import tune
from sklearn.datasets import fetch_covtype
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score

def train_random_forest(param_grid, data):
    X_train, X_test, y_train, y_test = data

    # Initialize and train the Random Forest classifier
    clf = RandomForestClassifier(
        max_depth=param_grid["max_depth"],
        n_estimators=param_grid["n_estimators"],
        ccp_alpha=param_grid["ccp_alpha"],
        random_state=42
    )
    clf.fit(X_train, y_train)
    
    # Predict and calculate accuracy
    accuracy = cross_val_score(clf, X_test, y_test, cv=5).mean()
    
    # Report the accuracy to Ray Tune
    tune.report(accuracy=accuracy)

# Define the search space for hyperparameters
search_space = {
    "max_depth": tune.choice([10, 20, 30, 40, 50]),
    "n_estimators": tune.choice([50, 100, 150, 200]),
    "ccp_alpha": tune.loguniform(1e-6, 1e-2)
}

if __name__ == "__main__":
    print("Loading data")
    # Load dataset once
    data = fetch_covtype()
    X_train, X_test, y_train, y_test = train_test_split(data.data[:10_000], data.target[:10_000], test_size=0.2, random_state=42)
    dataset = (X_train, X_test, y_train, y_test)
    
    print("Looking for Ray cluster")
    # Initialize Ray
    ray.init(address='auto')  # Automatically detect the Ray cluster
    
    print("Start tuning")
    # Run hyperparameter tuning
    analysis = tune.run(
        tune.with_parameters(train_random_forest, data=dataset),
        config=search_space,
        num_samples=20,  # Number of hyperparameter configurations to try
        metric="accuracy",
        mode="max",
        # resources_per_trial={"cpu": 1, "gpu": 0},  # Adjust based on your VM configuration
    )
    
    # Print the best hyperparameter configuration
    print("Best hyperparameter configuration found:")
    best_config = analysis.best_config
    print(best_config)
    
    # Evaluate the best configuration on the test set
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
