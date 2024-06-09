import ray
from ray import tune
from sklearn.datasets import fetch_covtype
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def train_random_forest(config):
    # Load dataset
    data = fetch_covtype()
    X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2, random_state=42)

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
    "max_depth": tune.choice([10, 20, 30, 40, 50]),
    "n_estimators": tune.choice([50, 100, 150, 200]),
    "ccp_alpha": tune.loguniform(1e-6, 1e-2)
}

if __name__ == "__main__":
    # Initialize Ray
    ray.init(address='auto')  # Automatically detect the Ray cluster
    
    # Run hyperparameter tuning
    analysis = tune.run(
        train_random_forest,
        config=search_space,
        #resources_per_trial={"cpu": 1, "gpu": 0},  # Adjust based on your VM configuration
        num_samples=10,  # Number of hyperparameter configurations to try
        metric="accuracy",
        mode="max"
    )
    
    # Print the best hyperparameter configuration
    print("Best hyperparameter configuration found:")
    print(analysis.best_config)
