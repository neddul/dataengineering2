from sklearn.datasets import fetch_covtype
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score

import ray
from ray import train, tune


ray.init(address="auto")

config={
            "max_depth": tune.grid_search([10, 20, 30, None]),
            "n_estimators": tune.grid_search([100, 200, 300, 400]),
            "ccp_alpha": tune.grid_search([0.0, 0.1, 0.2])
        }

def train_rf(config):
    covtype = fetch_covtype()
    dsl = 10000 # Max dataset size
    X = covtype.data[:dsl]
    y = covtype.target[:dsl]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    covtype = None

    model = RandomForestClassifier(
        max_depth=config["max_depth"],
        n_estimators=config["n_estimators"],
        ccp_alpha=config["ccp_alpha"]

    )
    # Train the classifier
    model.fit(X_train, y_train)

    score = cross_val_score(model, X_test, y_test, cv=3).mean()
    # Report the accuracy to Ray Tune
    train.report({"mean_accuracy": score})


# Configure the tuner to use the available resources in the cluster
tuner = tune.Tuner(
    tune.with_resources(train_rf, {"cpu": 2, "memory": 2 * 1024 * 1024 * 1024}),  # Request 2 CPUs and 2GB of memory
    tune_config=tune.TuneConfig(
        num_samples=2,
        max_concurrent_trials=3  # Adjust this based on the number of available nodes and resources
    ),
    param_space=config,
)

results = tuner.fit()

# Print the best hyperparameters found
best_result = results.get_best_result(metric="mean_accuracy", mode="max")
print("Best config: ", best_result.config)
print("Best mean accuracy: ", best_result.metrics["mean_accuracy"])

