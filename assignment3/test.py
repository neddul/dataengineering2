import ray
from ray import tune
from ray.tune import run_experiments
from ray.tune.analysis.experiment_analysis import ExperimentAnalysis
from sklearn.datasets import fetch_covtype
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

#Step 3: Configure the search space

config = {
    "max_depth": tune.grid_search([10, 20, 30, None]),
    "n_estimators": tune.grid_search([100, 200, 300]),
    "ccp_alpha": tune.grid_search([0.0, 0.1, 0.2])
}
# Step 4: Implement the train function
def train_rf(config):

    covtype = fetch_covtype()
    # Select the first 10,000 samples
    X = covtype.data[:1000]
    y = covtype.target[:1000]
    covtype = None
    #X, y = covtype.data, covtype.target
    model = RandomForestClassifier(
        max_depth=config["max_depth"],
        n_estimators=config["n_estimators"],
        ccp_alpha=config["ccp_alpha"],
        random_state=42,
    )
    score = cross_val_score(model, X, y, cv=3).mean()
    tune.report(mean_accuracy=score)

# Step 5: Execute the pipeline

# Initialize Ray
ray.init(address="auto")

# Run experiments
run_experiments({
    "rf_hyperparameter_tuning": {
        "run": train_rf,
        "config": config,
        "num_samples": 1,
        "resources_per_trial": {"cpu": 2},
#        "stop": {"mean_accuracy": 0.99},
    }
})

# Analyze the results and get the best trial
analysis = ExperimentAnalysis("ray_results/rf_hyperparameter_tuning")
best_trial = analysis.get_best_trial(metric="mean_accuracy", mode="max")
best_config = best_trial.config

# Print the best trial config and mean accuracy
print("Best trial config: ", best_config)
print("Best trial mean accuracy: ", best_trial.last_result["mean_accuracy"])
