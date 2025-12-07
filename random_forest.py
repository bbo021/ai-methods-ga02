import numpy as np
import sklearn
from sklearn.tree import DecisionTreeRegressor
import dataloader as dl

np.random.seed(42)

class RandomForest:

    def __init__(self, n_features, n_trees=50):
        self.ensemble = []
        self.n_trees = n_trees
        self.n_features = n_features

        for i in range(n_trees):
            random_hyperparams = self._create_random_hyperparams(42)
            tree = DecisionTreeRegressor(**random_hyperparams)
            self.ensemble.append(tree)


    def fit(self, X, y):
        for tree in self.ensemble:
            X_bootstrap, y_bootstrap = dl.bootstrap_dataset(X, y)

            tree.fit(X_bootstrap, y_bootstrap)

    def predict(self, X):
        y_pred = []
        for tree in self.ensemble:
            y_pred.append(tree.predict(X))

        return np.array(y_pred).mean(axis=0)

    def predict_sliding_window(self, X, window_size):
        y_pred = []

        for i in range(len(X) - window_size):
            X_window = X[i : i + window_size]
            X_window = X_window.reshape(1, -1)
            y_pred.append(self.predict(X_window)[0])

        return np.array(y_pred)

    def _create_random_hyperparams(self, random_state):
        hyperparams = {
            'random_state': random_state,
            'max_depth': np.random.randint(2, self.n_features),
            'max_features': np.random.randint(1, self.n_features + 1),
            'min_samples_split': np.random.randint(2, self.n_features)
        }

        return hyperparams