import numpy as np
import sklearn
import sklearn.neural_network as nn
import dataloader

np.random.seed(42)

class MLPRegressorEnsemble():

    N_ITERATIONS = 1000

    def __init__(self, size, hidden_layers):
        
        self.ensemble = []
        self.random_seed = 42
        for i in range(size):
            mlp_regressor = nn.MLPRegressor(
                hidden_layer_sizes=hidden_layers, 
                random_state=self.random_seed + i, 
                activation='relu',
                solver='lbfgs',
                max_iter=MLPRegressorEnsemble.N_ITERATIONS)
            self.ensemble.append(mlp_regressor)

    
    def train(self, X, y):
        for mlp in self.ensemble:
            X_bootstrapped, y_bootstrapped = dataloader.bootstrap_dataset(X, y)

            mlp.fit(X_bootstrapped, y_bootstrapped)


    def predict(self, X):
        y_pred = []

        for mlp in self.ensemble:
            y_pred.append(mlp.predict(X))

        return np.array(y_pred).mean(axis=0)
    
    def predict_sliding_window(self, X, window_size):
        y_pred = []

        for i in range(len(X) - window_size):
            X_window = X[i : i + window_size]
            X_window = X_window.reshape(1, -1)
            y_pred.append(self.predict(X_window)[0])

        return np.array(y_pred)