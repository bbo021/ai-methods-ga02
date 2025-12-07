import numpy as np
import sklearn
from sklearn.tree import DecisionTreeRegressor

def train_single_decision_tree_regressor(X, y, depth):
    dtr = DecisionTreeRegressor(random_state=42, max_depth=depth)
    dtr.fit(X, y)

    return dtr

def predict_single_decision_tree_regressor_sliding_window(dtr, target, window_size):
    n_samples = len(target)
    predictions = []
    
    for i in range(n_samples - window_size):
        X_window = target[i : i + window_size]
        X_window = X_window.reshape(1, -1)
        y_pred = dtr.predict(X_window)[0]
        predictions.append(y_pred)

    return np.array(predictions)
