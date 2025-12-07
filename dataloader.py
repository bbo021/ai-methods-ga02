import numpy as np
import pandas as pd

PATH_DATA_2022 = "dataset/data_2022.csv"
PATH_DATA_2023 = "dataset/data_2023.csv"

def create_dataset_pandas(path_csv):
    """
    Wrapper to get pandas dataframe for the csv dataset
    
    :param path_csv: str path to csv dataset
    """
    return pd.read_csv(path_csv)

def create_dataset(path_csv):
    df = pd.read_csv(path_csv)
    date = df['Date']
    demand = df['Demand']
    X = np.array(date.values)
    y = np.array(demand.values)

    return X, y

def create_sliding_window_dataset(path_csv, window):
    """
    Generates a sliding window regression dataset
    
    :param path_csv: Description
    :param window: Description
    :return X, y: X and y as np arrays
    """
    df = pd.read_csv(path_csv)
    demand = df['Demand']
    values_demand = demand.values

    X, y = [], []
    for i in range(len(values_demand) - window):
        X_window = values_demand[i : i + window]
        y_window = values_demand[i + window]
        X.append(X_window)
        y.append(y_window)
    return np.array(X), np.array(y)

def bootstrap_dataset(X, y):
    n_samples = X.shape[0]

    # Selecting n_samples of indices from n_samples with replacement (to feed variability)
    indices = np.random.choice(n_samples, n_samples, replace=True)

    X_bootstrap = X[indices]
    y_bootstrap = y[indices]

    return X_bootstrap, y_bootstrap

    
