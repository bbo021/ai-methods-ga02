import dataloader as dl
import decision_tree_regression
import mlp_regressors_ensemble
import random_forest
import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics
import pandas as pd

# Globals
np.random.seed(42)
window_size = 7 # A week for sliding windows (7 features for each label)

def task2():
    # Create data for single regression tree training
    df = dl.create_dataset_pandas(dl.PATH_DATA_2022)
    X, y = dl.create_sliding_window_dataset(dl.PATH_DATA_2022, window_size)

    # Train the DTR (on sliding window dataset from 2022)
    dtr = decision_tree_regression.train_single_decision_tree_regressor(X, y, window_size)

    # Use sliding window approach from start of 2022 dataset to predict each next value, save predictions in a list.
    y_pred = decision_tree_regression.predict_single_decision_tree_regressor_sliding_window(dtr, df['Demand'].values, window_size)

    # - Visualize predictions with matplotlib and comment results -
    # Print first 20 comparisons
    print(np.array(y[:20] - y_pred[:20]))

    # Print mean error
    error_mean = sklearn.metrics.mean_absolute_error(y, y_pred)
    print(f'Using MAE to indicate loss for DTR model: {error_mean}')
    label_dates = df['Date'].values[window_size:]

    # Print shapes for verification
    print(f'Shapes - label_dates {label_dates.shape}, y {y.shape}, y_pred {y_pred.shape}')

    plt.figure(figsize=(10, 4))
    plt.plot(label_dates, y, label="Target demand")
    plt.plot(label_dates, y_pred, label="Predicted demand")

    plt.title(f"2 - Decision Tree Regressor applied to data_2022.csv, depth:{window_size}")
    plt.xlabel("Date")
    plt.ylabel("Demand")
    plt.legend()
    plt.tight_layout()
    plt.show()

    return y, y_pred, dtr, error_mean

def task3():
    # Create data for training and testing
    random_forest_size = 75 # Suggested lower-bound in assignment specification
    df = dl.create_dataset_pandas(dl.PATH_DATA_2022)
    X, y = dl.create_sliding_window_dataset(dl.PATH_DATA_2022, window_size)

    # Train and test random forest
    rf = random_forest.RandomForest(window_size, random_forest_size)
    rf.fit(X, y)

    # Predict sliding window dataset for 2022
    y_pred = rf.predict_sliding_window(df['Demand'].values, window_size)

    error_mean = sklearn.metrics.mean_absolute_error(y, y_pred)
    print(f'Using MAE to indicate loss for Random Forest (ensemble of DTRs): {error_mean}')

    label_dates = df['Date'].values[window_size:]

    # - Visualize predictions with matplotlib and comment results -
    plt.figure(figsize=(10, 4))
    plt.plot(label_dates, y, label="Target demand")
    plt.plot(label_dates, y_pred, label="Predicted demand")

    plt.title(f"3 - Random forest applied to data_2022.csv, number of trees:{rf.n_trees}")
    plt.xlabel("Date")
    plt.ylabel("Demand")
    plt.legend()
    plt.tight_layout()
    plt.show()

    return y, y_pred, rf, error_mean


def task4():
    # Create data for training and testing
    ensemble_size = 10 # We have a small dataset of 365, even with bagging etc, an ensemble size above 20-50 is rarely providing much accuracy gain for small models
    hidden_layers_tuple = (35,) # Hidden layer with 35 neurons, small dataset and we don't want to underfit or overfit so we choose a safe inbetween
    df = dl.create_dataset_pandas(dl.PATH_DATA_2022)
    X, y = dl.create_sliding_window_dataset(dl.PATH_DATA_2022, window_size)

    # Initialize and train ensemble MLPs
    ensemble = mlp_regressors_ensemble.MLPRegressorEnsemble(ensemble_size, hidden_layers_tuple)
    ensemble.train(X, y)

    # Predict sliding window dataset for 2022
    y_pred = ensemble.predict_sliding_window(df['Demand'].values, window_size)

    error_mean = sklearn.metrics.mean_absolute_error(y, y_pred)
    print(f'Using MAE to indicate loss for ensemble of MLPs: {error_mean}')

    label_dates = df['Date'].values[window_size:]

    # Print shapes for verification
    print(f'Shapes - label_dates {label_dates.shape}, y {y.shape}, y_pred {y_pred.shape}')

    # - Visualize predictions with matplotlib and comment results -
    plt.figure(figsize=(10, 4))
    plt.plot(label_dates, y, label="Target demand")
    plt.plot(label_dates, y_pred, label="Predicted demand")

    plt.title(f"4 - MLPRegressor ensemble applied to data_2022.csv, iterations:{mlp_regressors_ensemble.MLPRegressorEnsemble.N_ITERATIONS}")
    plt.xlabel("Date")
    plt.ylabel("Demand")
    plt.legend()
    plt.tight_layout()
    plt.show()

    return y, y_pred, ensemble, error_mean


if __name__ == "__main__":
    task2_y, task2_y_pred, task2_dtr, task2_error_mean = task2()
    task3_y, task3_y_pred, task3_random_forest, task3_error_mean = task3()
    task4_y, task4_y_pred, task4_mlp_ensemble, task4_error_mean = task4()

    # Perform task 5 - Prediction on dataset for 2023 (dates from 26.08.2023 are missing)
    df = dl.create_dataset_pandas(dl.PATH_DATA_2023)
    label_demand = df['Demand'].values.copy()
    label_dates = df['Date'].values.copy()
    X_test, y_test = dl.create_sliding_window_dataset(dl.PATH_DATA_2023, window_size)

    # Eval known 2023 with model trained on 2022
    y_pred_2023 = task4_mlp_ensemble.predict_sliding_window(label_demand, window_size)
    mean_error = sklearn.metrics.mean_absolute_error(y_test, y_pred_2023)
    print(f'MAE for MLP Regressor Ensemble on the 2023 dataset: {mean_error}')

    # The problem is to now predict missing values, my idea is to use the last window available in data_2023.csv
    # to predict the next value that way, then we do that until end of year.
    index_last = len(label_demand)
    window_last = label_demand[index_last - window_size : index_last]
    y_forecast = []
    n_missing_dates = 365 - index_last

    for index in range(n_missing_dates):
        x = np.array(window_last).reshape(1, -1)
        y_pred_next = task4_mlp_ensemble.predict(x)[0]

        window_last = window_last[1 : 1 + window_size]
        window_last = np.append(window_last, y_pred_next)

        y_forecast.append(y_pred_next)

    y_forecast = np.array(y_forecast)

    # Visualize 2023 existing and forecasted demand
    dates = pd.to_datetime(label_dates, dayfirst=True)
    dates_missing = pd.date_range(dates[-1] + pd.Timedelta(days=1), periods=n_missing_dates, freq=pd.Timedelta(days=1))
    plt.figure(figsize=(10, 4))
    plt.plot(dates, label_demand, label="Existing demand 2023")
    plt.plot(dates_missing, y_forecast, label="Forecasted demand 2023 using MLP ensemble")
    plt.xlabel("Date")
    plt.ylabel("Demand")
    plt.title("Task 5 - Missing demand prediction using model 4")
    plt.legend()
    plt.tight_layout()
    plt.show()