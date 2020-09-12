import numpy as np

from helper.process_data import get_processed_data_age, split_data_for_training, process_all_data
from helper.math_helper import cost_function_linear_regression, create_polynomial_dataset, find_best_polynomial_degree


def train_ages():
    x, x_missing_age = get_processed_data_age("../data_csv/train_normalized.csv")
    # Save copy of original data
    dataset_original_ages = x.copy()

    # Prediction data - y - we wish to predict the missing ages
    y = x["Age"].values.reshape(-1, 1)
    # remove y from dataset x
    del x["Age"]

    x_train, y_train, x_cv, y_cv, x_test, y_test = split_data_for_training(x.values, y, 0.6, 0.2)

    d, theta_d = find_best_polynomial_degree(10, x_train, y_train, x_cv, y_cv)
    print("Best polynomial degree found is: %d" % d)
    # Save theta to file
    np.savetxt('../data_csv/theta_ages.txt', theta_d)

    # Calculate Train and Test errors to evaluate the model
    x_train_d = create_polynomial_dataset(x_train, d)[d]
    J_train = cost_function_linear_regression(x_train_d @ theta_d - y_train)  # ~73
    print("Training error is: %f" % J_train)
    x_test_d = create_polynomial_dataset(x_test, d)[d]
    J_test = cost_function_linear_regression(x_test_d @ theta_d - y_test)  # ~82
    print("Testing error is: %f" % J_test)

    # Use optimized theta to get missing ages
    x_missing_age_d = create_polynomial_dataset(x_missing_age.values, d)[d]
    y_missing_age = x_missing_age_d @ theta_d

    # Join predicted dataset with original and save to csv
    x_missing_age['Age'] = y_missing_age
    dataset_full = dataset_original_ages.append(x_missing_age)
    dataset_full.to_csv("../data_csv/train_with_ages.csv", index=False)


def predict_ages():
    # Predict ages of test data
    test_data, test_data_missing_age = get_processed_data_age("../data_csv/test_normalized.csv")
    theta = np.loadtxt('../data_csv/theta_ages.txt')
    d = int((theta.shape[0] - 1) / test_data.shape[1])

    # Remove theta elements corresponding with survived
    theta_without_survived = np.delete(theta, np.arange(1, theta.size, test_data.shape[1]))
    test_data_d = create_polynomial_dataset(test_data_missing_age.values, d)
    y_missing_age = test_data_d[d] @ theta_without_survived

    # Join predicted dataset with test dataset
    test_data_missing_age['Age'] = y_missing_age.reshape(-1, 1)
    dataset_full = test_data.append(test_data_missing_age)
    dataset_full_sorted = dataset_full.sort_index()
    dataset_full_sorted.to_csv("../data_csv/test_with_ages.csv", index=False)


# Data Completing, Correcting, Creating
process_all_data()

# Get theta to predict ages
train_ages()

# Use theta to predict test data ages
predict_ages()
