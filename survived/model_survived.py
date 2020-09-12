import numpy as np
import pandas as pd

from helper.math_helper import sigmoid, \
    cost_function_logistic_regression, find_best_regularization_logistic_regression
from helper.process_data import get_processed_data_survived, split_data_for_training


def train_survived():
    x = get_processed_data_survived("data_csv/train_with_ages.csv")

    # Prediction data - y - we wish to predict survived
    y = x["Survived"].values.reshape(-1, 1)
    # remove y from dataset x
    del x["Survived"]

    # Add x0 column of ones
    x.insert(0, "x0", np.ones(x.shape[0]).reshape(-1, 1))

    x_train, y_train, x_cv, y_cv, x_test, y_test = split_data_for_training(x.values, y, 0.6, 0.2)

    c, theta_c = find_best_regularization_logistic_regression(x_train, y_train, x_cv, y_cv)

    # Save theta to file
    np.savetxt('data_csv/theta_survived.txt', theta_c)

    # Calculate Train and Test errors to evaluate the model
    J_train = cost_function_logistic_regression(sigmoid(x_train @ theta_c), y_train)  # ~ 0.47
    print("Training error is: %f" % J_train)
    J_test = cost_function_logistic_regression(sigmoid(x_test @ theta_c), y_test)  # ~0.44
    print("Testing error is: %f" % J_test)
    print("Best regularization parameter found is: %f" % c)


def predict_survived():
    # Predict survived of test data
    test_data = get_processed_data_survived("./data_csv/test_with_ages.csv")
    test_data.insert(0, "x0", np.ones(test_data.shape[0]).reshape(-1, 1))

    theta = np.loadtxt('./data_csv/theta_survived.txt')
    y_test = np.where(sigmoid(test_data @ theta) >= 0.5, 1, 0)

    # Create submission file
    submission = pd.DataFrame(test_data.axes[0] + 892, columns=['PassengerId'])
    submission['Survived'] = y_test.reshape(-1, 1)
    submission_sorted = submission.sort_values('PassengerId')
    submission_sorted.to_csv("./data_csv/submission.csv", index=False)


# Get theta to predict survived
train_survived()

# Use theta to predict test data survived
predict_survived()
