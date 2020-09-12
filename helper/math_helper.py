from sklearn.linear_model import LogisticRegression, LinearRegression
import numpy as np


def feature_scaling_mean_normalization(df, column_name):
    return (df[column_name] - df[column_name].mean()) / df[column_name].std()


def cost_function_linear_regression(m_error):
    """
    Returns Cost Function for Linear Regression
    """
    return (m_error.transpose() @ m_error) / (2 * m_error.size)


def cost_function_logistic_regression(h, y):
    """
    Returns Cost Function for Logistic Regression
    """
    return 1 / h.shape[0] * (-y.transpose() @ np.log(h) - (1 - y).transpose() @ np.log(1 - h))


def get_theta_vector(fit):
    """
    Expected model shouldn't be initialized with a column of ones
    :param fit: model from sklearn fit
    :return: theta vector with intercept in position 0 and coefficients
    """
    theta = np.insert(fit.coef_, 0, fit.intercept_[0])
    fit.coef_[0, 0] = fit.intercept_[0]  # replace theta0 with intercept_ value
    return theta.reshape(-1, 1)


def sigmoid(x):
    """
    Sigmoid function of x
    """
    return 1 / (1 + np.exp(-x))


def logistic_regression(x, y, c=1.0):
    """
    fit logistic regression model
    """
    model = LogisticRegression(C=c)
    return get_theta_vector(model.fit(x, y))


def linear_regression(x, y):
    """
    fit linear regression model
    """
    model = LinearRegression()
    return get_theta_vector(model.fit(x, y))


def get_logistic_regression_error(h, y):
    """
    Return error for logistic regression
    """
    err_fun = np.where((h >= 0.5) & (y == 0) | (h < 0.5) & (y == 1), 1, 0)
    return 1 / h.shape[0] * np.sum(err_fun)


def create_polynomial_dataset(dataset, degree):
    """
    Get dataset and return a list of dataset to the d polynomial degree
    List index matches degree, 0 is set to x0 vector (ones column).

    x - numpy array
    d - integer
    """
    dataset_list = [np.ones(dataset.shape[0])]
    for i in range(1, degree + 1):
        dataset_list.append(np.column_stack((dataset_list[i - 1], dataset ** i)))
    return dataset_list


def find_best_polynomial_degree(d, x_train, y_train, x_cv, y_cv):
    """
    Given train and cross validation data find polynomial degree that fits data best from 1-d
    Return degree and the corresponding theta
    """
    x_train_list = create_polynomial_dataset(x_train, d)

    minimized_theta = [0]
    for training_data in x_train_list[1:]:
        minimized_theta.append(linear_regression(training_data[:, 1:], y_train))

    # Calculate cv error for each polynomial degree
    x_cv_list = create_polynomial_dataset(x_cv, d)
    J_cv = [0]  # initializing 0 so index will start from 1
    for i in range(1, len(x_cv_list)):
        J_cv.append(cost_function_linear_regression(x_cv_list[i] @ minimized_theta[i] - y_cv)[0, 0])

    # Find polynomial degree with least error
    d_minimized = J_cv.index(min(J_cv[1:]))

    return d_minimized, minimized_theta[d_minimized]


def find_best_regularization_logistic_regression(x_train, y_train, x_cv, y_cv):
    """
    Given train and cross validation data find regularization term that fits data best
    Return regularization term and the corresponding theta
    """
    reg_list = [0.01, 0.05, 0.09, 0.95, 1.0]

    minimized_theta = [0]  # initializing 0 so index will start from 1
    for reg_term in reg_list:
        minimized_theta.append(logistic_regression(x_train[:, 1:], y_train.ravel(), c=reg_term))

    # Calculate cv error for each regularization term
    J_cv = [0]  # initializing 0 so index will start from 1
    for theta in minimized_theta[1:]:
        J_cv.append(get_logistic_regression_error(sigmoid(x_cv @ theta), y_cv))

    # Find polynomial degree with least error
    minimized_index = J_cv.index(min(J_cv[1:]))
    reg_minimized = reg_list[minimized_index - 1]

    return reg_minimized, minimized_theta[minimized_index]
