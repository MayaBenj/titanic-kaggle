import re

import pandas as pd

from helper.math_helper import feature_scaling_mean_normalization


def get_title(name):
    """
    Extract title from name
    :param name: String
    :return: String
    """
    title_search = re.search(' ([A-Za-z]+)\.', name)
    return title_search.group(1) if title_search else ""


def split_data_for_training(x, y, train_part, cv_part):
    """
    NO VALIDATION
    get dataset and return
        training: train_part of x
        cv: cv_part of x
        test: x - (training + cv)

    :param x: numpy array
    :param y: numpy array
    :param train_part: float 0-1
    :param cv_part: float 0-1
    :return:
    """
    m = x.shape[0]  # number of samples
    train_num = round(m * train_part)
    cv_num = train_num + round(m * cv_part)  # index for train_part + cv_part of data samples

    x_train = x[:train_num]
    y_train = y[:train_num]
    x_cv = x[train_num + 1: cv_num]
    y_cv = y[train_num + 1: cv_num]
    x_test = x[cv_num + 1:]
    y_test = y[cv_num + 1:]

    return x_train, y_train, x_cv, y_cv, x_test, y_test


def process_all_data():
    train_data = pd.read_csv("../data_csv/train.csv")
    test_data = pd.read_csv("../data_csv/test.csv")
    combined_data = train_data.append(test_data)

    # Change 'Sex' feature: Male - 0, Female - 1
    combined_data.loc[combined_data.Sex == 'male', 'Sex'] = 0
    combined_data.loc[combined_data.Sex == 'female', 'Sex'] = 1

    # Change 'Embarked' feature C = 0; S = 1; Q = 2;
    combined_data.loc[combined_data.Embarked == 'C', 'Embarked'] = 0
    combined_data.loc[combined_data.Embarked == 'S', 'Embarked'] = 1
    combined_data.loc[combined_data.Embarked == 'Q', 'Embarked'] = 2

    # Replace missing 'Embarked' with most common
    combined_data['Embarked'].fillna(combined_data['Embarked'].mode()[0], inplace=True)
    # Replace missing 'Fare' data with median
    combined_data['Fare'].fillna(combined_data['Fare'].median(), inplace=True)

    # Feature scaling and mean normalization
    combined_data['Fare'] = feature_scaling_mean_normalization(combined_data, 'Fare')

    # Create new feature Family - Parch  + SibSp + 1(self)
    combined_data['Family'] = combined_data["Parch"] + combined_data["SibSp"] + 1
    # Remove Parch and SibSp
    combined_data.drop(['Parch', 'SibSp'], 1, inplace=True)

    # Create new feature Title - Extracted from name
    combined_data['Title'] = combined_data['Name'].apply(get_title)
    # Combine together rare occurrences
    combined_data['Title'] = combined_data['Title']\
        .replace(['Lady', 'Countess', 'Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    combined_data['Title'] = combined_data['Title'].replace('Mlle', 'Miss')
    combined_data['Title'] = combined_data['Title'].replace('Ms', 'Miss')
    combined_data['Title'] = combined_data['Title'].replace('Mme', 'Mrs')
    # Change 'Title' feature Rare = 0; Master = 1; Miss = 2; Mrs = 3; Mr = 4;
    combined_data.loc[combined_data.Title == 'Rare', 'Title'] = 0
    combined_data.loc[combined_data.Title == 'Master', 'Title'] = 1
    combined_data.loc[combined_data.Title == 'Miss', 'Title'] = 2
    combined_data.loc[combined_data.Title == 'Mrs', 'Title'] = 3
    combined_data.loc[combined_data.Title == 'Mr', 'Title'] = 4
    # Replace missing 'Title' most common
    combined_data['Title'].fillna(combined_data['Title'].mode()[0], inplace=True)

    # Remove 'PassengerId', 'Name', 'Ticket', 'Cabin
    combined_data.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], 1, inplace=True)

    # Split back to train and test
    combined_data[:train_data.shape[0]].to_csv("../data_csv/train_normalized.csv", index=False)
    combined_data.drop(['Survived'], 1, inplace=True)
    combined_data[train_data.shape[0]:].to_csv("../data_csv/test_normalized.csv", index=False)


def get_processed_data_age(file_path):
    data = pd.read_csv(file_path)

    # For predicting ages - Remove passengers with age unknown and shuffle
    data_with_ages = data[pd.to_numeric(data['Age'], errors='ignore').notnull()]
    data_with_ages = data_with_ages.sample(frac=1)

    # For evaluating ages - Remove passengers with age known and remove 'Age' column
    data_no_ages = data[pd.to_numeric(data['Age'], errors='ignore').isnull()]
    data_no_ages = data_no_ages.drop(['Age'], 1)

    return data_with_ages, data_no_ages


def get_processed_data_survived(file_path):
    data = pd.read_csv(file_path)

    # Shuffle data
    data = data.sample(frac=1)

    return data
