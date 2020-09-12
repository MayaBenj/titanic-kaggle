# titanic-kaggle

Suggested solution for https://www.kaggle.com/c/titanic

#### model_age.py
* Load data, correct and clean it. Save new data to test_normalized.csv and train_normalized.csv
* Use linear regression to predict missing ages. 
    1. Try different polynomial degrees and choose one with lowest cost function using cross-validation.
    2. Output train and test error.
* Save computed theta to theta_ages.txt, and full training data to train_with_ages.csv
* Use theta to predict ages of test data, save to test_with_ages.csv

#### model_survived.py
* Use logistic regression to to predict survived.
    1. Try different regularization term and choose one with lowest cost function using cross-validation.
    2. Output train and test error.
* Save computed theta to theta_survived.txt.
* Use theta to predict survived of test data, generate submission.csv
