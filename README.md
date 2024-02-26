# Machine Learning Models and Pipelines

## Logistic Regression Model
### Task 1: Data Preprocessing and Model Training
1. Load the dataset from 'kc_house_data.csv'.
2. Drop irrelevant columns ('date', 'zipcode') from the dataset.
3. Normalize the data and split it into features (X) and target variable (y).
4. Binarize the target variable to convert the regression problem into a classification problem.
5. Split the dataset into training and testing sets.
6. Train a Logistic Regression model using the Pipeline approach.
7. Evaluate the model's performance using Mean Squared Error, Accuracy, and Log Loss.
8. Save the trained model using joblib.

## Regularized Regression Models
### Task 2: Data Preprocessing and Model Training
1. Load the dataset from 'kc_house_data.csv'.
2. Drop irrelevant columns ('date', 'zipcode') from the dataset.
3. Normalize the data and split it into features (X) and target variable (y).
4. Split the dataset into training, validation, and testing sets.
5. Scale the features using StandardScaler.
6. Train Lasso, Ridge, and Elastic Net models with different alpha values.
7. Compare the Mean Squared Error (MSE) of each model.
8. Choose the best alpha value for each model and retrain the models on the full training set.
9. Save the best Ridge regression model using joblib.
