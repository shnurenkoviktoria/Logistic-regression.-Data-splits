import joblib
import pandas as pd
from sklearn import preprocessing
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, mean_squared_error, log_loss
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# work with data
data = pd.read_csv("kc_house_data.csv")

data = data.drop(["date", "zipcode"], axis=1)

data_norm = preprocessing.normalize(data)
data_norm = pd.DataFrame(data_norm, columns=data.columns)

X = data_norm.drop("price", axis=1)
y = data_norm["price"]

# binarize y to make it a classification problem
y_binary = (y > y.median()).astype(int)

# train test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y_binary, test_size=0.25, random_state=20
)

numeric_features = [
    "sqft_living",
    "sqft_lot",
    "sqft_above",
    "sqft_basement",
    "sqft_living15",
    "sqft_lot15",
]


model = Pipeline(steps=[("classifier", LogisticRegression())])

model.fit(X_train, y_train)
y_pred = model.predict(X_test)


mse = mean_squared_error(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
logloss = log_loss(y_test, y_pred)

print(f"Accuracy: {accuracy}")
print(f"Log Loss: {logloss}")
print(f"MSE: {mse}")

joblib.dump(model, "logistic_regression_model_scaled.joblib")
