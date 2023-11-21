# Імпорт необхідних бібліотек
import joblib
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

data = pd.read_csv("kc_house_data.csv")

data = data.drop(["date", "zipcode"], axis=1)

data_norm = preprocessing.normalize(data)
data_norm = pd.DataFrame(data_norm, columns=data.columns)

X = data_norm.drop("price", axis=1)
y = data_norm["price"]

# Розбиття даних на train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# Порівняння точності моделей із регуляризаціями Lasso & Ridge & Elastic Net
alphas = [0.001, 0.01, 0.1, 1, 10]

for alpha in alphas:
    # Lasso Regression
    lasso_model = Lasso(alpha=alpha)
    lasso_model.fit(X_train, y_train)
    lasso_pred = lasso_model.predict(X_test)
    lasso_mse = mean_squared_error(y_test, lasso_pred)
    print(f"Lasso Regression with alpha={alpha}: Mean Squared Error = {lasso_mse}")

    # Ridge Regression
    ridge_model = Ridge(alpha=alpha)
    ridge_model.fit(X_train, y_train)
    ridge_pred = ridge_model.predict(X_test)
    ridge_mse = mean_squared_error(y_test, ridge_pred)
    print(f"Ridge Regression with alpha={alpha}: Mean Squared Error = {ridge_mse}")

    # Elastic Net
    elastic_net_model = ElasticNet(alpha=alpha, l1_ratio=0.5)
    elastic_net_model.fit(X_train, y_train)
    elastic_net_pred = elastic_net_model.predict(X_test)
    elastic_net_mse = mean_squared_error(y_test, elastic_net_pred)
    print(f"Elastic Net with alpha={alpha}: Mean Squared Error = {elastic_net_mse}")


print("-----------------------------------------")

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.4, random_state=42
)
X_cv, X_test, y_cv, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_cv_scaled = scaler.transform(X_cv)
X_test_scaled = scaler.transform(X_test)

param_grid = {
    "alpha": [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 2, 3, 4, 10]
}
degrees = [1, 2]

best_degree = None
best_mse = float("inf")

for degree in degrees:
    poly = PolynomialFeatures(degree=degree)
    X_train_poly = poly.fit_transform(X_train_scaled)
    X_cv_poly = poly.transform(X_cv_scaled)
    X_test_poly = poly.transform(X_test_scaled)

    ridge = Ridge()

    grid_search = GridSearchCV(
        ridge, param_grid, scoring="neg_mean_squared_error", cv=5
    )
    grid_search.fit(X_cv_poly, y_cv)

    best_alpha = grid_search.best_params_["alpha"]
    print(f"Best alpha for degree={degree}: {best_alpha}")

    ridge_final_model = Ridge(alpha=best_alpha)
    ridge_final_model.fit(X_train_poly, y_train)
    ridge_final_pred = ridge_final_model.predict(X_test_poly)
    ridge_final_mse = mean_squared_error(y_test, ridge_final_pred)

    print(
        f"Ridge Regression with degree={degree} and best alpha={best_alpha}: Mean Squared Error on Test Set = {ridge_final_mse}"
    )

    if ridge_final_mse < best_mse:
        best_mse = ridge_final_mse
        best_degree = degree

joblib.dump(ridge_final_model, "ridge_regression_model.joblib")

print(f"Best Degree: {best_degree}")
