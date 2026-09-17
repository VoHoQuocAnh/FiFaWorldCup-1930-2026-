from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

def run_linear_regression(x_train, x_test, y_train, y_test):
    model = LinearRegression()
    model.fit(x_train, y_train)
    y_predict = model.predict(x_test)
    mse = mean_squared_error(y_test, y_predict)
    r2 = r2_score(y_test, y_predict)
    print("Linear Regression Model:")
    print("Mean Squared Error:", round(mse, 2))
    print("R2 Score:", round(r2, 2))
    return model