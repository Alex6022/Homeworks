# This serves as a template which will guide you through the implementation of this task.  It is advised
# to first read the whole template and get a sense of the overall structure of the code before trying to fill in any of the TODO gaps
# First, we import necessary libraries:
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge, Lasso, BayesianRidge, HuberRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline


def transform_data(X):
    """
    This function transforms the 5 input features of matrix X (x_i denoting the i-th component of X) 
    into 21 new features phi(X) in the following manner:
    5 linear features: phi_1(X) = x_1, phi_2(X) = x_2, phi_3(X) = x_3, phi_4(X) = x_4, phi_5(X) = x_5
    5 quadratic features: phi_6(X) = x_1^2, phi_7(X) = x_2^2, phi_8(X) = x_3^2, phi_9(X) = x_4^2, phi_10(X) = x_5^2
    5 exponential features: phi_11(X) = exp(x_1), phi_12(X) = exp(x_2), phi_13(X) = exp(x_3), phi_14(X) = exp(x_4), phi_15(X) = exp(x_5)
    5 cosine features: phi_16(X) = cos(x_1), phi_17(X) = cos(x_2), phi_18(X) = cos(x_3), phi_19(X) = cos(x_4), phi_20(X) = cos(x_5)
    1 constant features: phi_21(X)=1

    Parameters
    ----------
    X: matrix of floats, dim = (700,5), inputs with 5 features

    Returns
    ----------
    X_transformed: array of floats: dim = (700,21), transformed input with 21 features
    """
    X_transformed = np.zeros((700, 21))
    # TODO: Enter your code here
    X_transformed[:, 0:4] = X[:, 0:4]
    X_transformed[:, 5:9] = X[:, 0:4] ** 2
    X_transformed[:, 10:14] = np.exp(X[:, 0:4])
    X_transformed[:, 15:19] = np.cos(X[:, 0:4])
    X_transformed[:, 20] = 1
    assert X_transformed.shape == (700, 21)
    return X_transformed


def fit(X, y):
    """
    This function receives training data points, transform them, and then fits the linear regression on this 
    transformed data. Finally, it outputs the weights of the fitted linear regression. 

    Parameters
    ----------
    X: matrix of floats, dim = (700,5), inputs with 5 features
    y: array of floats, dim = (700,), input labels)

    Returns
    ----------
    w: array of floats: dim = (21,), optimal parameters of linear regression
    """
    w = np.zeros((21,))
    X_transformed = transform_data(X)
    # TODO: Enter your code here


    clf_select = Lasso(alpha=0.03162277660168379, fit_intercept=False, normalize=True).fit(X_transformed, y)
    model = SelectFromModel(clf_select, prefit=True)
    X_new = model.transform(X_transformed)

    indices = np.zeros((len(X_new[0][:])),)

    for i in range(len(X_new[0][:])):
        indices[i] = np.where(X_transformed[0][:] == X_new[0][i])[0]
        print(X_transformed[0][int(indices[i])], X_new[0][i])

    reg1 = LinearRegression()
    param1 = {"regression" : [reg1], "regression__fit_intercept": [True, False]}

    reg2 = Ridge()
    param2 = {"regression" : [reg2], "regression__alpha": np.logspace(-3, 3, 81), "regression__fit_intercept": [False], "regression__normalize": [True], "regression__solver": ["svd"], "regression__fit_intercept": [False], "regression__normalize": [True]}

    pipeline = Pipeline([("regression", reg1)])
    params = [param1, param2]

    gs = GridSearchCV(pipeline, params, cv=5, scoring="neg_root_mean_squared_error")
    gs.fit(X_new, y)
    print("Best parameters: ", gs.best_params_)
    print("Best score: ", gs.best_score_)
    bestModel = gs.best_estimator_
    selectedWeights = bestModel.named_steps["regression"].coef_
    for i in range(len(selectedWeights)):
        w[int(indices[i])] = selectedWeights[i]
    assert w.shape == (21,)
    return w


# Main function. You don't have to change this
if __name__ == "__main__":
    # Data loading
    data = pd.read_csv("train.csv")
    y = data["y"].to_numpy()
    data = data.drop(columns=["Id", "y"])
    # print a few data samples
    print(data.head())

    X = data.to_numpy()
    # The function retrieving optimal LR parameters
    w = fit(X, y)
    # Save results in the required format
    np.savetxt("./results.csv", w, fmt="%.12f")
