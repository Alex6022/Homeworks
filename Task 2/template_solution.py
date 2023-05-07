# This serves as a template which will guide you through the implementation of this task.  It is advised
# to first read the whole template and get a sense of the overall structure of the code before trying to fill in any of the TODO gaps
# First, we import necessary libraries:
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, RBF, Matern, RationalQuadratic
from sklearn.model_selection import GridSearchCV
from sklearn.metrics._scorer import make_scorer

def calculate_R2(y_pred, y):

    r2 = r2_score(y, y_pred)

    assert np.isscalar(r2)
    return r2

def data_loading():
    """
    This function loads the training and test data, preprocesses it, removes the NaN values and interpolates the missing 
    data using imputation

    Parameters
    ----------
    Returns
    ----------
    X_train: matrix of floats, training input with features
    y_train: array of floats, training output with labels
    X_test: matrix of floats: dim = (100, ?), test input with features
    """
    # Load training data
    train_df = pd.read_csv("train.csv")
    
    
    print("Training data:")
    print("Shape:", train_df.shape)
    print(train_df.head(2))
    print('\n') 
    
    # Load test data
    test_df = pd.read_csv("test.csv")
    """ 
    print("Test data:")
    print(test_df.shape)
    print(test_df.head(2)) """

    # Dummy initialization of the X_train, X_test and y_train   
    """ 
    X_train = np.zeros_like(train_df.drop(['price_CHF'],axis=1))
    y_train = np.zeros_like(train_df['price_CHF'])
    X_test = np.zeros_like(test_df) """

    # TODO: Perform data preprocessing, imputation and extract X_train, y_train and X_test
    
    #Training data: interpolate missing values and replace missing values with mean. Encode categorical data with dummy variables.
    interpol_train_df = train_df.interpolate(method="akima")
    interpol_train_df = interpol_train_df.fillna(train_df.mean())

    dummy_seasons = pd.get_dummies(interpol_train_df['season'])
    interpol_train_df = interpol_train_df.drop(columns=['season']).join(dummy_seasons)

    #Test data: Likewise as for training data except but use the previously encoded dummy variables
    interpol_test_df = test_df.interpolate(method="akima")
    interpol_test_df = interpol_test_df.fillna(test_df.mean())

    interpol_test_df = interpol_test_df.drop(columns=['season']).join(dummy_seasons)

    #Convert created dataframe to numpy to properly return them
    y_train = interpol_train_df['price_CHF'].to_numpy()
    X_train = interpol_train_df.drop(columns=['price_CHF']).to_numpy()
    X_test = interpol_test_df.to_numpy()


    assert (X_train.shape[1] == X_test.shape[1]) and (X_train.shape[0] == y_train.shape[0]) and (X_test.shape[0] == 100), "Invalid data shape"
    return X_train, y_train, X_test

def modeling_and_prediction(X_train, y_train, X_test):
    """
    This function defines the model, fits training data and then does the prediction with the test data 

    Parameters
    ----------
    X_train: matrix of floats, training input with 10 features
    y_train: array of floats, training output
    X_test: matrix of floats: dim = (100, ?), test input with 10 features

    Returns
    ----------
    y_test: array of floats: dim = (100,), predictions on test set
    """

    y_pred=np.zeros(X_test.shape[0])
    #TODO: Define the model and fit it using training data. Then, use test data to make predictions

    n_splits = 9

    custom_scorer = make_scorer(calculate_R2, greater_is_better=True)

    # Define grid for hyperparameter tuning
    gsparams = [{
        "kernel": [RationalQuadratic(length_scale = l) for l in np.logspace(-1, 1, 3)]
    }, {
        "kernel": [RationalQuadratic(alpha = a) for a in np.logspace(-1, 3, 5)]
    }]

    # Perform grid search
    gpr = GaussianProcessRegressor()
    gs = GridSearchCV(gpr, gsparams, scoring=custom_scorer, cv=n_splits)

    gs.fit(X_train, y_train)

    print("Best parameters: ", gs.best_params_)
    print("Best score: ", gs.best_score_)
    best_model = gs.best_estimator_

    y_pred = best_model.predict(X_test)

    X_train = np.array(X_train)

    assert y_pred.shape == (100,), "Invalid data shape"
    return y_pred

# Main function. You don't have to change this
if __name__ == "__main__":
    # Data loading
    X_train, y_train, X_test = data_loading()
    # The function retrieving optimal LR parameters
    y_pred=modeling_and_prediction(X_train, y_train, X_test)
    # Save results in the required format
    dt = pd.DataFrame(y_pred) 
    dt.columns = ['price_CHF']
    dt.to_csv('results.csv', index=False)
    print("\nResults file successfully generated!")

