from sklearn.linear_model import ElasticNet, Lasso,  BayesianRidge, LassoLarsIC
from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import lightgbm as lgb
import pandas as pd 
import numpy as np

def split_train_test(X, test_df):
    ntest = len(test_df)
    X_train = X.iloc[:-ntest, :]
    X_test = X.iloc[-ntest:, :]
    
    return X_train, X_test


def rmsle_cv(df, model, y):
    n_folds = 5
    kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(df.values)
    rmse= np.sqrt(-cross_val_score(model, df.values, y, scoring="neg_mean_squared_error", cv = kf))
    return(rmse)

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn.model_selection import KFold, GridSearchCV
import xgboost as xgb

def tune_xgb_regressor(X, y, numerical_columns):
    # Create a ColumnTransformer that performs scaling and normalization
    preprocessor = ColumnTransformer(
        transformers=[
            ('scale', StandardScaler(), numerical_columns),
            ('normalize', Normalizer(), numerical_columns)
        ])

    # Create a pipeline that includes the preprocessor and XGBoost regressor
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', xgb.XGBRegressor())
    ])

    # Define the hyperparameters to tune
    param_grid = {
        'regressor__n_estimators': [100, 200, 300],
        'regressor__max_depth': [3, 5, 7],
        'regressor__learning_rate': [0.1, 0.01, 0.001],
        'regressor__subsample': [0.5, 0.7, 1.0],
        'regressor__colsample_bytree': [0.5, 0.7, 1.0]
    }

    # Perform grid search with cross-validation
    kfold = KFold(n_splits=10)
    grid_search = GridSearchCV(pipeline, param_grid, cv=kfold, scoring='neg_mean_squared_error')
    grid_search.fit(X, y)

    # Print the best parameters and the best score
    print("Best parameters: ", grid_search.best_params_)
    print("Best score (RMSE): %.2f" % np.sqrt(np.abs(grid_search.best_score_)))
    
    return grid_search


from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFE

def select_features(X, y, n_features_to_select= 10):
    model = RandomForestRegressor()
    rfe = RFE(model, n_features_to_select=n_features_to_select)
    rfe.fit(X, y)

    return X.columns[rfe.support_]



