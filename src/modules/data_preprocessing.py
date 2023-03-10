import numpy as np 
import pandas as pd 

def create_combined_features(X):
    X["TotalBsmtArea"] = X["BsmtFinSF1"] + X["BsmtFinSF2"] + X["BsmtUnfSF"]
    X["TotalBaths"] = X["BsmtFullBath"] + (X['BsmtHalfBath'] * 0.5) + X["FullBath"] + (X["HalfBath"] * 0.5)
    X["TotalPorchArea"] = X["OpenPorchSF"] + X["3SsnPorch"] + X["EnclosedPorch"] + X["ScreenPorch"]
    X["TotalIndoorArea"] = X["TotalBsmtSF"] + X["1stFlrSF"] + X["2ndFlrSF"] + X["GarageArea"]
    X["HouseAge"] = X["YrSold"] - X["YearBuilt"]
    return X


def create_features(X):
    X['HasGarage'] = X['GarageArea'].apply(lambda x: 1 if x > 0 else 0)
    X['HasPool'] = X['PoolArea'].apply(lambda x: 1 if x > 0 else 0)
    X['GarageAge'] = X['YrSold'] - X['GarageYrBlt']

    # Reset the index of the DataFrame
    X = X.reset_index(drop=True)
    
    # Only fill in GarageAge if HasGarage is equal to 1
    X.loc[X['HasGarage'] == 1, 'GarageAge'].fillna(X['HouseAge'], inplace=True)
    
    # Fill in GarageAge with 0 if HasGarage is equal to 0
    X.loc[X['HasGarage'] == 0, 'GarageAge'] = 0
    
    X['ReModeled'] = np.where(X.YearRemodAdd == X.YearBuilt, 0, 1)
    X['NewBuild'] = np.where(X.YrSold == X.YearBuilt, 1, 0)
    
    return X


def drop_transformed_columns(X):
    columns_to_drop =  ['BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 
                        'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath', 
                        'OpenPorchSF', '3SsnPorch', 'EnclosedPorch', 'ScreenPorch', 
                        'TotalBsmtSF', 'GrLivArea', '1stFlrSF', '2ndFlrSF']
    X = X.drop(columns=columns_to_drop, axis=1)
    return X

def drop_columns(X):
    columns_to_drop = ['Id', 'YearBuilt', 'YearRemodAdd', 'GarageYrBlt', 'MoSold', 'YrSold']
    X.drop(columns_to_drop, axis=1, inplace=True)
    return X

def label_encode_categorical_features(df, columns):
    from sklearn.preprocessing import LabelEncoder
    for col in columns:
        lbl = LabelEncoder()
        lbl.fit(list(df[col].values))
        df[col] = lbl.transform(list(df[col].values))
    return df


cols = ('FireplaceQu', 'BsmtQual', 'BsmtCond', 'GarageQual', 'GarageCond', 
        'ExterQual', 'ExterCond','HeatingQC', 'PoolQC', 'KitchenQual', 'BsmtFinType1', 
        'BsmtFinType2', 'Functional', 'Fence', 'BsmtExposure', 'GarageFinish', 'LandSlope',
        'LotShape', 'PavedDrive', 'Street', 'Alley', 'CentralAir', 'MSSubClass', 'OverallCond', 
        )



def fill_missing_values_none(data):
    data["PoolQC"] = data["PoolQC"].fillna("None")
    data["MiscFeature"] = data["MiscFeature"].fillna("None")
    data["Alley"] = data["Alley"].fillna("None")
    data["Fence"] = data["Fence"].fillna("None")
    data["FireplaceQu"] = data["FireplaceQu"].fillna("None")
    
    for col in ('GarageType', 'GarageFinish', 'GarageQual', 'GarageCond'):
        data[col] = data[col].fillna("None")
    
    for col in ('BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'):
        data[col] = data[col].fillna("None")
    
    data["MasVnrType"] = data["MasVnrType"].fillna("None")
    data['MSSubClass'] = data['MSSubClass'].fillna("None")
    
    return data

def fill_missing_values_mode(data):
    data['SaleType'] = data['SaleType'].fillna(data['SaleType'].mode()[0])
    data['Exterior1st'] = data['Exterior1st'].fillna(data['Exterior1st'].mode()[0])
    data['Exterior2nd'] = data['Exterior2nd'].fillna(data['Exterior2nd'].mode()[0])
    data['KitchenQual'] = data['KitchenQual'].fillna(data['KitchenQual'].mode()[0])
    data['Electrical'] = data['Electrical'].fillna(data['Electrical'].mode()[0])
    data['MSZoning'] = data['MSZoning'].fillna(data['MSZoning'].mode()[0])
    data["LotFrontage"] = data.groupby("Neighborhood")["LotFrontage"].transform(
        lambda x: x.fillna(x.median()))
    data["Functional"] = data["Functional"].fillna("Typ")
    
    return data


def fill_numeric_features(data):
    numeric_features = ['GarageYrBlt', 'GarageArea', 'GarageCars',
                       'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF',
                       'BsmtFullBath', 'BsmtHalfBath', "MasVnrArea"]
    for col in numeric_features:
        data[col] = data[col].fillna(0)
    return data


#def fill_missing_values(all_data):
    all_data["LotFrontage"] = all_data.groupby("Neighborhood")["LotFrontage"].transform(
        lambda x: x.fillna(x.median()))

    all_data["Functional"] = all_data["Functional"].fillna("Typ")
    
    return all_data

def check_missing_values(df):
    missing_data_count = df.isnull().sum()
    missing_data_percent = df.isnull().sum() / len(df) * 100
    missing_data = pd.DataFrame({
        'Count': missing_data_count,
        'Percent': missing_data_percent
    })
    missing_data = missing_data[missing_data.Count > 0]
    missing_data.sort_values(by='Count', ascending=False, inplace=True)
    return missing_data

import pandas as pd
from scipy.stats import skew

def check_skewness(data):
    from scipy.stats import skew
    numeric_feats = data.dtypes[data.dtypes != "object"].index

    # Check the skew of all numerical features
    skewed_feats = data[numeric_feats].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
    print("\nSkew in numerical features: \n")
    skewness = pd.DataFrame({'Skew' :skewed_feats})
    return skewness.head(20)


from scipy.special import boxcox1p

def box_cox_transform_skewed_features(df, threshold=0.75):
    numeric_feats = df.dtypes[df.dtypes != "object"].index
    skewed_feats = df[numeric_feats].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
    skewed_features = skewed_feats[abs(skewed_feats) > threshold].index
    lam = 0.15
    for feat in skewed_features:
        df[feat] = boxcox1p(df[feat], lam)
    return df

def box_cox_transform(data, skew_threshold=0.75):
    numeric_feats = data.dtypes[data.dtypes != "object"].index

    # Check the skew of all numerical features
    skewed_feats = data[numeric_feats].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
    skewed_feats = skewed_feats[abs(skewed_feats) > skew_threshold]

    print("There are {} skewed numerical features to Box Cox transform".format(skewed_feats.shape[0]))

    from scipy.special import boxcox1p
    skewed_features = skewed_feats.index
    lam = 0.15
    for feat in skewed_features:
        if feat not in ['SalePrice', 'MSSubClass']:
            data[feat] = boxcox1p(data[feat], lam)

    return data


def scale_numerical_features(X):
    import sklearn 
    from sklearn import preprocessing
    from sklearn.preprocessing import RobustScaler
    numerical_cols = list(X.select_dtypes(exclude=['object']).columns)
    scaler = RobustScaler()
    X[numerical_cols] = scaler.fit_transform(X[numerical_cols])
    return X

def one_hot_encode_dataframe(df):
    return pd.get_dummies(df)

def concatenate_dataframes(data, df):
    comp_df = pd.concat([data, df], axis=0)
    return comp_df
