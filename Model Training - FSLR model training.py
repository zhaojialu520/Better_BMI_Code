import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso,LassoCV
from sklearn.linear_model import Ridge
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import VotingRegressor
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
from sklearn.metrics import r2_score
from scipy.stats import pearsonr
from sklearn.model_selection import train_test_split, GridSearchCV, RepeatedKFold, cross_val_score
from sklearn.model_selection import cross_validate    ## Import cross-validation package
from sklearn.model_selection import cross_val_score   ## Import cross-validation package
from sklearn.model_selection import KFold   ## Import KFold splitter
from sklearn.metrics import roc_auc_score,make_scorer
from scipy.stats import pearsonr
from sklearn import linear_model
import matplotlib.pyplot as plt
from numpy import genfromtxt
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.svm import SVR      ## Import support vector machine package
from sklearn.model_selection import KFold
# import xgboost as xgb
# from xgboost import XGBRegressor
# import lightgbm as lgb
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from sklearn.model_selection import cross_val_score, KFold
from scipy import stats
from statsmodels.tools.tools import add_constant
from joblib import dump

import warnings
warnings.filterwarnings("ignore")

## Import data
X_train = pd.read_csv("your_path/X_train_All3D_NoScaled.csv",sep=",")
X_train.set_index(X_train.columns[0], inplace= True)  ## Set the first column as the row index of the dataframe
name = X_train.columns.tolist()    ## Extract column names of the dataframe
name = [item.replace('人体外观测量.三维人体扫描分析系统.', '') for item in name]    ## Remove "人体外观测量-三维人体扫描分析系统:" part from column names
name = [item.replace('.cm.', '') for item in name]
X_train.columns = name   ## Replace dataframe column names with simplified names
print(X_train)
X_train = X_train.drop(columns = ['height', 'Waist_To_Hip_Ratio.x.100.'])

X_test = pd.read_csv("your_path/X_test_All3D_NoScaled.csv",sep=",")
X_test.set_index(X_test.columns[0], inplace= True)  ## Set the first column as the row index of the dataframe
X_test.columns = name   ## Replace dataframe column names with simplified names
X_test = X_test.drop(columns = ['height', 'Waist_To_Hip_Ratio.x.100.'])

y_train = pd.read_csv("your_path/y_train_All3D_NoScaled.csv",sep=",")
y_train.set_index(y_train.columns[0], inplace= True)  ## Set the first column as the row index of the dataframe


y_test = pd.read_csv("your_path/y_test_All3D_NoScaled.csv",sep=",")
y_test.set_index(y_test.columns[0], inplace= True)  ## Set the first column as the row index of the dataframe

## Perform stepwise linear regression on training set to get FMI, A_G, FM, LM, VATmass, Android, Gynoid, BFP
def forward_stepwise_regression_bic(data, target):
    ## Initialize data
    initial_features = data.columns.tolist()  ## All features
    selected_features = []   ## Store selected features
    remaining_features = initial_features.copy()  ## Store remaining features
    best_bic = float('inf')   ## Initialize minimum BIC value
    final_model = None  ## Final model

    while remaining_features:
        bic_with_candidates = []

        ## Traverse each unselected feature and add one by one
        for feature in remaining_features:
            ## Combine selected features and current feature
            candidate_features = selected_features + [feature]
            X = data[candidate_features]
            X = add_constant(X)   ## Add constant term
            y = target

            ## Fit regression model
            model = sm.OLS(y, X).fit()

            ## Record BIC value for each model
            bic_with_candidates.append((feature, model.bic))

        ## Sort by BIC from small to large, select the model with minimum BIC
        bic_with_candidates.sort(key = lambda x: x[1])
        best_feature, best_bic_for_feature = bic_with_candidates[0]

        ## If the BIC of this feature is less than the current optimal BIC, select this feature
        if best_bic_for_feature < best_bic:
            selected_features.append(best_feature)
            remaining_features.remove(best_feature)
            best_bic = best_bic_for_feature
        else:
            ## If BIC doesn't improve after adding new feature, stop
            break

    ## Final model
    X_final = data[selected_features]
    X_final = add_constant(X_final)
    y_final = target

    final_model = sm.OLS(y_final, X_final).fit()

    params = final_model.params   ## Get model parameters
    p_values = final_model.pvalues   ## Get P-values
    r_squared = final_model.rsquared   ## Get R2
    y_pred = final_model.predict(X_final)   ## Get predicted Y values
    rmse = np.sqrt(mean_squared_error(y, y_pred))  ## Calculate RMSE
    results = pd.DataFrame({
        'Parameter': params.index,
        'Coefficient': params.values,
        'P-value': p_values.values
    })
    
    r2_row = pd.DataFrame({
        'Parameter': ['R-Squared'],
        'Coefficient': [r_squared],
        'P-Value':[None]
    })   ## Add R2
    
    rmse_row = pd.DataFrame({
        'Parameter': ['RMSE'],
        'Coefficient': [rmse],
        'P-Value': [None]
    })   ## Add RMSE
    
    results = pd.concat([results, r2_row, rmse_row], ignore_index= True)

    results['Coefficient'] = results['Coefficient'].round(3)

    return results, final_model, selected_features


# Multiple linear regression model _ Stepwise forward regression model
## Predict FMI
FMI_params, FMI_forward_linear_model, FMI_selected_feature = forward_stepwise_regression_bic(X_train, y_train['FMI'])
FMI_params.to_csv("your_path/FMI_params.csv", index = False)

## Predict A/G
A_G_params, A_G_forward_linear_model, A_G_selected_feature = forward_stepwise_regression_bic(X_train, y_train['A_G'])
A_G_params.to_csv("your_path/A_G_params.csv", index = False)

## Predict FM
FM_params, FM_forward_linear_model, FM_selected_feature = forward_stepwise_regression_bic(X_train, y_train['FM'])
FM_params.to_csv("your_path/FM_params.csv", index = False)

## Predict LM
LM_params, LM_forward_linear_model, LM_selected_feature = forward_stepwise_regression_bic(X_train, y_train['LM'])
LM_params.to_csv("your_path/LM_params.csv", index = False)

## Predict VATmass
VAT_params, VAT_forward_linear_model, VAT_selected_feature = forward_stepwise_regression_bic(X_train, y_train['VATmass'])
VAT_params.to_csv("your_path/VAT_params.csv", index = False)

## Predict Android
Android_params, Android_forward_linear_model, Android_selected_feature = forward_stepwise_regression_bic(X_train, y_train['Android'])
Android_params.to_csv("your_path/Android_params.csv", index = False)

## Predict Gynoid
Gynoid_params, Gynoid_forward_linear_model, Gynoid_selected_feature = forward_stepwise_regression_bic(X_train, y_train['Gynoid'])
Gynoid_params.to_csv("your_path/Gynoid_params.csv", index = False)

## Predict BFP
BFP_params, BFP_forward_linear_model, BFP_selected_feature = forward_stepwise_regression_bic(X_train, y_train['BFP'])
BFP_params.to_csv("your_path/BFP_params.csv", index = False)

## Save multiple models and parameters to a dictionary
AA_20240105_All_models = {
    'FMI_forward_linear_model':{ 'Forward_linear_model': FMI_forward_linear_model, 'Selected_feature': FMI_selected_feature},
    'A_G_forward_linear_model':{ 'Forward_linear_model': A_G_forward_linear_model, 'Selected_feature': A_G_selected_feature},
    'FM_forward_linear_model':{ 'Forward_linear_model' : FM_forward_linear_model, 'Selected_feature': FM_selected_feature},
    'LM_forward_linear_model':{ 'Forward_linear_model' : LM_forward_linear_model, 'Selected_feature': LM_selected_feature},
    'VAT_forward_linear_model':{ 'Forward_linear_model' : VAT_forward_linear_model, 'Selected_feature': VAT_selected_feature},
    'Android_forward_linear_model':{ 'Forward_linear_model' : Android_forward_linear_model, 'Selected_feature': Android_selected_feature},
    'Gynoid_forward_linear_model':{ 'Forward_linear_model' : Gynoid_forward_linear_model, 'Selected_feature': Gynoid_selected_feature},
    'BFP_forward_linear_model': { 'Forward_linear_model' : BFP_forward_linear_model, 'Selected_feature': BFP_selected_feature}
}

## Save dictionary
dump(AA_20240105_All_models, 'your_path/AA_20240105_All_models.joblib')
