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
import xgboost as xgb
from xgboost import XGBRegressor
import lightgbm as lgb
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from sklearn.model_selection import cross_val_score, KFold
from scipy import stats
from statsmodels.tools.tools import add_constant

from sklearn.model_selection import RandomizedSearchCV
from sklearn.linear_model import RidgeCV

import pprint
import json

import warnings
warnings.filterwarnings("ignore")


## ----------------------------------------------------------Import Data---------------------------------------------------------

## Import data
X_train = pd.read_csv("your_path/X_train_All3D_NoScaled.csv",sep=",")
X_train.set_index(X_train.columns[0], inplace= True)  ## Set the first column as the row index of the dataframe
name = X_train.columns.tolist()    ## Extract column names of the dataframe
name = [item.replace('人体外观测量.三维人体扫描分析系统.', '') for item in name]    ## Remove "人体外观测量-三维人体扫描分析系统:" part from column names
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

# print("X_train_All3D_scaled : ", X_train)
# print("X_test_All3D_scaled : ", X_test)
# print("y_train_All3D_scaled : ", y_train)
# print("y_test_All3D_scaled : ", y_test)


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

    return final_model, selected_features

def get_hyperparameter_grids(model_name, random_state=None):
    """Unified hyperparameter search space"""
    param_grids = {
        'lasso': {
            'alpha': np.logspace(-5, 2, 100)
        },
        'ridge': {
            'alpha': np.logspace(-3, 3, 100)
        },
        'elasticnet': {
            'alpha': np.logspace(-4, 2, 50),
            'l1_ratio': np.linspace(0.1, 1.0, 10)
        },
        'tree': {
            'max_depth': [3, 5, 7, 10, 15, 20, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['auto', 'sqrt', 'log2']
        },
        'randomforest': {
            'n_estimators': [50, 100, 200, 500],
            'max_depth': [5, 10, 15, 20, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['auto', 'sqrt', 'log2']
        },
        'gbdt': {
            'n_estimators': [50, 100, 200, 500],
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
            'max_depth': [3, 5, 7, 9],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        },
        'xgboost': {
            'n_estimators': [50, 100, 200, 500],
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
            'max_depth': [3, 5, 7, 9],
            'subsample': [0.8, 0.9, 1.0],
            'colsample_bytree': [0.8, 0.9, 1.0]
        }
    }
    return param_grids.get(model_name, {})

def nested_cv_model_selection(X, y, model, param_grid, outer_cv=5, inner_cv=3, n_iter = 10, 
                              random_state=None):
    """Nested cross-validation for hyperparameter optimization"""
    outer_kf = KFold(n_splits=outer_cv, shuffle=True, random_state=random_state)
    
    best_params_list = []
    best_scores = []
    
    for train_idx, test_idx in outer_kf.split(X):
        X_outer_train, X_outer_test = X.iloc[train_idx], X.iloc[test_idx]
        y_outer_train, y_outer_test = y.iloc[train_idx], y.iloc[test_idx]
        
        # Inner cross-validation for hyperparameter search
        inner_search = RandomizedSearchCV(
            model, param_grid, 
            n_iter=n_iter,  # Number of iterations for random search
            cv=inner_cv, 
            scoring='neg_mean_squared_error',
            n_jobs=-1,
            random_state=random_state
        )
        
        inner_search.fit(X_outer_train, y_outer_train.values.ravel() if len(y_outer_train.shape) > 1 else y_outer_train)
        
        best_params_list.append(inner_search.best_params_)
        best_scores.append(inner_search.best_score_)
    
    return best_params_list, best_scores

def optimized_regression_model(X, Y, random_state=None, use_nested_cv=True):
    """Optimized regression model with unified hyperparameter optimization strategy and nested cross-validation"""
    
    # Set up cross-validation
    kf = KFold(n_splits=10, shuffle=True, random_state=random_state)
    
    # Initialize result storage
    results = {
        'Linear': {'r2': [], 'rmse': []},
        'Lasso': {'r2': [], 'rmse': []},
        'Ridge': {'r2': [], 'rmse': []},
        'ElasticNet': {'r2': [], 'rmse': []},
        'Tree': {'r2': [], 'rmse': []},
        'RandomForest': {'r2': [], 'rmse': []},
        'GBDT': {'r2': [], 'rmse': []},
        'XGBoost': {'r2': [], 'rmse': []}
    }
    
    # Store best parameters (for analysis)
    best_params_history = {
        'Lasso': [], 'Ridge': [], 'ElasticNet': [], 'Tree': [], 'RandomForest': [], 'GBDT': [], 'XGBoost': []
    }

    for fold, (train_index, test_index) in enumerate(kf.split(X)):
        print(f"Processing fold {fold + 1}/10")
        
        # Split training and test sets
        X_train_inner, X_test_inner = X.iloc[train_index], X.iloc[test_index]
        y_train_inner, y_test_inner = Y.iloc[train_index], Y.iloc[test_index]
        
        """
        # Data standardization
        scaler_X = StandardScaler()
        scaler_y = StandardScaler()
        
        X_train_scaled = pd.DataFrame(
            scaler_X.fit_transform(X_train), 
            columns=X.columns, 
            index=X_train.index
        )
        X_test_scaled = pd.DataFrame(
            scaler_X.transform(X_test), 
            columns=X.columns, 
            index=X_test.index
        )
        
        # For regression targets, standardize if needed
        if len(Y.shape) > 1 and Y.shape[1] > 1:
            y_train_scaled = y_train.copy()
            y_test_scaled = y_test.copy()
        else:
            y_train_scaled = pd.DataFrame(
                scaler_y.fit_transform(y_train.values.reshape(-1, 1)), 
                columns=Y.columns,
                index=y_train.index
            )
            y_test_scaled = pd.DataFrame(
                scaler_y.transform(y_test.values.reshape(-1, 1)), 
                columns=Y.columns,
                index=y_test.index
            )
        """

      
        # 1. Multiple linear regression model _ Stepwise forward regression model
        forward_linear_model, selected_feature = forward_stepwise_regression_bic(X_train_inner, y_train_inner)
        
       # Record R2 and RMSE for each cross-validation fold
        X_test_selected = X_test_inner[list(selected_feature)]
        X_test_selected = add_constant(X_test_selected)
        Linear_y_pred = forward_linear_model.predict(X_test_selected)  # Predict new points
        Linear_r_squre = r2_score(y_test_inner, Linear_y_pred)  # R2
        Linear_RMSE = np.sqrt(mean_squared_error(y_test_inner, Linear_y_pred))
        results['Linear']['r2'].append(Linear_r_squre )
        results['Linear']['rmse'].append(Linear_RMSE)
        
        # 2. Lasso regression (unified hyperparameter optimization)
        lasso_param_grid = get_hyperparameter_grids('lasso')
        if use_nested_cv and lasso_param_grid:
            
            # Inner cross-validation for hyperparameter search
            inner_search_lasso = RandomizedSearchCV(
                Lasso(max_iter=10000, random_state=random_state), lasso_param_grid, 
                n_iter=10,  # Number of iterations for random search
                cv=5, 
                scoring='neg_mean_squared_error',
                n_jobs=-1,
                random_state=random_state
            )
            
            inner_search_lasso.fit(X_train_inner, y_train_inner.values.ravel() if len(y_train_inner.shape) > 1 else y_train_inner)
            lasso_best_params = inner_search_lasso.best_params_
            best_params_history['Lasso'].append({'alpha': lasso_best_params})
            
            # Retrain model with best parameters
            lasso_model = Lasso(**lasso_best_params, max_iter=10000, random_state=random_state)
            
        else:
            lasso_model = LassoCV(alphas=np.logspace(-5, 2, 100), max_iter=10000, cv=5, random_state=random_state)
        
        
        lasso_model.fit(X_train_inner, y_train_inner.values.ravel())
        
        # Outer test set evaluation
        lasso_pred = lasso_model.predict(X_test_inner)
        results['Lasso']['r2'].append(r2_score(y_test_inner, lasso_pred))
        results['Lasso']['rmse'].append(np.sqrt(mean_squared_error(y_test_inner, lasso_pred)))

        # 3. Ridge regression
        ridge_param_grid = get_hyperparameter_grids('ridge')
        if use_nested_cv and ridge_param_grid:
            # Inner cross-validation for hyperparameter search
            inner_search_ridge = RandomizedSearchCV(
                Ridge(), ridge_param_grid, 
                n_iter=10,  # Number of iterations for random search
                cv=5, 
                scoring='neg_mean_squared_error',
                n_jobs=-1,
                random_state=random_state
            )    
            inner_search_ridge.fit(X_train_inner, y_train_inner.values.ravel() if len(y_train_inner.shape) > 1 else y_train_inner)
            ridge_best_params = inner_search_ridge.best_params_
            
            best_params_history['Ridge'].append({'alpha': ridge_best_params})
            
            # Retrain model with best parameters
            ridge_model = Ridge(**ridge_best_params, random_state=random_state)
        else:
            ridge_model = RidgeCV(alphas=np.logspace(-3, 3, 100))
        
        ridge_model.fit(X_train_inner, y_train_inner.values.ravel())
        
        # Outer test set evaluation
        ridge_pred = ridge_model.predict(X_test_inner)
        results['Ridge']['r2'].append(r2_score(y_test_inner, ridge_pred))
        results['Ridge']['rmse'].append(np.sqrt(mean_squared_error(y_test_inner, ridge_pred)))

        # 4. Elastic Net (using unified hyperparameter search)
        enet_param_grid = get_hyperparameter_grids('elasticnet')
        enet_model = ElasticNet(max_iter=10000, random_state=random_state)
        if use_nested_cv:
            enet_model = ElasticNet(max_iter=10000, random_state=random_state)
            # Inner cross-validation for hyperparameter search
            inner_search_enet = RandomizedSearchCV(
                enet_model, enet_param_grid, 
                n_iter=10,  # Number of iterations for random search
                cv=5, 
                scoring='neg_mean_squared_error',
                n_jobs=-1,
                random_state=random_state
            )    
            inner_search_enet.fit(X_train_inner, y_train_inner.values.ravel() if len(y_train_inner.shape) > 1 else y_train_inner)
            enet_best_params = inner_search_enet.best_params_
            
            best_params_history['ElasticNet'].append(enet_best_params)
            
            # Retrain model with best parameters
            enet_model = ElasticNet(**enet_best_params, max_iter=10000, random_state=random_state)
          
        else:
            enet_model = ElasticNet(alpha=1.0, l1_ratio=0.5, max_iter=10000, random_state=random_state)
        
        enet_model.fit(X_train_inner, y_train_inner.values.ravel())
        
        # Outer test set evaluation
        enet_pred = enet_model.predict(X_test_inner)
        results['ElasticNet']['r2'].append(r2_score(y_test_inner, enet_pred))
        results['ElasticNet']['rmse'].append(np.sqrt(mean_squared_error(y_test_inner, enet_pred)))

        # 5. Decision Tree (tree-based model hyperparameter optimization)
        tree_param_grid = get_hyperparameter_grids('tree')
        if use_nested_cv:
            tree_model = DecisionTreeRegressor(random_state=random_state)
            inner_search_tree = RandomizedSearchCV(
                tree_model, tree_param_grid, n_iter=19, cv=5,
                scoring='neg_mean_squared_error', n_jobs=-1, random_state=random_state
            )
            inner_search_tree.fit(X_train_inner, y_train_inner.values.ravel())
            best_tree_params = inner_search_tree.best_params_
            best_params_history['Tree'].append(best_tree_params)
            # Retrain model with best parameters
            tree_model = DecisionTreeRegressor(**best_tree_params, random_state=random_state)
           
        else:
            tree_model = DecisionTreeRegressor(max_depth=10, random_state=random_state)
        
        tree_model.fit(X_train_inner, y_train_inner.values.ravel())
        
        # Outer test set evaluation
        tree_pred = tree_model.predict(X_test_inner)
        results['Tree']['r2'].append(r2_score(y_test_inner, tree_pred))
        results['Tree']['rmse'].append(np.sqrt(mean_squared_error(y_test_inner, tree_pred)))

        # 7. Random Forest (tree-based model hyperparameter optimization)
        rf_param_grid = get_hyperparameter_grids('randomforest')
        if use_nested_cv:
            rf_model = RandomForestRegressor(random_state=random_state, n_jobs=-1)
            inner_search_rf = RandomizedSearchCV(
                rf_model, rf_param_grid, n_iter=10, cv=5,  # Reduce iterations and folds to speed up
                scoring='neg_mean_squared_error', n_jobs=-1, random_state=random_state
            )
            inner_search_rf.fit(X_train_inner, y_train_inner.values.ravel())
            best_rf_params = inner_search_rf.best_params_
            best_params_history['RandomForest'].append(best_rf_params)
            
            # Retrain model with best parameters
            rf_model = RandomForestRegressor(**best_rf_params, random_state=random_state, n_jobs=-1)
            
        else:
            rf_model = RandomForestRegressor(n_estimators=100, random_state=random_state, n_jobs=-1)
        
        rf_model.fit(X_train_inner, y_train_inner.values.ravel())
        
        # Outer test set evaluation
        rf_pred = rf_model.predict(X_test_inner)
        results['RandomForest']['r2'].append(r2_score(y_test_inner, rf_pred))
        results['RandomForest']['rmse'].append(np.sqrt(mean_squared_error(y_test_inner, rf_pred)))

        # 8. GBDT (tree-based model hyperparameter optimization)
        gbdt_param_grid = get_hyperparameter_grids('gbdt')
        if use_nested_cv:
            gbdt_model = GradientBoostingRegressor(random_state=random_state)
            inner_search_gbdt = RandomizedSearchCV(
                gbdt_model, gbdt_param_grid, n_iter=10, cv=5,
                scoring='neg_mean_squared_error', n_jobs=-1, random_state=random_state
            )
            inner_search_gbdt.fit(X_train_inner, y_train_inner.values.ravel())
            best_gbdt_params = inner_search_gbdt.best_params_
            best_params_history['GBDT'].append(best_gbdt_params)

            # Retrain model with best parameters
            gbdt_model = GradientBoostingRegressor(**best_gbdt_params, random_state=random_state)
           
        else:
            gbdt_model = GradientBoostingRegressor(n_estimators=100, random_state=random_state)
        
        gbdt_model.fit(X_train_inner, y_train_inner.values.ravel())
        
        # Outer test set evaluation
        gbdt_pred = gbdt_model.predict(X_test_inner)
        results['GBDT']['r2'].append(r2_score(y_test_inner, gbdt_pred))
        results['GBDT']['rmse'].append(np.sqrt(mean_squared_error(y_test_inner, gbdt_pred)))

        # 9. XGBoost (tree-based model hyperparameter optimization)
        xgb_param_grid = get_hyperparameter_grids('xgboost')
        if use_nested_cv:
            xgb_model = XGBRegressor(random_state=random_state, n_jobs=-1)
            inner_search_xgb = RandomizedSearchCV(
                xgb_model, xgb_param_grid, n_iter=10, cv=5,
                scoring='neg_mean_squared_error', n_jobs=-1, random_state=random_state
            )
            inner_search_xgb.fit(X_train_inner, y_train_inner.values.ravel())
            best_xgb_params = inner_search_xgb.best_params_
            best_params_history['XGBoost'].append(best_xgb_params)
            
            # Retrain model with best parameters
            xgb_model = XGBRegressor(**best_xgb_params, random_state=random_state, n_jobs=-1)
            
        else:
            xgb_model = XGBRegressor(n_estimators=100, random_state=random_state, n_jobs=-1)
        
        xgb_model.fit(X_train_inner, y_train_inner.values.ravel())
        
        # Outer test set evaluation
        xgb_pred = xgb_model.predict(X_test_inner)
        results['XGBoost']['r2'].append(r2_score(y_test_inner, xgb_pred))
        results['XGBoost']['rmse'].append(np.sqrt(mean_squared_error(y_test_inner, xgb_pred)))
    return results, best_params_history

def get_model_summary(results, best_params_history):
    """Correct model summary: select parameters with best performance"""
    
    print("=== Model Generalization Performance and Best Parameters ===\n")
    
    model_summary = {}
    
    for model_name in results:
        # Calculate average performance
        mean_r2 = np.mean(results[model_name]['r2'])
        mean_rmse = np.mean(results[model_name]['rmse'])
        
        # Find the fold with best performance
        best_fold_idx = np.argmax(results[model_name]['r2'])  # The fold with highest R²
        
        # Get best parameters (parameters from the fold with best performance)
        if model_name in best_params_history and best_params_history[model_name]:
            if len(best_params_history[model_name]) > best_fold_idx:
                best_params = best_params_history[model_name][best_fold_idx]
            else:
                best_params = best_params_history[model_name][0]  # Use first fold as fallback
        else:
            best_params = "Default parameters"
        
        # Record performance of best fold
        best_r2 = results[model_name]['r2'][best_fold_idx]
        best_rmse = results[model_name]['rmse'][best_fold_idx]
        
        model_summary[model_name] = {
            'mean_r2': mean_r2,
            'mean_rmse': mean_rmse,
            'best_r2': best_r2,
            'best_rmse': best_rmse,
            'best_fold': best_fold_idx,
            'best_params': best_params
        }
        
        # Print results
        print(f"{model_name:15s} | Mean R²: {mean_r2:.4f} | Mean RMSE: {mean_rmse:.4f}")
        print(f"{' ':15s} | Best R²: {best_r2:.4f} (Fold {best_fold_idx+1})")
        print(f"{' ':15s} | Best Parameters: {best_params}")
        print("-" * 70)
    
    return model_summary

# Use your data


################################## FMI ##################################
print("All_People: ")
print("All_FMI: ")
FMI_model_results, FMI_model_best_params = optimized_regression_model(X_train, y_train['FMI'].to_frame(), random_state = 42, use_nested_cv=True)

pprint.pprint(FMI_model_results)
# Convert nested dictionary to DataFrame
records = []
for model_name, metrics in FMI_model_results.items():
    for i in range(len(metrics['r2'])):
        records.append({
            'model': model_name,
            'fold': i + 1,
            'r2': metrics['r2'][i],
            'rmse': metrics['rmse'][i]
        })

FMI_model_results_df = pd.DataFrame(records)

FMI_model_results_df.to_csv("your_path/FMI_model_results.csv", index = True)


# Save dictionary
with open('your_path/FMI_model_params.json', 'w', encoding='utf-8') as f:
    json.dump(FMI_model_best_params, f, indent=4, ensure_ascii=False)


# Read dictionary
with open('your_path/FMI_model_params.json', 'r', encoding='utf-8') as f:
    FMI_model_best_params = json.load(f)
    
FMI_model_summary = get_model_summary(FMI_model_results, FMI_model_best_params)

# Convert nested dictionary to flattened DataFrame
records = []
for model_name, model_info in FMI_model_summary.items():
    record = {
        'model': model_name,
        'best_fold': model_info['best_fold'],
        'best_r2': model_info['best_r2'],
        'best_rmse': model_info['best_rmse'],
        'mean_r2': model_info['mean_r2'],
        'mean_rmse': model_info['mean_rmse']
    }
    
    # Process parameters (convert to string or further expand)
    if isinstance(model_info['best_params'], dict):
        # Method 1: Convert parameter dictionary to string
        record['best_params'] = str(model_info['best_params'])
        
    else:
        record['best_params'] = model_info['best_params']
    
    records.append(record)

FMI_summary_df = pd.DataFrame(records)

FMI_summary_df.to_csv("your_path/FMI_model_summary.csv", index = True)


################################# A_G ##############################
print("All_A/G: ")
A_G_model_results, A_G_model_best_params = optimized_regression_model(X_train, y_train['A_G'].to_frame(), random_state= 42, use_nested_cv = True)

# Convert nested dictionary to DataFrame
records = []
for model_name, metrics in A_G_model_results.items():
    for i in range(len(metrics['r2'])):
        records.append({
            'model': model_name,
            'fold': i + 1,
            'r2': metrics['r2'][i],
            'rmse': metrics['rmse'][i]
        })

A_G_model_results_df = pd.DataFrame(records)

A_G_model_results_df.to_csv("your_path/A_G_model_results.csv", index = True)

# Save dictionary
with open('your_path/A_G_model_params.json', 'w', encoding='utf-8') as f:
    json.dump(A_G_model_best_params, f, indent=4, ensure_ascii=False)


A_G_model_summary = get_model_summary(A_G_model_results, A_G_model_best_params)

# Convert nested dictionary to flattened DataFrame
records = []
for model_name, model_info in A_G_model_summary.items():
    record = {
        'model': model_name,
        'best_fold': model_info['best_fold'],
        'best_r2': model_info['best_r2'],
        'best_rmse': model_info['best_rmse'],
        'mean_r2': model_info['mean_r2'],
        'mean_rmse': model_info['mean_rmse']
    }
    
    # Process parameters (convert to string or further expand)
    if isinstance(model_info['best_params'], dict):
        # Method 1: Convert parameter dictionary to string
        record['best_params'] = str(model_info['best_params'])
        
    else:
        record['best_params'] = model_info['best_params']
    
    records.append(record)

A_G_summary_df = pd.DataFrame(records)

A_G_summary_df.to_csv("your_path/A_G_model_summary.csv", index = True)


################################## FM #########################################3
print("All_FM: ")
FM_model_results, FM_model_best_params = optimized_regression_model(X_train, y_train['FM'].to_frame(), random_state= 42, use_nested_cv = True)

# Convert nested dictionary to DataFrame
records = []
for model_name, metrics in FM_model_results.items():
    for i in range(len(metrics['r2'])):
        records.append({
            'model': model_name,
            'fold': i + 1,
            'r2': metrics['r2'][i],
            'rmse': metrics['rmse'][i]
        })

FM_model_results_df = pd.DataFrame(records)

FM_model_results_df.to_csv("your_path/FM_model_results.csv", index = True)

# Save dictionary
with open('your_path/FM_model_params.json', 'w', encoding='utf-8') as f:
    json.dump(FM_model_best_params, f, indent=4, ensure_ascii=False)
    
FM_model_summary = get_model_summary(FM_model_results, FM_model_best_params)

# Convert nested dictionary to flattened DataFrame
records = []
for model_name, model_info in FM_model_summary.items():
    record = {
        'model': model_name,
        'best_fold': model_info['best_fold'],
        'best_r2': model_info['best_r2'],
        'best_rmse': model_info['best_rmse'],
        'mean_r2': model_info['mean_r2'],
        'mean_rmse': model_info['mean_rmse']
    }
    
    # Process parameters (convert to string or further expand)
    if isinstance(model_info['best_params'], dict):
        # Method 1: Convert parameter dictionary to string
        record['best_params'] = str(model_info['best_params'])
        
    else:
        record['best_params'] = model_info['best_params']
    
    records.append(record)

FM_summary_df = pd.DataFrame(records)

FM_summary_df.to_csv("your_path/FM_model_summary.csv", index = True)


#############################  LM  #########################################

print("All_LM: ")
LM_model_results, LM_model_best_params = optimized_regression_model(X_train, y_train['LM'].to_frame(), random_state= 42, use_nested_cv = True)

# Convert nested dictionary to DataFrame
records = []
for model_name, metrics in LM_model_results.items():
    for i in range(len(metrics['r2'])):
        records.append({
            'model': model_name,
            'fold': i + 1,
            'r2': metrics['r2'][i],
            'rmse': metrics['rmse'][i]
        })

LM_model_results_df = pd.DataFrame(records)

LM_model_results_df.to_csv("your_path/LM_model_results.csv", index = True)

# Save dictionary
with open('your_path/LM_model_params.json', 'w', encoding='utf-8') as f:
    json.dump(LM_model_best_params, f, indent=4, ensure_ascii=False)
    
LM_model_summary = get_model_summary(LM_model_results, LM_model_best_params)

# Convert nested dictionary to flattened DataFrame
records = []
for model_name, model_info in LM_model_summary.items():
    record = {
        'model': model_name,
        'best_fold': model_info['best_fold'],
        'best_r2': model_info['best_r2'],
        'best_rmse': model_info['best_rmse'],
        'mean_r2': model_info['mean_r2'],
        'mean_rmse': model_info['mean_rmse']
    }
    
    # Process parameters (convert to string or further expand)
    if isinstance(model_info['best_params'], dict):
        # Method 1: Convert parameter dictionary to string
        record['best_params'] = str(model_info['best_params'])
        
    else:
        record['best_params'] = model_info['best_params']
    
    records.append(record)

LM_summary_df = pd.DataFrame(records)

LM_summary_df.to_csv("your_path/LM_model_summary.csv", index = True)

#################################  VATmass #############################                                
print("All_VAT: ")
VAT_model_results, VAT_model_best_params = optimized_regression_model(X_train, y_train['VATmass'].to_frame(), random_state= 42, use_nested_cv = True)

# Convert nested dictionary to DataFrame
records = []
for model_name, metrics in VAT_model_results.items():
    for i in range(len(metrics['r2'])):
        records.append({
            'model': model_name,
            'fold': i + 1,
            'r2': metrics['r2'][i],
            'rmse': metrics['rmse'][i]
        })

VAT_model_results_df = pd.DataFrame(records)

VAT_model_results_df.to_csv("your_path/VAT_model_results.csv", index = True)

# Save dictionary
with open('your_path/VAT_model_params.json', 'w', encoding='utf-8') as f:
    json.dump(VAT_model_best_params, f, indent=4, ensure_ascii=False)
   
VAT_model_summary = get_model_summary(VAT_model_results, VAT_model_best_params)

# Convert nested dictionary to flattened DataFrame
records = []
for model_name, model_info in VAT_model_summary.items():
    record = {
        'model': model_name,
        'best_fold': model_info['best_fold'],
        'best_r2': model_info['best_r2'],
        'best_rmse': model_info['best_rmse'],
        'mean_r2': model_info['mean_r2'],
        'mean_rmse': model_info['mean_rmse']
    }
    
    # Process parameters (convert to string or further expand)
    if isinstance(model_info['best_params'], dict):
        # Method 1: Convert parameter dictionary to string
        record['best_params'] = str(model_info['best_params'])
        
    else:
        record['best_params'] = model_info['best_params']
    
    records.append(record)

VAT_summary_df = pd.DataFrame(records)

VAT_summary_df.to_csv("your_path/VAT_model_summary.csv", index = True)


################################ Android ################################
print("All_Android: ")
Android_model_results, Android_model_best_params = optimized_regression_model(X_train, y_train['Android'].to_frame(), random_state= 42, use_nested_cv = True)

# Convert nested dictionary to DataFrame
records = []
for model_name, metrics in Android_model_results.items():
    for i in range(len(metrics['r2'])):
        records.append({
            'model': model_name,
            'fold': i + 1,
            'r2': metrics['r2'][i],
            'rmse': metrics['rmse'][i]
        })

Android_model_results_df = pd.DataFrame(records)

Android_model_results_df.to_csv("your_path/Android_model_results.csv", index = True)

# Save dictionary
with open('your_path/Android_model_params.json', 'w', encoding='utf-8') as f:
    json.dump(Android_model_best_params, f, indent=4, ensure_ascii=False)

Android_model_summary = get_model_summary(Android_model_results, Android_model_best_params)

# Convert nested dictionary to flattened DataFrame
records = []
for model_name, model_info in Android_model_summary.items():
    record = {
        'model': model_name,
        'best_fold': model_info['best_fold'],
        'best_r2': model_info['best_r2'],
        'best_rmse': model_info['best_rmse'],
        'mean_r2': model_info['mean_r2'],
        'mean_rmse': model_info['mean_rmse']
    }
    
    # Process parameters (convert to string or further expand)
    if isinstance(model_info['best_params'], dict):
        # Method 1: Convert parameter dictionary to string
        record['best_params'] = str(model_info['best_params'])
        
    else:
        record['best_params'] = model_info['best_params']
    
    records.append(record)

Android_summary_df = pd.DataFrame(records)

Android_summary_df.to_csv("your_path/Android_model_summary.csv", index = True)


############################## Gynoid ####################################
print("All_Gynoid: ")
Gynoid_model_results, Gynoid_model_best_params = optimized_regression_model(X_train, y_train['Gynoid'].to_frame(), random_state= 42, use_nested_cv = True)

# Convert nested dictionary to DataFrame
records = []
for model_name, metrics in Gynoid_model_results.items():
    for i in range(len(metrics['r2'])):
        records.append({
            'model': model_name,
            'fold': i + 1,
            'r2': metrics['r2'][i],
            'rmse': metrics['rmse'][i]
        })

Gynoid_model_results_df = pd.DataFrame(records)

Gynoid_model_results_df.to_csv("your_path/Gynoid_model_results.csv", index = True)

# Save dictionary
with open('your_path/Gynoid_model_params.json', 'w', encoding='utf-8') as f:
    json.dump(Gynoid_model_best_params, f, indent=4, ensure_ascii=False)
    
Gynoid_model_summary = get_model_summary(Gynoid_model_results, Gynoid_model_best_params)

# Convert nested dictionary to flattened DataFrame
records = []
for model_name, model_info in Gynoid_model_summary.items():
    record = {
        'model': model_name,
        'best_fold': model_info['best_fold'],
        'best_r2': model_info['best_r2'],
        'best_rmse': model_info['best_rmse'],
        'mean_r2': model_info['mean_r2'],
        'mean_rmse': model_info['mean_rmse']
    }
    
    # Process parameters (convert to string or further expand)
    if isinstance(model_info['best_params'], dict):
        # Method 1: Convert parameter dictionary to string
        record['best_params'] = str(model_info['best_params'])
        
    else:
        record['best_params'] = model_info['best_params']
    
    records.append(record)

Gynoid_summary_df = pd.DataFrame(records)

Gynoid_summary_df.to_csv("your_path/Gynoid_model_summary.csv", index = True)


############################# BFP #########################################
print("All_BFP: ")
BFP_model_results, BFP_model_best_params = optimized_regression_model(X_train, y_train['BFP'].to_frame(), random_state= 42, use_nested_cv = True)

# Convert nested dictionary to DataFrame
records = []
for model_name, metrics in BFP_model_results.items():
    for i in range(len(metrics['r2'])):
        records.append({
            'model': model_name,
            'fold': i + 1,
            'r2': metrics['r2'][i],
            'rmse': metrics['rmse'][i]
        })

BFP_model_results_df = pd.DataFrame(records)

BFP_model_results_df.to_csv("your_path/BFP_model_results.csv", index = True)

# Save dictionary
with open('your_path/BFP_model_params.json', 'w', encoding='utf-8') as f:
    json.dump(BFP_model_best_params, f, indent=4, ensure_ascii=False)

BFP_model_summary = get_model_summary(BFP_model_results, BFP_model_best_params)

# Convert nested dictionary to flattened DataFrame
records = []
for model_name, model_info in BFP_model_summary.items():
    record = {
        'model': model_name,
        'best_fold': model_info['best_fold'],
        'best_r2': model_info['best_r2'],
        'best_rmse': model_info['best_rmse'],
        'mean_r2': model_info['mean_r2'],
        'mean_rmse': model_info['mean_rmse']
    }
    
    # Process parameters (convert to string or further expand)
    if isinstance(model_info['best_params'], dict):
        # Method 1: Convert parameter dictionary to string
        record['best_params'] = str(model_info['best_params'])
        
    else:
        record['best_params'] = model_info['best_params']
    
    records.append(record)

BFP_summary_df = pd.DataFrame(records)

BFP_summary_df.to_csv("your_path/BFP_model_summary.csv", index = True)

###################### Machine Learning Model Performance ###############

############ Android

df_Andorid = {
    'Outcome': ['R2', 'RMSE'],
    'FSLR': [Android_model_summary['Linear']['mean_r2'], Android_model_summary['Linear']['mean_rmse']],
    'Lasso': [Android_model_summary['Lasso']['mean_r2'], Android_model_summary['Lasso']['mean_rmse']],
    'Ridge': [Android_model_summary['Ridge']['mean_r2'], Android_model_summary['ElasticNet']['mean_rmse']],
    'Elastic': [Android_model_summary['ElasticNet']['mean_r2'],Android_model_summary['Ridge']['mean_rmse']],
    'RF': [Android_model_summary['RandomForest']['mean_r2'], Android_model_summary['RandomForest']['mean_rmse']],
    'GBDT': [Android_model_summary['GBDT']['mean_r2'], Android_model_summary['GBDT']['mean_rmse']],
    'XGBoost': [Android_model_summary['XGBoost']['mean_r2'], Android_model_summary['XGBoost']['mean_rmse']]
}

# Convert dictionary to DataFrame
df_Andorid = pd.DataFrame(df_Andorid)

df_Andorid.to_csv("your_path/Metrics_Android.csv", index = True)

########### Gynoid

df_Gynoid = {
    'Outcome': ['R2', 'RMSE'],
    'FSLR': [Gynoid_model_summary['Linear']['mean_r2'], Gynoid_model_summary['Linear']['mean_rmse']],
    'Lasso': [Gynoid_model_summary['Lasso']['mean_r2'], Gynoid_model_summary['Lasso']['mean_rmse']],
    'Ridge': [Gynoid_model_summary['Ridge']['mean_r2'], Gynoid_model_summary['ElasticNet']['mean_rmse']],
    'Elastic': [Gynoid_model_summary['ElasticNet']['mean_r2'], Gynoid_model_summary['Ridge']['mean_rmse']],
    'RF': [Gynoid_model_summary['RandomForest']['mean_r2'], Gynoid_model_summary['RandomForest']['mean_rmse']],
    'GBDT': [Gynoid_model_summary['GBDT']['mean_r2'], Gynoid_model_summary['GBDT']['mean_rmse']],
    'XGBoost': [Gynoid_model_summary['XGBoost']['mean_r2'], Gynoid_model_summary['XGBoost']['mean_rmse']]
}

# Convert dictionary to DataFrame
df_Gynoid = pd.DataFrame(df_Gynoid)

df_Gynoid.to_csv("your_path/Metrics_Gynoid.csv", index = True)


########### AGFMR

df_A_G = {
    'Outcome': ['R2', 'RMSE'],
    'FSLR': [A_G_model_summary['Linear']['mean_r2'], A_G_model_summary['Linear']['mean_rmse']],
    'Lasso': [A_G_model_summary['Lasso']['mean_r2'], A_G_model_summary['Lasso']['mean_rmse']],
    'Ridge': [A_G_model_summary['Ridge']['mean_r2'], A_G_model_summary['ElasticNet']['mean_rmse']],
    'Elastic': [A_G_model_summary['ElasticNet']['mean_r2'], A_G_model_summary['Ridge']['mean_rmse']],
    'RF': [A_G_model_summary['RandomForest']['mean_r2'], A_G_model_summary['RandomForest']['mean_rmse']],
    'GBDT': [A_G_model_summary['GBDT']['mean_r2'], A_G_model_summary['GBDT']['mean_rmse']],
    'XGBoost': [A_G_model_summary['XGBoost']['mean_r2'], A_G_model_summary['XGBoost']['mean_rmse']]
}

# Convert dictionary to DataFrame
df_A_G = pd.DataFrame(df_A_G)

df_A_G.to_csv("your_path/Metrics_A_G.csv", index = True)


########### FM

df_FM = {
    'Outcome': ['R2', 'RMSE'],
    'FSLR': [FM_model_summary['Linear']['mean_r2'], FM_model_summary['Linear']['mean_rmse']],
    'Lasso': [FM_model_summary['Lasso']['mean_r2'], FM_model_summary['Lasso']['mean_rmse']],
    'Ridge': [FM_model_summary['Ridge']['mean_r2'], FM_model_summary['ElasticNet']['mean_rmse']],
    'Elastic': [FM_model_summary['ElasticNet']['mean_r2'], FM_model_summary['Ridge']['mean_rmse']],
    'RF': [FM_model_summary['RandomForest']['mean_r2'], FM_model_summary['RandomForest']['mean_rmse']],
    'GBDT': [FM_model_summary['GBDT']['mean_r2'], FM_model_summary['GBDT']['mean_rmse']],
    'XGBoost': [FM_model_summary['XGBoost']['mean_r2'], FM_model_summary['XGBoost']['mean_rmse']]
}

# Convert dictionary to DataFrame
df_FM = pd.DataFrame(df_FM)

df_FM.to_csv("your_path/Metrics_FM.csv", index = True)


########### FMI

df_FMI = {
    'Outcome': ['R2', 'RMSE'],
    'FSLR': [FMI_model_summary['Linear']['mean_r2'], FMI_model_summary['Linear']['mean_rmse']],
    'Lasso': [FMI_model_summary['Lasso']['mean_r2'], FMI_model_summary['Lasso']['mean_rmse']],
    'Ridge': [FMI_model_summary['Ridge']['mean_r2'], FMI_model_summary['ElasticNet']['mean_rmse']],
    'Elastic': [FMI_model_summary['ElasticNet']['mean_r2'], FMI_model_summary['Ridge']['mean_rmse']],
    'RF': [FMI_model_summary['RandomForest']['mean_r2'], FMI_model_summary['RandomForest']['mean_rmse']],
    'GBDT': [FMI_model_summary['GBDT']['mean_r2'], FMI_model_summary['GBDT']['mean_rmse']],
    'XGBoost': [FMI_model_summary['XGBoost']['mean_r2'], FMI_model_summary['XGBoost']['mean_rmse']]
}

# Convert dictionary to DataFrame
df_FMI = pd.DataFrame(df_FMI)

df_FMI.to_csv("your_path/Metrics_FMI.csv", index = True)


########### BFP

df_BFP = {
    'Outcome': ['R2', 'RMSE'],
    'FSLR': [BFP_model_summary['Linear']['mean_r2'], BFP_model_summary['Linear']['mean_rmse']],
    'Lasso': [BFP_model_summary['Lasso']['mean_r2'], BFP_model_summary['Lasso']['mean_rmse']],
    'Ridge': [BFP_model_summary['Ridge']['mean_r2'], BFP_model_summary['ElasticNet']['mean_rmse']],
    'Elastic': [BFP_model_summary['ElasticNet']['mean_r2'], BFP_model_summary['Ridge']['mean_rmse']],
    'RF': [BFP_model_summary['RandomForest']['mean_r2'], BFP_model_summary['RandomForest']['mean_rmse']],
    'GBDT': [BFP_model_summary['GBDT']['mean_r2'], BFP_model_summary['GBDT']['mean_rmse']],
    'XGBoost': [BFP_model_summary['XGBoost']['mean_r2'], BFP_model_summary['XGBoost']['mean_rmse']]
}

# Convert dictionary to DataFrame
df_BFP = pd.DataFrame(df_BFP)

df_BFP.to_csv("your_path/Metrics_BFP.csv", index = True)


########### LM

df_LM = {
    'Outcome': ['R2', 'RMSE'],
    'FSLR': [LM_model_summary['Linear']['mean_r2'], LM_model_summary['Linear']['mean_rmse']],
    'Lasso': [LM_model_summary['Lasso']['mean_r2'], LM_model_summary['Lasso']['mean_rmse']],
    'Ridge': [LM_model_summary['Ridge']['mean_r2'], LM_model_summary['ElasticNet']['mean_rmse']],
    'Elastic': [LM_model_summary['ElasticNet']['mean_r2'], LM_model_summary['Ridge']['mean_rmse']],
    'RF': [LM_model_summary['RandomForest']['mean_r2'], LM_model_summary['RandomForest']['mean_rmse']],
    'GBDT': [LM_model_summary['GBDT']['mean_r2'], LM_model_summary['GBDT']['mean_rmse']],
    'XGBoost': [LM_model_summary['XGBoost']['mean_r2'], LM_model_summary['XGBoost']['mean_rmse']]
}

# Convert dictionary to DataFrame
df_LM = pd.DataFrame(df_LM)

df_LM.to_csv("your_path/Metrics_LM.csv", index = True)


########### VAT

df_VAT = {
    'Outcome': ['R2', 'RMSE'],
    'FSLR': [VAT_model_summary['Linear']['mean_r2'], VAT_model_summary['Linear']['mean_rmse']],
    'Lasso': [VAT_model_summary['Lasso']['mean_r2'], VAT_model_summary['Lasso']['mean_rmse']],
    'Ridge': [VAT_model_summary['Ridge']['mean_r2'], VAT_model_summary['ElasticNet']['mean_rmse']],
    'Elastic': [VAT_model_summary['ElasticNet']['mean_r2'], VAT_model_summary['Ridge']['mean_rmse']],
    'RF': [VAT_model_summary['RandomForest']['mean_r2'], VAT_model_summary['RandomForest']['mean_rmse']],
    'GBDT': [VAT_model_summary['GBDT']['mean_r2'], VAT_model_summary['GBDT']['mean_rmse']],
    'XGBoost': [VAT_model_summary['XGBoost']['mean_r2'], VAT_model_summary['XGBoost']['mean_rmse']]
}

# Convert dictionary to DataFrame
df_VAT = pd.DataFrame(df_VAT)

df_VAT.to_csv("your_path/Metrics_VAT.csv", index = True)
