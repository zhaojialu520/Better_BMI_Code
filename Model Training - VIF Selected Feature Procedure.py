import matplotlib
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
from sklearn.model_selection import  cross_val_score   ## Import cross-validation package
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
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from joblib import load
import seaborn as sns
from scipy.stats import t
from matplotlib import font_manager
from matplotlib.colors import TwoSlopeNorm, ListedColormap, LinearSegmentedColormap
from statsmodels.stats.outliers_influence import variance_inflation_factor
from joblib import dump

from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
from statsmodels.stats.stattools import durbin_watson

import warnings
warnings.filterwarnings("ignore")


## Import data
X_train = pd.read_csv("your_path/X_train_All3D_NoScaled.csv",sep=",")
X_train.set_index(X_train.columns[0], inplace= True)  ## Set the first column as the row index
name = X_train.columns.tolist()    ## Extract column names
name = [item.replace('人体外观测量.三维人体扫描分析系统.', '') for item in name]    ## Remove specific prefix from column names
name = [item.replace('.cm.', '') for item in name]    ## Remove ".cm." from column names
X_train.columns = name   ## Replace column names with simplified names
# print(X_train)
X_train = X_train.drop(columns = ['height', 'Waist_To_Hip_Ratio.x.100.'])
print(X_train)

X_test = pd.read_csv("your_path/X_test_All3D_NoScaled.csv",sep=",")
X_test.set_index(X_test.columns[0], inplace= True)  ## Set the first column as the row index
X_test.columns = name   ## Replace column names with simplified names
X_test = X_test.drop(columns = ['height', 'Waist_To_Hip_Ratio.x.100.'])
# print(X_test)

y_train = pd.read_csv("your_path/y_train_All3D_NoScaled.csv",sep=",")
y_train.set_index(y_train.columns[0], inplace= True)  ## Set the first column as the row index

y_test = pd.read_csv("your_path/y_test_All3D_NoScaled.csv",sep=",")
y_test.set_index(y_test.columns[0], inplace= True)  ## Set the first column as the row index

## Load previously trained models and parameters using training set
AA_20240105_All_models = load('your_path/AA_20240105_All_models.joblib')

def feature_selection_process4(data, target_col, significance_level=0.05, vif_threshold=10, r2_tolerance=0.02):
    """
    Optimized feature selection process: Maximize model R² accuracy while screening for VIF and significance
    Prioritize removing features with minimal impact on model performance
    
    Parameters:
    data: DataFrame containing features and target variable
    target_col: Target variable column name
    significance_level: Significance level, default 0.05
    vif_threshold: VIF threshold, default 10
    r2_tolerance: R² drop tolerance, default 0.02 (2%)
    
    Returns:
    Final selected feature list and final regression model
    """
    
    # 1. Filter out features containing specific keywords
    exclude_keywords = ['3D', '腰带', '垂直', '内部', '躯干扭', '角度']
    initial_features = [col for col in data.columns 
                       if col != target_col and 
                       not any(keyword in col for keyword in exclude_keywords)]
    
    print(f"Initial feature count: {len(initial_features)}")
    print(f"Excluded features: {[col for col in data.columns if col != target_col and col not in initial_features]}")
    print(f"R² tolerance: {r2_tolerance}")
    
    current_features = initial_features.copy()
    iteration = 1
    best_r2 = 0
    best_features = current_features.copy()
    
    # Calculate initial model R² as baseline
    X_initial = data[initial_features]
    y_initial = data[target_col]
    X_initial_with_const = add_constant(X_initial)
    initial_model = sm.OLS(y_initial, X_initial_with_const).fit()
    baseline_r2 = initial_model.rsquared
    best_r2 = baseline_r2
    print(f"Baseline model R²: {baseline_r2:.4f}")
    
    while True:
        print(f"\n=== Iteration {iteration} ===")
        
        # Prepare data
        X = data[current_features]
        y = data[target_col]
        X_with_const = add_constant(X)
        
        # Fit regression model
        try:
            model = sm.OLS(y, X_with_const).fit()
            current_r2 = model.rsquared
            current_adj_r2 = model.rsquared_adj
        except Exception as e:
            print(f"Regression model fitting failed: {e}")
            break
            
        # 2. Check independent variable significance
        pvalues = model.pvalues.drop('const')
        insignificant_features = pvalues[pvalues > significance_level].index.tolist()
        
        # 3. Calculate VIF values
        vif_data = pd.DataFrame()
        vif_data["feature"] = X_with_const.columns
        vif_data["VIF"] = [variance_inflation_factor(X_with_const.values, i) 
                          for i in range(X_with_const.shape[1])]
        
        vif_data = vif_data[vif_data['feature'] != 'const']
        high_vif_features = vif_data[vif_data["VIF"] > vif_threshold]
        
        print(f"Current feature count: {len(current_features)}")
        print(f"Current R²: {current_r2:.4f}, Adjusted R²: {current_adj_r2:.4f}")
        print(f"Insignificant features (p > {significance_level}): {insignificant_features}")
        print(f"Features with VIF > {vif_threshold}:")
        print(high_vif_features)
        
        # Check termination conditions
        if len(insignificant_features) == 0 and len(high_vif_features) == 0:
            print("\n✅ Termination condition reached: All features are significant and VIF < 10")
            break
            
        # Record current best model (if R² drop is within tolerance and other conditions are met)
        r2_drop = baseline_r2 - current_r2
        if (len(insignificant_features) == 0 and 
            r2_drop <= r2_tolerance and 
            current_r2 > best_r2 - 0.01):  # Allow slight drop
            best_r2 = current_r2
            best_features = current_features.copy()
            print(f"Updated best model: R² = {current_r2:.4f}, Feature count = {len(current_features)}")
        
        # Generate all candidate features to remove (including insignificant and high VIF features)
        candidate_features_to_remove = []
        
        # Collect all candidate features to remove (no priority distinction)
        if len(insignificant_features) > 0:
            for feature in insignificant_features:
                candidate_features_to_remove.append({
                    'feature': feature,
                    'reason': 'insignificant',
                    'p_value': pvalues[feature],
                    'vif': vif_data[vif_data['feature'] == feature]['VIF'].values[0] if feature in vif_data['feature'].values else 0
                })
        
        if len(high_vif_features) > 0:
            for _, row in high_vif_features.iterrows():
                # Avoid duplicate addition (if a feature has both high VIF and is insignificant)
                if not any(candidate['feature'] == row['feature'] for candidate in candidate_features_to_remove):
                    candidate_features_to_remove.append({
                        'feature': row['feature'],
                        'reason': 'high_vif',
                        'p_value': pvalues[row['feature']] if row['feature'] in pvalues else 1.0,
                        'vif': row['VIF']
                    })
        
        # Intelligently select features to remove - based on impact on R²
        features_to_remove = []
        
        if candidate_features_to_remove:
            # Evaluate impact on R² when removing each candidate feature
            removal_impact = []
            for candidate in candidate_features_to_remove:
                feature = candidate['feature']
                # Test R² after removing this feature
                test_features = [f for f in current_features if f != feature]
                if test_features:  # Ensure there are remaining features
                    X_test = data[test_features]
                    X_test_with_const = add_constant(X_test)
                    try:
                        test_model = sm.OLS(y, X_test_with_const).fit()
                        r2_after_removal = test_model.rsquared
                        r2_drop = current_r2 - r2_after_removal
                    except:
                        r2_drop = float('inf')  # If fitting fails, consider impact large
                else:
                    r2_drop = float('inf')
                
                removal_impact.append({
                    'feature': feature,
                    'reason': candidate['reason'],
                    'p_value': candidate['p_value'],
                    'vif': candidate['vif'],
                    'r2_drop': r2_drop
                })
            
            # Sort by R² impact from smallest to largest
            removal_impact_sorted = sorted(removal_impact, key=lambda x: x['r2_drop'])
            
            # Select feature with minimal impact on R² for removal
            best_candidate = None
            
            # First look for feature with impact within tolerance range
            for candidate in removal_impact_sorted:
                if candidate['r2_drop'] <= r2_tolerance:
                    best_candidate = candidate
                    break
            
            # If no feature within tolerance range, select one with relatively smallest impact
            if best_candidate is None and removal_impact_sorted:
                best_candidate = removal_impact_sorted[0]
                print(f"⚠️ All removals would cause R² drop exceeding tolerance, selecting one with minimal impact")
            
            if best_candidate:
                features_to_remove.append(best_candidate['feature'])
                print(f"Remove feature: {best_candidate['feature']} ({best_candidate['reason']})")
                print(f"  p-value: {best_candidate['p_value']:.4f}, VIF: {best_candidate['vif']:.2f}, R² drop: {best_candidate['r2_drop']:.4f}")
                
                # Show R² impact of other candidate features (for debugging)
                if len(removal_impact_sorted) > 1:
                    print(f"  R² impact of other candidate features:")
                    for i, candidate in enumerate(removal_impact_sorted[1:4]):  # Show first few
                        print(f"    {candidate['feature']}: R² drop {candidate['r2_drop']:.4f} ({candidate['reason']})")
        
        # If no suitable candidate features (all removals would cause significant R² drop)
        if not features_to_remove:
            print("⚠️ No suitable features to remove (all removals would cause significant R² drop)")
            # Check if relaxed conditions are met
            if len(insignificant_features) == 0:
                print("✅ All features are significant, accept current model")
                break
            else:
                # Force remove feature with largest p-value
                if insignificant_features:
                    max_p_feature = max(insignificant_features, key=lambda x: pvalues[x])
                    features_to_remove.append(max_p_feature)
                    print(f"Force remove most insignificant feature: {max_p_feature}")
        
        # Remove features
        for feature in features_to_remove:
            if feature in current_features:
                current_features.remove(feature)
                print(f"Removed feature: {feature}")
            
        # Check if there are still features remaining
        if len(current_features) == 0:
            print("❌ All features have been removed!")
            current_features = best_features  # Restore to best feature set
            break
            
        iteration += 1
        
        # Prevent infinite loop
        if iteration > 30:
            print("⚠️ Maximum iteration count reached, forced termination")
            break
    
    # Final model evaluation
    print(f"\n🎯 Final selected features ({len(current_features)}):")
    X_final = data[current_features]
    y_final = data[target_col]
    X_final_with_const = add_constant(X_final)
    final_model = sm.OLS(y_final, X_final_with_const).fit()
    
    # Calculate final metrics
    final_vif_data = pd.DataFrame()
    final_vif_data["feature"] = X_final_with_const.columns
    final_vif_data["VIF"] = [variance_inflation_factor(X_final_with_const.values, i) 
                           for i in range(X_final_with_const.shape[1])]
    final_vif_data = final_vif_data[final_vif_data['feature'] != 'const']
    
    final_r2 = final_model.rsquared
    final_adj_r2 = final_model.rsquared_adj
    r2_reduction = baseline_r2 - final_r2
    
    print(f"\n📊 Model performance summary:")
    print(f"Initial R²: {baseline_r2:.4f}")
    print(f"Final R²: {final_r2:.4f}")
    print(f"R² reduction: {r2_reduction:.4f} ({r2_reduction/baseline_r2*100:.1f}%)")
    print(f"Adjusted R²: {final_adj_r2:.4f}")
    
    for feature in current_features:
        p_value = final_model.pvalues[feature]
        vif_value = final_vif_data[final_vif_data['feature'] == feature]['VIF'].values[0]
        significance_indicator = " ✓" if p_value <= significance_level else " ✗"
        vif_indicator = " ✓" if vif_value <= vif_threshold else " ✗"
        print(f"  {feature}: p-value={p_value:.4f}{significance_indicator}, VIF={vif_value:.2f}{vif_indicator}")
    
    return current_features, final_model

# Calculate VIF
def calculate_vif(data, include_const=True, threshold=10):
    """
    Calculate VIF value for each variable in the dataset
    
    Parameters:
    data: pandas DataFrame, input data
    include_const: bool, whether to include constant term in results
    threshold: float, VIF threshold for marking high multicollinearity
    
    Returns:
    pandas DataFrame, containing variable names and corresponding VIF values
    """
    # Input validation
    if not isinstance(data, pd.DataFrame):
        raise ValueError("Input data must be pandas DataFrame")
    
    if data.empty:
        raise ValueError("Input data cannot be empty")
    
    # Check for missing values
    if data.isnull().any().any():
        raise ValueError("Data contains missing values, please handle missing values first")
    
    # Add constant term
    X = add_constant(data)
    
    # Calculate VIF for each variable
    vif_data = pd.DataFrame()
    vif_data['Variable'] = X.columns
    vif_data['VIF'] = [variance_inflation_factor(X.values, i) 
                      for i in range(X.shape[1])]
    
    # Add diagnostic information
    vif_data['High_VIF'] = vif_data['VIF'] > threshold
    
    # Whether to exclude constant term
    if not include_const:
        vif_data = vif_data[vif_data['Variable'] != 'const']
    
    # Sort by VIF value
    vif_data = vif_data.sort_values('VIF', ascending=False)
    
    return vif_data.reset_index(drop=True)

##  --------------------------------  VAT  --------------------------

VAT_feature = AA_20240105_All_models['VAT_forward_linear_model']['Selected_feature'] ## Independent variables from stepwise regression model
X_VAT_train = X_train[VAT_feature]  ## Independent variable values
Y_VAT_train = y_train["VATmass"]
Y_VAT_train = Y_VAT_train.to_frame().rename(columns={'VATmass': 'VATmass'})

# Merge X and Y
VAT_train_data = pd.concat([X_VAT_train, Y_VAT_train], axis=1)

## For reproducibility, perform VIF and significance screening
# Keep ["标准站姿腰围", "大腿围度左边.水平线.", "对高臀部的腰部后面",   "臂长右", "上臂围度左边", "age"]

VAT_targets, VAT_forward_linear_model = feature_selection_process4(data = VAT_train_data, target_col= "VATmass", significance_level=0.05, vif_threshold=10, r2_tolerance=0.01)

X_VAT_train_Update = X_train[VAT_targets]
vif_VAT_result_Update = calculate_vif(X_VAT_train_Update, include_const= False)  ## Calculate VIF of independent variables
print(vif_VAT_result_Update)

vif_VAT_result_Update.to_csv("your_path/VIF_VAT_params.csv", index=False)

# Calculate DW statistic
dw_VAT = durbin_watson(VAT_forward_linear_model.resid)
print(f"VAT-Durbin-Watson statistic: {dw_VAT:.4f}") # VAT-Durbin-Watson statistic: 1.8993

##  --------------------------------  FM  --------------------------

FM_feature = AA_20240105_All_models['FM_forward_linear_model']['Selected_feature']
X_FM_train = X_train[FM_feature]  ## Independent variable values
Y_FM_train = y_train["FM"]
Y_FM_train = Y_FM_train.to_frame().rename(columns={'FM': 'FM'})

# Merge X and Y
FM_train_data = pd.concat([X_FM_train, Y_FM_train], axis=1)

# For reproducibility, combined with VIF < 10 and regression significance
# Keep [ "膝围度左边", "gender", "脚踝围左边", "前臂围度右边","标准站姿腰围", "age", "胯部长度", "头高", ,'腿肚围度左边', "最小腿围度左边"]

FM_targets, FM_forward_linear_model = feature_selection_process4(data = FM_train_data, target_col= "FM", significance_level=0.05, vif_threshold=10, r2_tolerance=0.01)

X_FM_train_Update = X_train[FM_targets]
vif_FM_result_Update = calculate_vif(X_FM_train_Update, include_const= False)  ## Calculate VIF of independent variables
vif_FM_result_Update.to_csv("your_path/VIF_FM_params.csv", index=False)
print(vif_FM_result_Update)

dw_FM = durbin_watson(FM_forward_linear_model.resid)
print(f"FM-Durbin-Watson statistic: {dw_FM:.4f}") # FM-Durbin-Watson statistic: 1.9210

## -------------------------------- FMI ----------------------------------

FMI_feature = AA_20240105_All_models['FMI_forward_linear_model']['Selected_feature']
X_FMI_train = X_train[FMI_feature]  ## Independent variable values
Y_FMI_train = y_train["FMI"]
Y_FMI_train = Y_FMI_train.to_frame().rename(columns={'FMI': 'FMI'})

# Merge X and Y
FMI_train_data = pd.concat([X_FMI_train, Y_FMI_train], axis=1)

# For reproducibility, combined with VIF < 10 and regression significance
# Keep ["中臀围", "身体高度", "膝围度左边", "gender",  "age",  "脚踝围右边", "臂长左", "上臂围度右边"]

FMI_targets, FMI_forward_linear_model = feature_selection_process4(data = FMI_train_data, target_col= "FMI", significance_level=0.05, vif_threshold=10, r2_tolerance=0.01)

X_FMI_train_Update = X_train[FMI_targets]
vif_FMI_result_Update = calculate_vif(X_FMI_train_Update, include_const= False)  ## Calculate VIF of independent variables
vif_FMI_result_Update.to_csv("your_path/VIF_FMI_params.csv", index=False)
print(vif_FMI_result_Update)

dw_FMI = durbin_watson(FMI_forward_linear_model.resid)
print(f"FMI-Durbin-Watson statistic: {dw_FMI:.4f}") # FMI-Durbin-Watson statistic: 1.9920

##  --------------------------------  LM  --------------------------

LM_feature = AA_20240105_All_models['LM_forward_linear_model']['Selected_feature']
X_LM_train = X_train[LM_feature]  ## Independent variable values
Y_LM_train = y_train["LM"]
Y_LM_train = Y_LM_train.to_frame().rename(columns={'LM': 'LM'})

# Merge X and Y
LM_train_data = pd.concat([X_LM_train, Y_LM_train], axis=1)

# For reproducibility, combined with VIF < 10 and regression significance
# Keep ["gender", "高臀部围度", "膝围度左边", "脚踝围左边",  "前臂围度右边",  "身体高度", "腿肚围度左边", "宽度腋窝", "最小腿围度左边"]

LM_targets, LM_forward_linear_model = feature_selection_process4(data = LM_train_data, target_col= "LM", significance_level=0.05, vif_threshold=10, r2_tolerance=0.01)

X_LM_train_Update = X_train[LM_targets]
vif_LM_result_Update = calculate_vif(X_LM_train_Update, include_const= False)  ## Calculate VIF of independent variables
vif_LM_result_Update.to_csv("your_path/VIF_LM_params.csv", index=False)
print(vif_LM_result_Update)

dw_LM = durbin_watson(LM_forward_linear_model.resid)
print(f"LM-Durbin-Watson statistic: {dw_LM:.4f}") # LM-Durbin-Watson statistic: 2.0129


##  --------------------------------  Android  --------------------------

Android_feature = AA_20240105_All_models['Android_forward_linear_model']['Selected_feature']
X_Android_train = X_train[Android_feature]  ## Independent variable values
Y_Android_train = y_train["Android"]
Y_Android_train = Y_Android_train.to_frame().rename(columns={'Android': 'Android'})

# Merge X and Y
Android_train_data = pd.concat([X_Android_train, Y_Android_train], axis=1)

# For reproducibility, combined with VIF < 10 and regression significance
# Keep ["标准站姿腰围",  "gender", "前臂围度右边", "脚踝围左边",   "臀部.大腿围度",  "臂长左",  "膝围度左边", "腰部对臀部高度左边"]

Android_targets, Android_forward_linear_model = feature_selection_process4(data = Android_train_data, target_col= "Android", significance_level=0.05, vif_threshold=10, r2_tolerance=0.01)

X_Android_train_Update = X_train[Android_targets]
vif_Android_result_Update = calculate_vif(X_Android_train_Update, include_const= False)  ## Calculate VIF of independent variables
vif_Android_result_Update.to_csv("your_path/VIF_Android_params.csv", index=False)
print(vif_Android_result_Update)

dw_Android = durbin_watson(Android_forward_linear_model.resid)
print(f"Android-Durbin-Watson statistic: {dw_Android:.4f}")   ## Android-Durbin-Watson statistic: 1.9673

##  --------------------------------  Gynoid  --------------------------

Gynoid_feature = AA_20240105_All_models['Gynoid_forward_linear_model']['Selected_feature']
X_Gynoid_train = X_train[Gynoid_feature]  ## Independent variable values
Y_Gynoid_train = y_train["Gynoid"]
Y_Gynoid_train = Y_Gynoid_train.to_frame().rename(columns={'Gynoid': 'Gynoid'})

# Merge X and Y
Gynoid_train_data = pd.concat([X_Gynoid_train, Y_Gynoid_train], axis=1)

# For reproducibility, combined with VIF < 10 and regression significance
## Keep [ "gender", "膝围度左边","高臀部围度", "age", "前臂围度右边", "大腿围度右边.水平线.", "腿肚围度右边", "宽度腋窝", "横过后面宽度.腋窝水平.", "胯长度.后面", "上臂长度左边"]

Gynoid_targets, Gynoid_forward_linear_model = feature_selection_process4(data = Gynoid_train_data, target_col= "Gynoid", significance_level=0.05, vif_threshold=10, r2_tolerance=0.01)

X_Gynoid_train_Update = X_train[Gynoid_targets]
vif_Gynoid_result_Update = calculate_vif(X_Gynoid_train_Update, include_const= False)  ## Calculate VIF of independent variables
vif_Gynoid_result_Update.to_csv("your_path/VIF_Gynoid_params.csv", index=False)
print(vif_Gynoid_result_Update)

dw_Gynoid = durbin_watson(Gynoid_forward_linear_model.resid)
print(f"Gynoid-Durbin-Watson statistic: {dw_Gynoid:.4f}") # Gynoid-Durbin-Watson statistic: 2.0369


##  --------------------------------  Android / Gynoid  --------------------------

A_G_feature = AA_20240105_All_models['A_G_forward_linear_model']['Selected_feature']
X_A_G_train = X_train[A_G_feature]  ## Independent variable values
Y_A_G_train = y_train["A_G"]
Y_A_G_train = Y_A_G_train.to_frame().rename(columns={'A_G': 'A_G'})

# Merge X and Y
A_G_train_data = pd.concat([X_A_G_train, Y_A_G_train], axis=1)

# For reproducibility, combined with VIF < 10 and regression significance
## Delete ['高腰围','标准站姿腰围','Underbust圆周.水平线.','腰带.膝距离','腰带.腰的偏移量.前面.','7CV.垂直的距离','脖子前面到垂直的距离','胸部.胸围度.水平线.','脖子到横过后面宽度.腋窝水平.','上面的躯干扭...']
## Keep ["臀部围度", "对高臀部的腰部后面", "大腿围度左边.水平线.", "脚踝围左边", "中间的脖子围度", "标准站姿腰围", "胸部.胸围度", "脖子到横过后面宽度.腋窝水平", "前臂长度左边"]

A_G_targets, A_G_forward_linear_model = feature_selection_process4(data = A_G_train_data, target_col= "A_G", significance_level=0.05, vif_threshold=10, r2_tolerance=0.01)

X_A_G_train_Update = X_train[A_G_targets]
vif_A_G_result_Update = calculate_vif(X_A_G_train_Update, include_const= False)  ## Calculate VIF of independent variables
vif_A_G_result_Update.to_csv("your_path/VIF_A_G__params.csv", index=False)
print(vif_A_G_result_Update)

dw_A_G = durbin_watson(A_G_forward_linear_model.resid)
print(f"A_G-Durbin-Watson statistic: {dw_A_G:.4f}") # A_G-Durbin-Watson statistic: 1.8479

##  --------------------------------  BFP  --------------------------

BFP_feature = AA_20240105_All_models['BFP_forward_linear_model']['Selected_feature']
X_BFP_train = X_train[BFP_feature]  ## Independent variable values
Y_BFP_train = y_train["BFP"] * 100
Y_BFP_train = Y_BFP_train.to_frame().rename(columns={'BFP': 'BFP'})

# Merge X and Y
BFP_train_data = pd.concat([X_BFP_train, Y_BFP_train], axis=1)

# For reproducibility, combined with VIF < 10 and regression significance
# Delete ['中臀围','腹圆周','脖子对腰部中心向后地','7CV.垂直的距离','脖子前面到垂直的距离','age','中间的脖子围度','腰带.腰的偏移量.前面.','后背到垂直面距离.在胸部水平面','高腰围','脖子正确地对腰部后面','对高臀部的腰部后面','臀部围度','weight','胸部到垂直的距离','肩角度右边','横肩覆盖脖子']
# Keep["gender", "臂长左", "前臂围度右边", "膝围度左边", "脚踝围左边", "臀部.大腿围度",  "头高",  "腿肚围度左边",  "最小腿围度左边", "大腿围度右边.水平线.","宽度腋窝", "脖子左边对腰部后面", "横过后面宽度.腋窝水平.", "标准站姿腰围",  "颈到臀距离","胯长度.后面"]

BFP_targets, BFP_forward_linear_model = feature_selection_process4(data = BFP_train_data, target_col= "BFP", significance_level=0.05, vif_threshold=10, r2_tolerance=0.01)

X_BFP_train_Update = X_train[BFP_targets]
vif_BFP_result_Update = calculate_vif(X_BFP_train_Update, include_const= False)  ## Calculate VIF of independent variables
vif_BFP_result_Update.to_csv("your_path/VIF_BFP__params.csv", index=False)
print(vif_BFP_result_Update)

dw_BFP = durbin_watson(BFP_forward_linear_model.resid)
print(f"BFP-Durbin-Watson statistic: {dw_BFP:.4f}") # BFP-Durbin-Watson statistic: 1.9218

#########  Get model parameters and test set model performance ################

def model_result(X_final, Y_final, final_model):
    params = final_model.params  ## Get model parameters
    p_values = final_model.pvalues  ## Get P-values
    r_squared = final_model.rsquared  ## Get R2
    y_pred = final_model.predict(X_final)  ## Get Y predicted values
    rmse = np.sqrt(mean_squared_error(Y_final, y_pred))  ## Calculate RMSE
    print("r2:",r_squared, "rmse:",rmse)

    results = pd.DataFrame({
        'Parameter': params.index,
        'Coefficient': params.values,
        'P-value': p_values.values
    })

    r2_row = pd.DataFrame({
        'Parameter': ['R-Squared'],
        'Coefficient': [r_squared],
        'P-Value': [None]
    })  ## Add R2

    rmse_row = pd.DataFrame({
        'Parameter': ['RMSE'],
        'Coefficient': [rmse],
        'P-Value': [None]
    })  ## Add RMSE

    results = pd.concat([results, r2_row, rmse_row], ignore_index=True)

    results['Coefficient'] = results['Coefficient'].round(3)
    
    return results

X_VAT_test = X_test[VAT_targets]
X_VAT_test = add_constant(X_VAT_test)
Y_VAT_test = y_test['VATmass']
print("VATmass") # r2: 0.8058659068946479 rmse: 0.23225482234553405
VIF_VAT_params = model_result(X_VAT_test, Y_VAT_test, VAT_forward_linear_model)
VIF_VAT_params.to_csv("your_path/VIF_VAT_params.csv", index=False)

X_FM_test = X_test[FM_targets]
X_FM_test = add_constant(X_FM_test)
Y_FM_test = y_test['FM']
print("FM")  # r2: 0.8766785992784824 rmse: 2.1806469681696576
VIF_FM_params = model_result(X_FM_test, Y_FM_test, FM_forward_linear_model)
VIF_FM_params.to_csv("your_path/VIF_FM_params.csv", index=False)

X_FMI_test = X_test[FMI_targets]
X_FMI_test = add_constant(X_FMI_test)
Y_FMI_test = y_test['FMI']
print("FMI") # r2: 0.8676449370555529 rmse: 0.798729658671095
VIF_FMI_params = model_result(X_FMI_test, Y_FMI_test, FMI_forward_linear_model)
VIF_FMI_params.to_csv("your_path/VIF_FMI_params.csv", index=False)

X_LM_test = X_test[LM_targets]
X_LM_test = add_constant(X_LM_test)
Y_LM_test = y_test['LM']
print("LM")  # 0.9210978254059097 rmse: 2.488947923219758
VIF_LM_params = model_result(X_LM_test, Y_LM_test, LM_forward_linear_model)
VIF_LM_params.to_csv("your_path/VIF_LM_params.csv", index=False)

X_Android_test = X_test[Android_targets]
X_Android_test = add_constant(X_Android_test)
Y_Android_test = y_test['Android']
print("Android")  # r2: 0.8852250182118173 rmse: 0.2596830447532017
VIF_Android_params = model_result(X_Android_test, Y_Android_test, Android_forward_linear_model)
VIF_Android_params.to_csv("your_path/VIF_Android_params.csv", index=False)

X_Gynoid_test = X_test[Gynoid_targets]
X_Gynoid_test = add_constant(X_Gynoid_test)
Y_Gynoid_test = y_test['Gynoid']
print("Gynoid")  # 0.8660905559562025 rmse: 0.37364140066743995
VIF_Gynoid_params = model_result(X_Gynoid_test, Y_Gynoid_test, Gynoid_forward_linear_model)
VIF_Gynoid_params.to_csv("your_path/VIF_Gynoid_params.csv", index=False)

X_A_G_test = X_test[A_G_targets]
X_A_G_test = add_constant(X_A_G_test)
Y_A_G_test = y_test['A_G']
print("AGFMR")  # r2: 0.8438780901807932 rmse: 0.07848113372038507
VIF_A_G_params = model_result(X_A_G_test, Y_A_G_test, A_G_forward_linear_model)
VIF_A_G_params.to_csv("your_path/VIF_A_G_params.csv", index=False)

X_BFP_test = X_test[BFP_targets]
X_BFP_test = add_constant(X_BFP_test)
Y_BFP_test = y_test['BFP'] * 100
print("BFP")   # r2: 0.7872491935850359 rmse: 3.374100525171791
VIF_BFP_params = model_result(X_BFP_test, Y_BFP_test, BFP_forward_linear_model)
VIF_BFP_params.to_csv("your_path/VIF_BFP_params.csv", index=False)


## Save multiple models and parameters into a dictionary
AA_20240105_All_VIF_models = {
    'FMI_forward_linear_model':{ 'Forward_linear_model': FMI_forward_linear_model, 'Selected_feature': FMI_targets},
    'A_G_forward_linear_model':{ 'Forward_linear_model': A_G_forward_linear_model, 'Selected_feature': A_G_targets},
    'FM_forward_linear_model':{ 'Forward_linear_model' : FM_forward_linear_model, 'Selected_feature': FM_targets},
    'LM_forward_linear_model':{ 'Forward_linear_model' : LM_forward_linear_model, 'Selected_feature': LM_targets},
    'VAT_forward_linear_model':{ 'Forward_linear_model' : VAT_forward_linear_model, 'Selected_feature': VAT_targets},
    'Android_forward_linear_model':{ 'Forward_linear_model' : Android_forward_linear_model, 'Selected_feature': Android_targets},
    'Gynoid_forward_linear_model':{ 'Forward_linear_model' : Gynoid_forward_linear_model, 'Selected_feature': Gynoid_targets},
    'BFP_forward_linear_model': { 'Forward_linear_model' : BFP_forward_linear_model, 'Selected_feature': BFP_targets}
}

## Save dictionary

dump(AA_20240105_All_VIF_models, 'your_path/AA_20240105_All_VIF_models.joblib')


####################### Compare model performance metrics R2, RMSE, ME(95%CI) on training and test sets #########################33

## Load previously trained models and parameters using training set
AA_20240105_All_VIF_models = load('your_path/AA_20240105_All_VIF_models.joblib')


## Define function to calculate model performance
def calculate_metrics(y_true, y_pred):
    """
    Calculate and return R², RMSE, ME and ME's 95% CI
    """
    n = len(y_true)
    
    # R-squared
    r2 = r2_score(y_true, y_pred)
    
    # RMSE
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    
    # Mean Error (ME)
    errors =  y_pred - y_true
    me = np.mean(errors)
    
    # 95% Confidence Interval for ME
    se_me = stats.sem(errors) # stats.sem directly calculates standard error
    ci_low, ci_high = stats.t.interval(0.95, df=n-1, loc=me, scale=se_me)
    
    return r2, rmse, me, (ci_low, ci_high)

################################   FMI  ######################################
FMI_FSLR_VIF_model = AA_20240105_All_VIF_models['FMI_forward_linear_model']['Forward_linear_model']
FMI_Selected_feature = AA_20240105_All_VIF_models['FMI_forward_linear_model']['Selected_feature']
FMI_train_pred = FMI_FSLR_VIF_model.predict(add_constant(X_train[FMI_Selected_feature]))
FMI_test_pred = FMI_forward_linear_model.predict(add_constant(X_test[FMI_Selected_feature]))


#  Calculate metrics for training and test sets separately
FMI_train_r2, FMI_train_rmse, FMI_train_me, FMI_train_ci = calculate_metrics(y_train['FMI'], FMI_train_pred)
FMI_test_r2, FMI_test_rmse, FMI_test_me, FMI_test_ci = calculate_metrics(y_test['FMI'], FMI_test_pred)

################################   A_G  ######################################
A_G_FSLR_VIF_model = AA_20240105_All_VIF_models['A_G_forward_linear_model']['Forward_linear_model']
A_G_Selected_feature = AA_20240105_All_VIF_models['A_G_forward_linear_model']['Selected_feature']
A_G_train_pred = A_G_FSLR_VIF_model.predict(add_constant(X_train[A_G_Selected_feature]))
A_G_test_pred = A_G_forward_linear_model.predict(add_constant(X_test[A_G_Selected_feature]))


#  Calculate metrics for training and test sets separately
A_G_train_r2, A_G_train_rmse, A_G_train_me, A_G_train_ci = calculate_metrics(y_train['A_G'], A_G_train_pred)
A_G_test_r2, A_G_test_rmse, A_G_test_me, A_G_test_ci = calculate_metrics(y_test['A_G'], A_G_test_pred)

################################   FM  ######################################
FM_FSLR_VIF_model = AA_20240105_All_VIF_models['FM_forward_linear_model']['Forward_linear_model']
FM_Selected_feature = AA_20240105_All_VIF_models['FM_forward_linear_model']['Selected_feature']
FM_train_pred = FM_FSLR_VIF_model.predict(add_constant(X_train[FM_Selected_feature]))
FM_test_pred = FM_forward_linear_model.predict(add_constant(X_test[FM_Selected_feature]))

#  Calculate metrics for training and test sets separately
FM_train_r2, FM_train_rmse, FM_train_me, FM_train_ci = calculate_metrics(y_train['FM'], FM_train_pred)
FM_test_r2, FM_test_rmse, FM_test_me, FM_test_ci = calculate_metrics(y_test['FM'], FM_test_pred)

################################   LM  ######################################
LM_FSLR_VIF_model = AA_20240105_All_VIF_models['LM_forward_linear_model']['Forward_linear_model']
LM_Selected_feature = AA_20240105_All_VIF_models['LM_forward_linear_model']['Selected_feature']
LM_train_pred = LM_FSLR_VIF_model.predict(add_constant(X_train[LM_Selected_feature]))
LM_test_pred = LM_forward_linear_model.predict(add_constant(X_test[LM_Selected_feature]))

#  Calculate metrics for training and test sets separately
LM_train_r2, LM_train_rmse, LM_train_me, LM_train_ci = calculate_metrics(y_train['LM'], LM_train_pred)
LM_test_r2, LM_test_rmse, LM_test_me, LM_test_ci = calculate_metrics(y_test['LM'], LM_test_pred)

################################  VAT  ######################################
VAT_FSLR_VIF_model = AA_20240105_All_VIF_models['VAT_forward_linear_model']['Forward_linear_model']
VAT_Selected_feature = AA_20240105_All_VIF_models['VAT_forward_linear_model']['Selected_feature']
VAT_train_pred = VAT_FSLR_VIF_model.predict(add_constant(X_train[VAT_Selected_feature]))
VAT_test_pred = VAT_forward_linear_model.predict(add_constant(X_test[VAT_Selected_feature]))

#  Calculate metrics for training and test sets separately
VAT_train_r2, VAT_train_rmse, VAT_train_me, VAT_train_ci = calculate_metrics(y_train['VATmass'], VAT_train_pred)
VAT_test_r2, VAT_test_rmse, VAT_test_me, VAT_test_ci = calculate_metrics(y_test['VATmass'], VAT_test_pred)

################################  Android  ######################################
Android_FSLR_VIF_model = AA_20240105_All_VIF_models['Android_forward_linear_model']['Forward_linear_model']
Android_Selected_feature = AA_20240105_All_VIF_models['Android_forward_linear_model']['Selected_feature']
Android_train_pred = Android_FSLR_VIF_model.predict(add_constant(X_train[Android_Selected_feature]))
Android_test_pred = Android_forward_linear_model.predict(add_constant(X_test[Android_Selected_feature]))

#  Calculate metrics for training and test sets separately
Android_train_r2, Android_train_rmse, Android_train_me, Android_train_ci = calculate_metrics(y_train['Android'], Android_train_pred)
Android_test_r2, Android_test_rmse, Android_test_me, Android_test_ci = calculate_metrics(y_test['Android'], Android_test_pred)


################################  Gynoid  ######################################
Gynoid_FSLR_VIF_model = AA_20240105_All_VIF_models['Gynoid_forward_linear_model']['Forward_linear_model']
Gynoid_Selected_feature = AA_20240105_All_VIF_models['Gynoid_forward_linear_model']['Selected_feature']
Gynoid_train_pred = Gynoid_FSLR_VIF_model.predict(add_constant(X_train[Gynoid_Selected_feature]))
Gynoid_test_pred = Gynoid_forward_linear_model.predict(add_constant(X_test[Gynoid_Selected_feature]))

#  Calculate metrics for training and test sets separately
Gynoid_train_r2, Gynoid_train_rmse, Gynoid_train_me, Gynoid_train_ci = calculate_metrics(y_train['Gynoid'], Gynoid_train_pred)
Gynoid_test_r2, Gynoid_test_rmse, Gynoid_test_me, Gynoid_test_ci = calculate_metrics(y_test['Gynoid'], Gynoid_test_pred)

################################  BFP ######################################
BFP_FSLR_VIF_model = AA_20240105_All_VIF_models['BFP_forward_linear_model']['Forward_linear_model']
BFP_Selected_feature = AA_20240105_All_VIF_models['BFP_forward_linear_model']['Selected_feature']
BFP_train_pred = BFP_FSLR_VIF_model.predict(add_constant(X_train[BFP_Selected_feature]))
BFP_test_pred = BFP_forward_linear_model.predict(add_constant(X_test[BFP_Selected_feature]))

#  Calculate metrics for training and test sets separately
BFP_train_r2, BFP_train_rmse, BFP_train_me, BFP_train_ci = calculate_metrics(y_train['BFP'] *100, BFP_train_pred)
BFP_test_r2, BFP_test_rmse, BFP_test_me, BFP_test_ci = calculate_metrics(y_test['BFP'] * 100, BFP_test_pred)


df = {
    'Outcome': ['Android', 'Gyroid', 'AGFMR', 'FM', 'FMI', 'BFP', 'LM', 'VAT'],
    'Training_R2': [Android_train_r2, Gynoid_train_r2, A_G_train_r2, FM_train_r2, 
                    FMI_train_r2, BFP_train_r2, LM_train_r2, VAT_train_r2],
    'Training_RMSE': [Android_train_rmse, Gynoid_train_rmse, A_G_train_rmse, FM_train_rmse, 
                      FMI_train_rmse, BFP_train_rmse, LM_train_rmse, VAT_train_rmse],
    'Training_ME': [Android_train_me, Gynoid_train_me, A_G_train_me, FM_train_me,
                    FMI_train_me, BFP_train_me, LM_train_me, VAT_train_me],
    'Training_ME_CI': [Android_train_ci, Gynoid_train_ci, A_G_train_ci, FM_train_ci, 
                       FMI_train_ci, BFP_train_ci, LM_train_ci, VAT_train_ci],
    'Test_R2': [Android_test_r2, Gynoid_test_r2, A_G_test_r2, FM_test_r2, 
                FMI_test_r2, BFP_test_r2, LM_test_r2, VAT_test_r2],
    'Test_RMSE': [Android_test_rmse, Gynoid_test_rmse, A_G_test_rmse, FM_test_rmse, 
                  FMI_test_rmse, BFP_test_rmse, LM_test_rmse, VAT_test_rmse],
    'Test_ME': [Android_test_me, Gynoid_test_me, A_G_test_me, FM_test_me,
                FMI_test_me, BFP_test_me, LM_test_me, VAT_test_me],
    'Test_ME_CI': [Android_test_ci, Gynoid_test_ci, A_G_test_ci, 
                  FM_test_ci, FMI_test_ci, BFP_test_ci, LM_test_ci, VAT_test_ci]
}

# Convert dictionary to DataFrame
df = pd.DataFrame(df)
df.to_csv("your_path/Model_evaluate_indices.csv", index=False)


import statsmodels.formula.api as smf
from patsy import Q

#################################### Add interaction terms ##############################

###################################################### VAT model 
VAT_targets = ["标准站姿腰围", "大腿围度左边.水平线.", "对高臀部的腰部后面", "臂长右", "上臂围度左边", "age"]

# Create dataset
VAT_data = pd.concat([
    y_train[["VATmass"]].rename(columns={"VATmass": "VAT"}),
    X_train[VAT_targets + ['gender']]
], axis=1)


VAT_formula = ('VAT ~ Q("标准站姿腰围") + Q("大腿围度左边.水平线.") + '
           'Q("对高臀部的腰部后面") + Q("臂长右") + Q("上臂围度左边") + '
           'age + gender + '
           'age:Q("标准站姿腰围") + gender:Q("标准站姿腰围")')

VAT_FSLR_interaction = smf.ols(VAT_formula, data=VAT_data).fit()

print(VAT_FSLR_interaction.summary())

## Remove insignificant feature 上臂围度左边

VAT_formula2 = ('VAT ~ Q("标准站姿腰围") + Q("大腿围度左边.水平线.") + '
           'Q("对高臀部的腰部后面") + Q("臂长右") + '
           'age + gender + '
           'age:Q("标准站姿腰围") + gender:Q("标准站姿腰围")')

VAT_FSLR_interaction2 = smf.ols(VAT_formula2, data=VAT_data).fit()

print(VAT_FSLR_interaction2.summary())  ## R-squared:  0.848； 

VAT_Selected_feature2 = ["标准站姿腰围", "大腿围度左边.水平线.", "对高臀部的腰部后面", "臂长右", "age", "gender"]

VAT_train_pred2 = VAT_FSLR_interaction2.predict(add_constant(X_train[VAT_Selected_feature2]))
VAT_test_pred2 = VAT_FSLR_interaction2.predict(add_constant(X_test[VAT_Selected_feature2]))

#  Calculate metrics for training and test sets separately
VAT_train2_r2, VAT_train2_rmse, VAT_train2_me, VAT_train2_ci = calculate_metrics(y_train['VATmass'], VAT_train_pred2)
VAT_test2_r2, VAT_test2_rmse, VAT_test2_me, VAT_test2_ci = calculate_metrics(y_test['VATmass'], VAT_test_pred2)

print("Performance of VAT model with interaction effects is as follows:")
print("Training set performance:" + "r2:" , VAT_train2_r2, "rmse:", VAT_train2_rmse,
      "me:", VAT_train_me, "me_ci:", VAT_train_ci)
print("Test set performance:"+ "r2:" , VAT_test2_r2, "rmse:", VAT_test2_rmse,
      "me:", VAT_test_me, "me_ci:", VAT_test_ci)

# Convert detailed model parameters to a DataFrame
VIF_VAT_params = pd.DataFrame({
    'Parameter': VAT_FSLR_interaction2.params.index,
    'Coefficient': VAT_FSLR_interaction2.params.values,
    'P-value': VAT_FSLR_interaction2.pvalues.values,
    'Std_Err': VAT_FSLR_interaction2.bse.values
})

print(VIF_VAT_params)

# Save DataFrame as CSV file
VIF_VAT_params.to_csv('your_path/VIF_VAT_params_interaction.csv', index=False)


## Add new interaction model to existing model dictionary

AA_20240105_All_VIF_models.update({
    'VAT_inetraction_FSLR': { 'Forward_linear_model': VAT_FSLR_interaction2, 'Selected_feature': VAT_Selected_feature2}
    })

dump(AA_20240105_All_VIF_models, 'your_path/AA_20240105_All_VIF_models2.joblib')

X_VAT_test = X_test[VAT_targets]
X_VAT_test = add_constant(X_VAT_test)
Y_VAT_test = y_test['VATmass']
print("VATmass") # r2: 0.8058659068946479 rmse: 0.23225482234553405
VIF_VAT_params = model_result(X_VAT_test, Y_VAT_test, VAT_forward_linear_model)
VIF_VAT_params.to_csv("your_path/VIF_VAT_params.csv", index=False)


