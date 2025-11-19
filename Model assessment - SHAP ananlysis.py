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
from sklearn.inspection import permutation_importance
import os
from joblib import dump
from joblib import load

import pprint
import json
import shap

# Set Chinese font
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']  # For normal display of Chinese labels
plt.rcParams['axes.unicode_minus'] = False  # For normal display of negative signs

import warnings
warnings.filterwarnings("ignore")



## Import data
X_train = pd.read_csv("your_path/X_train_All3D_NoScaled.csv",sep=",")
X_train.set_index(X_train.columns[0], inplace= True)  ## Set the first column as the row index of the dataframe
name = X_train.columns.tolist()    ## Extract column names of the dataframe
name = [item.replace('人体外观测量.三维人体扫描分析系统.', '') for item in name]    ## Remove "人体外观测量-三维人体扫描分析系统:" part from column names
# Remove ".cm." from all column names
name = [col.replace('.cm.', '') for col in name]
X_train.columns = name   ## Replace dataframe column names with simplified names
print(X_train)
X_train = X_train.drop(columns = ['height', 'Waist_To_Hip_Ratio.x.100.'])

## Remove non-repeatable phenotypes - phenotype features related to instrument measurement
exclude_keywords = ['3D', '腰带', '垂直', '内部', '躯干扭', '角度', 'Sideseam']
initial_features = [col for col in X_train.columns 
                       if not any(keyword in col for keyword in exclude_keywords)]

X_train2 = X_train[initial_features]

X_test = pd.read_csv("your_path/X_test_All3D_NoScaled.csv",sep=",")
X_test.set_index(X_test.columns[0], inplace= True)  ## Set the first column as the row index of the dataframe
X_test.columns = name   ## Replace dataframe column names with simplified names
X_test = X_test.drop(columns = ['height', 'Waist_To_Hip_Ratio.x.100.'])
X_test2 = X_test[initial_features]

y_train = pd.read_csv("your_path/y_train_All3D_NoScaled.csv",sep=",")
y_train.set_index(y_train.columns[0], inplace= True)  ## Set the first column as the row index of the dataframe
y_train['BFP'] = y_train['BFP'] * 100

y_test = pd.read_csv("your_path/y_test_All3D_NoScaled.csv",sep=",")
y_test.set_index(y_test.columns[0], inplace= True)  ## Set the first column as the row index of the dataframe
y_test['BFP'] = y_test['BFP'] * 100

## Load previously trained models and parameters using training set
AA_20240105_All_VIF_models = load('your_path/AA_20240105_All_VIF_models2.joblib')

VIF_A_G_params = pd.read_csv("your_path/VIF_A_G_params.csv" )
VIF_Android_params = pd.read_csv("your_path/VIF_Android_params.csv" )
VIF_Gynoid_params = pd.read_csv("your_path/VIF_Gynoid_params.csv" )
VIF_FM_params = pd.read_csv("your_path/VIF_FM_params.csv" )
VIF_FMI_params = pd.read_csv("your_path/VIF_FMI_params.csv" )
VIF_LM_params = pd.read_csv("your_path/VIF_LM_params.csv" )
VIF_VAT_params = pd.read_csv("your_path/VIF_VAT_params_interaction.csv")
VIF_BFP_params = pd.read_csv("your_path/VIF_BFP_params.csv" )

VIF_VAT_params = ["标准站姿腰围", "大腿围度左边.水平线.", "对高臀部的腰部后面",   "臂长右", "age",  "gender"]

# Specify the specific feature list you want to display
A_G_features = VIF_A_G_params['Parameter'].iloc[1:-2].tolist()
Android_features = VIF_Android_params['Parameter'].iloc[1:-2].tolist()
Gynoid_features = VIF_Gynoid_params['Parameter'].iloc[1:-2].tolist()
FM_features = VIF_FM_params['Parameter'].iloc[1:-2].tolist()
FMI_features = VIF_FMI_params['Parameter'].iloc[1:-2].tolist()
LM_features = VIF_LM_params['Parameter'].iloc[1:-2].tolist()
VAT_features = ["标准站姿腰围", "大腿围度左边.水平线.", "对高臀部的腰部后面",   "臂长右", "age", "gender"]
BFP_features = VIF_BFP_params['Parameter'].iloc[1:-2].tolist()

# Define feature set corresponding to each target variable
feature_mapping = {
    'A_G': A_G_features,
    'Android': Android_features,
    'Gynoid': Gynoid_features,
    'FM': FM_features,
    'FMI': FMI_features,
    'LM': LM_features,
    'VATmass': VAT_features,
    'BFP': BFP_features
}

def train_multiple_xgb_simple(X_train, X_test, y_train, y_test, 
                            target_columns=None, feature_mapping=None, n_iter=20, cv=5):
    """
    Improved batch training function - supports using different feature sets for each target variable
    
    Parameters:
    - feature_mapping: dictionary, format {target_name: [feature1, feature2, ...]}
    """
    if target_columns is None:
        target_columns = y_train.columns.tolist()
    
    if feature_mapping is None:
        feature_mapping = {}
    
    results = {}
    
    for target_name in target_columns:
        print(f"Training {target_name}...")
        
        # Get feature set corresponding to this target variable
        if target_name in feature_mapping:
            target_features = feature_mapping[target_name]
            # Ensure all features exist in training set
            available_features = [f for f in target_features if f in X_train.columns]
            missing_features = set(target_features) - set(available_features)
            
            if missing_features:
                print(f"  Warning: The following features do not exist and will be skipped: {missing_features}")
            
            if not available_features:
                print(f"  Error: {target_name} has no available features, skipping training")
                continue
                
            X_train_subset = X_train[available_features]
            X_test_subset = X_test[available_features]
            
            print(f"  Using {len(available_features)} features: {available_features}")
        else:
            # If no feature mapping specified, use all features
            X_train_subset = X_train
            X_test_subset = X_test
            print(f"  Using all {X_train.shape[1]} features")
        
        # Base model
        model = xgb.XGBRegressor(random_state=42, n_jobs=-1)
        
        # Parameter grid
        param_grid = {
            'n_estimators': [100, 200, 500],
            'learning_rate': [0.01, 0.05, 0.1],
            'max_depth': [3, 5, 7],
            'subsample': [0.8, 0.9, 1.0],
            'colsample_bytree': [0.8, 0.9, 1.0]
        }
        
        # Random search
        search = RandomizedSearchCV(
            model, param_grid, n_iter=n_iter, cv=cv,
            scoring='neg_mean_squared_error', n_jobs=-1, random_state=42
        )
        
        search.fit(X_train_subset, y_train[target_name].values.ravel())
        
        # Final model
        final_model = xgb.XGBRegressor(**search.best_params_, random_state=42, n_jobs=-1)
        final_model.fit(X_train_subset, y_train[target_name])
        
        # Evaluation
        y_pred = final_model.predict(X_test_subset)
        test_r2 = r2_score(y_test[target_name], y_pred)
        test_rmse = np.sqrt(mean_squared_error(y_test[target_name], y_pred))
        
        results[target_name] = {
            'model': final_model,
            'best_params': search.best_params_,
            'test_r2': test_r2,
            'test_rmse': test_rmse,
            'features_used': available_features if target_name in feature_mapping else X_train.columns.tolist(),
            'feature_count': len(available_features) if target_name in feature_mapping else X_train.shape[1]
        }
        
        print(f"  {target_name} - R²: {test_r2:.4f}, RMSE: {test_rmse:.4f}")
        print(f"  Feature count: {results[target_name]['feature_count']}\n")
    
    return results

XGB_results = train_multiple_xgb_simple(
        X_train=X_train2,
        X_test=X_test2,
        y_train=y_train, 
        y_test=y_test,
        target_columns=['A_G', 'Android', 'Gynoid', 'BFP', 'FM', 'FMI', 'LM', 'VATmass'],
        feature_mapping= feature_mapping
 )
    
## Save dictionary

dump(XGB_results, 'your_path/XGB_models_specific.joblib')

## Load previously trained models and parameters
XGB_results = load('your_path/XGB_models_specific.joblib')


### A_G Model

A_G_xgb_model = XGB_results['A_G']['model']

# Validate final model performance on test set
A_G_r2 = XGB_results['A_G']['test_r2']
A_G_rmse = XGB_results['A_G']['test_rmse']

print(f"XGBoost model performance on test set: R² = {A_G_r2:.4f}, RMSE = {A_G_rmse:.4f}")  ##  R² = 0.8325, RMSE = 0.0729

### Android Model

Android_xgb_model = XGB_results['Android']['model']

# Validate final model performance on test set
Android_r2 = XGB_results['Android']['test_r2']
Android_rmse = XGB_results['Android']['test_rmse']

print(f"XGBoost model performance on test set: R² = {Android_r2:.4f}, RMSE = {Android_rmse:.4f}")  ##  R² = 0.8830, RMSE = 0.2543

### BFP Model

BFP_xgb_model = XGB_results['BFP']['model']

BFP_r2 =  XGB_results['BFP']['test_r2']
BFP_rmse = XGB_results['BFP']['test_rmse']

print(f"XGBoost model performance on test set: R² = {BFP_r2:.4f}, RMSE = {BFP_rmse:.4f}")  ##  R² = 0.7454, RMSE = 3.5262

### FM Model

FM_xgb_model = XGB_results['FM']['model']

FM_r2 = XGB_results['FM']['test_r2']
FM_rmse = XGB_results['FM']['test_rmse']

print(f"XGBoost model performance on test set: R² = {FM_r2:.4f}, RMSE = {FM_rmse:.4f}")  ##   R² = 0.8514, RMSE = 2.3591

### FMI Model

FMI_xgb_model = XGB_results['FMI']['model']

FMI_r2 = XGB_results['FMI']['test_r2']
FMI_rmse = XGB_results['FMI']['test_rmse']

print(f"XGBoost model performance on test set: R² = {FMI_r2:.4f}, RMSE = {FMI_rmse:.4f}")  ##   R² = 0.8567, RMSE = 0.8644

### Gynoid Model

Gynoid_xgb_model = XGB_results['Gynoid']['model']

Gynoid_r2 = XGB_results['Gynoid']['test_r2']
Gynoid_rmse = XGB_results['Gynoid']['test_rmse']

print(f"XGBoost model performance on test set: R² = {Gynoid_r2:.4f}, RMSE = {Gynoid_rmse:.4f}")  ##  R² = 0.8473, RMSE = 0.3916

### LM Model

LM_xgb_model = XGB_results['LM']['model']

LM_r2 = XGB_results['LM']['test_r2']
LM_rmse = XGB_results['LM']['test_rmse']

print(f"XGBoost model performance on test set: R² = {LM_r2:.4f}, RMSE = {LM_rmse:.4f}")  ##  R² = 0.9139, RMSE = 2.4336

### VAT Model

VAT_xgb_model = XGB_results['VATmass']['model']

VAT_r2 = XGB_results['VATmass']['test_r2']
VAT_rmse = XGB_results['VATmass']['test_rmse']

print(f"XGBoost model performance on test set: R² = {VAT_r2:.4f}, RMSE = {VAT_rmse:.4f}")  ##  R² = 0.8211, RMSE = 0.2107


model_configs = {
    'A_G': {
        'model': A_G_xgb_model,
        'features': A_G_features
    },
    'Android': {
        'model': Android_xgb_model,
        'features': Android_features,
    },
    'Gynoid': {
        'model': Gynoid_xgb_model,
        'features': Gynoid_features
    },
    'FM': {
        'model': FM_xgb_model,
        'features': FM_features
        },
    'FMI': {
        'model': FMI_xgb_model,
        'features': FMI_features
        },
    'LM': {
        'model': LM_xgb_model,
        'features': LM_features
        },
    'VAT': {
        'model': VAT_xgb_model,
        'features': VAT_features
        },
    'BFP':{
        'model': BFP_xgb_model,
        'features': BFP_features
        }

}


##############  Plot SHAP summary plots ###################

shap_dir2 = "your_path"

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 12
plt.rcParams['pdf.fonttype'] = 42  # Ensure font embedding
plt.rcParams['ps.fonttype'] = 42   # Ensure font embedding

def plot_shap_analysis2(model, target_name, X_data, specific_features, model_type='tree', save_dir=None):
    """
    Plot SHAP analysis summary plot (display all features)
    
    Parameters:
    - model: Trained model
    - X_data: Feature data (DataFrame)
    - model_type: Model type ('tree', 'general'), default 'tree'
    """
    
    # Create variable name mapping dictionary
    name_mapping = {
        "标准站姿腰围": "Waist girth (cm)",
        "大腿围度左边.水平线.": "Thigh girth left (horizontal, cm)",
        "对高臀部的腰部后面": "Waist to high hip back (cm)",
        "臂长右": "Arm length right (cm)",
        "膝围度左边": "Knee girth left (cm)",
        "脚踝围左边": "Ankle girth left (cm)",
        "前臂围度右边": "Forearm girth right (cm)",
        "胯部长度": "Crotch length (cm)",
        "头高": "Head height (cm)",
        "腿肚围度左边": "Calf girth left (cm)",
        "最小腿围度左边": "Min leg girth left (cm)",
        "中臀围": "Buttock girth (cm)",
        "身体高度": "Body height (cm)",
        "脚踝围右边": "Ankle girth right (cm)",
        "臂长左": "Arm length left (cm)",
        "上臂围度右边": "Upper arm girth right (cm)",
        "高臀部围度": "High hip grith (cm)",
        "宽度腋窝": "Width armpits (cm)",
        "臀部.大腿围度": "Hip/thigh girth (cm)",
        "腰部对臀部高度左边": "Waist to buttock left (cm)",
        "大腿围度右边.水平线.": "Thigh girth right (horizontal, cm)",
        "腿肚围度右边": "Calf girth right (cm)",
        "横过后面宽度.腋窝水平.": "Across back width (armpit level, cm)",
        "胯长度.后面": "Crotch length rear (cm)",
        "上臂长度左边": "Upper arm length left (cm)",
        "臀部围度1": "Hip girth (cm)",
        "中间的脖子围度": "Mid neck girth (cm)",
        "胸部.胸围度": "Bust/chest girth (cm)",
        "脖子到横过后面宽度.腋窝水平.": "Neck to across back width (armpit level, cm)",
        "前臂长度左边": "Forearm length left (cm)",
        "脖子左边对腰部后面": "Neck left to waist back (cm)",
        "颈到臀距离": "Distance neck to buttock (cm)",
        # New mappings (Chinese variables found in plots)
        "高腰围": "High waist girth (cm)",
        "臀部围度": "Hip girth (cm)",
        "腹圆周": "Belly circumference (cm)",
        "Underbust圆周.水平线.": "Under Bust circumference (horizontal, cm)",
        "最大值腹圆周": "Maximum belly circumference (cm)",
        "前臂长度右边": "Forearm length right (cm)",
        "胸部.胸围度.水平线.": "Bust/chest girth (horizontal, cm)",
        "胯长度.前面": "Crotch length front (cm)",
        "脖子.膝距离": "Distance neck-knee (cm)",
        "膝围度右边": "Knee girth right (cm)",
        "肩胛骨的高度": "Scapula height (cm)",
        "总数躯干围度": "Total torso girth (cm)",
        "脖子直径": "Neck diameter (cm)",
        "在腰部的躯干宽度": "Torso width at waist (cm)"
        }
    
    def translate_feature_name(feature_name):
        """Translate feature name from Chinese to English"""
        return name_mapping.get(feature_name, feature_name)
    
    # Calculate SHAP values
    print("Calculating SHAP values...")
    if model_type == 'tree':
        explainer = shap.TreeExplainer(model)
    else:
        explainer = shap.Explainer(model, X_data[specific_features])
    
    shap_values = explainer.shap_values(X_data[specific_features])
    
    # Handle SHAP value format for classification problems
    if isinstance(shap_values, list):
        if len(shap_values) == 2:  # Binary classification
            shap_values = shap_values[1]
        else:  # Multi-class, take first class or average
            shap_values = np.mean(shap_values, axis=0)
    
    print(f"SHAP values calculation completed! Shape: {shap_values.shape}")
    
    # Calculate feature importance
    shap_importance = np.abs(shap_values).mean(axis=0)
    importance_df = pd.DataFrame({
        'feature': X_data[specific_features].columns,
        'importance': shap_importance
    }).sort_values('importance', ascending=False)
    
    selected_features = importance_df['feature'].tolist()  # Fix 1: Convert to list
    
    # Create new feature names with ranking (using English)
    selected_features_ranked = []
    for i, feature in enumerate(selected_features):
        english_name = translate_feature_name(feature)
        ranked_name = f"#{i+1} {english_name}"  # Fix 2: Directly create ranked name
        selected_features_ranked.append(ranked_name)
    
    print(f"\nFinal selected feature count: {len(selected_features)}")
    print("Feature list (with ranking):")
    for i, feature in enumerate(selected_features_ranked):
        print(f"  {feature}")
    
    # Get indices of these features in data - Fix 3: Use specific_features subset indices
    selected_indices = [X_data[specific_features].columns.get_loc(f) for f in selected_features]
    
    # ===== SHAP summary plot (display all features) =====
    plt.figure(figsize=(10, 8))
    
    # Plot SHAP summary plot using all features
    shap.summary_plot(
        shap_values[:, selected_indices],  # Use correct indices
        X_data[specific_features][selected_features],  # Ensure correct data subset
        feature_names=selected_features_ranked,
        show=False,
        max_display=len(selected_features)  # Fix 4: Ensure display all features
    )
    plt.title(f"SHAP Summary Plot: {target_name.replace(' ', '_')} Model", fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # Save image
    if save_dir:
        filename = f"Last_{target_name.replace(' ', '_')}_shap_summary_all_features.png"
        filepath = os.path.join(save_dir, filename)
        plt.savefig(filepath, dpi=600, bbox_inches='tight', facecolor='white')
    plt.show()
    plt.close()
    
    # Output detailed information
    print("\n" + "="*60)
    print("Feature Importance Ranking")
    print("="*60)
    for i, (feature, ranked_name) in enumerate(zip(selected_features, selected_features_ranked)):
        importance_val = importance_df[importance_df['feature'] == feature]['importance'].values[0]
        print(f"{ranked_name:40} Importance: {importance_val:.4f}")
        
    return {
        'shap_values': shap_values,
        'explainer': explainer,
        'importance_df': importance_df,
        'name_mapping': name_mapping,
        'selected_features': selected_features,
        'selected_features_ranked': selected_features_ranked
    }

for target_name, config in model_configs.items():
    # Plot SHAP analysis plots
    shap_results = plot_shap_analysis2(
        model=config['model'],
        target_name=target_name,
        X_data=X_test2, 
        specific_features=config['features'],
        save_dir=shap_dir2
    )

