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

# Set Chinese font and graphic style
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

# Set random seed for reproducibility
np.random.seed(42)

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

models = {
    'FMI_FSLR': ( AA_20240105_All_VIF_models['FMI_forward_linear_model']['Forward_linear_model'], 
                 AA_20240105_All_VIF_models['FMI_forward_linear_model']['Selected_feature'],
                 y_test['FMI'], 'FMI' ) ,
    'A_G_FSLR': ( AA_20240105_All_VIF_models['A_G_forward_linear_model']['Forward_linear_model'], 
                 AA_20240105_All_VIF_models['A_G_forward_linear_model']['Selected_feature'],
                 y_test['A_G'], 'A_G' ),
    'FM_FSLR': ( AA_20240105_All_VIF_models['FM_forward_linear_model']['Forward_linear_model'], 
                AA_20240105_All_VIF_models['FM_forward_linear_model']['Selected_feature'],
                y_test['FM'], 'FM' ),
    'LM_FSLR': ( AA_20240105_All_VIF_models['LM_forward_linear_model']['Forward_linear_model'], 
                AA_20240105_All_VIF_models['LM_forward_linear_model']['Selected_feature'],
                y_test['LM'], 'LM' ),
    'VAT_FSLR': ( AA_20240105_All_VIF_models['VAT_inetraction_FSLR']['Forward_linear_model'], 
                 AA_20240105_All_VIF_models['VAT_inetraction_FSLR']['Selected_feature'],
                 y_test['VATmass'], 'VAT' ),
    'VAT_FSLR_Without_Interaction': ( AA_20240105_All_VIF_models['VAT_forward_linear_model']['Forward_linear_model'], 
                 AA_20240105_All_VIF_models['VAT_forward_linear_model']['Selected_feature'],
                 y_test['VATmass'], 'VAT' ),
    'Android_FSLR': ( AA_20240105_All_VIF_models['Android_forward_linear_model']['Forward_linear_model'],
                      AA_20240105_All_VIF_models['Android_forward_linear_model']['Selected_feature'],
                     y_test['Android'], 'Android' ),
    'Gynoid_FSLR': ( AA_20240105_All_VIF_models['Gynoid_forward_linear_model']['Forward_linear_model'],
                    AA_20240105_All_VIF_models['Gynoid_forward_linear_model']['Selected_feature'],
                    y_test['Gynoid'], 'Gynoid' ),
    'BFP_FSLR': ( AA_20240105_All_VIF_models['BFP_forward_linear_model']['Forward_linear_model'], 
                 AA_20240105_All_VIF_models['BFP_forward_linear_model']['Selected_feature'],
                 y_test['BFP'], 'BFP' )
}

# 3. Define age grouping function
def create_age_groups(age):
    """Divide age into different groups"""
    if age <= 30:
        return '20-30'
    elif age <= 40:
        return '30-40'
    elif age <= 50:
        return '40-50'
    elif age <= 60:
        return '50-60'
    else:
        return '60+'
    

# 4. Subgroup evaluation function
def evaluate_subgroups(models, X_test, test_df):
    """
    Evaluate performance of multiple models on different subgroups
    
    Parameters:
    models: Dictionary containing multiple trained models
    X_test: Test features
    y_test: Test target variables
    test_df: Test DataFrame containing age and gender information
    """
    
    # Copy test dataframe and add age groups
    evaluation_df = test_df.copy()
    evaluation_df['age_group'] = evaluation_df['age'].apply(create_age_groups)
    
    # Define evaluation metrics
    results = []
    
    # Calculate metrics for each model and each subgroup
    for model_name, (model, feature, y_test, target_name) in models.items():
        # Get model predictions
        X_test_subset = add_constant(X_test[feature])
        y_pred = model.predict(X_test_subset)
        evaluation_df[f'pred_{model_name}'] = y_pred
        
        # Evaluate by gender and age group
        for gender in ['Male', 'Female']:
            for age_group in ['20-30', '30-40', '40-50', '50-60', '60+']:
                # Create subgroup mask
                mask = (evaluation_df['gender'] == gender) & (evaluation_df['age_group'] == age_group)
                
                if mask.sum() > 0:  # Ensure there are samples in the subgroup
                    subgroup_true = y_test[mask]
                    subgroup_pred = y_pred[mask]
                    subgroup_mean = np.mean(subgroup_true)
                    
                    # Calculate metrics
                    r2 = r2_score(subgroup_true, subgroup_pred)
                    rmse = np.sqrt(mean_squared_error(subgroup_true, subgroup_pred))
                    norm_rmse = rmse / subgroup_mean
                    n_samples = mask.sum()
                    results.append({
                        'model': model_name,
                        'subgroup': f'{gender}_{age_group}',
                        'gender': gender,
                        'age_group': age_group,
                        'n_samples': n_samples,
                        'r2': r2,
                        'rmse': rmse,
                        'norm_rmse': norm_rmse
                    })
    
    return pd.DataFrame(results)

test_df = X_test[['age', 'gender']].copy()
test_df['gender'] = np.where(test_df['gender'] == 0, 'Female', 'Male')

# Evaluate subgroup performance
print("\nEvaluating subgroup performance...")
results_df = evaluate_subgroups(models, X_test2, test_df)

# Display results
print("\nSubgroup evaluation results:")
print("=" * 60)
    
# Display detailed results by model and subgroup
for model_name in models.keys():
    model_results = results_df[results_df['model'] == model_name]
    print(f"\n{model_name} subgroup performance:")
    print("-" * 40)
    print(f"{'Subgroup':<15} {'Samples':<8} {'R²':<10} {'RMSE':<10}")
    print("-" * 40)
    
    for _, row in model_results.iterrows():
        print(f"{row['subgroup']:<15} {row['n_samples']:<8} {row['r2']:.4f} {row['rmse']:.4f}")
    
    # Create summary table (for comparing different models in same subgroups)
    print("\n\nModel comparison (by subgroup):")
    print("=" * 60)
    
    # Pivot table for comparison
    pivot_r2 = results_df.pivot_table(index='subgroup', columns='model', values='r2')
    pivot_rmse = results_df.pivot_table(index='subgroup', columns='model', values='rmse')


print("\nR² scores:")
print(pivot_r2.round(4))
    
print("\nRMSE scores:")
print(pivot_rmse.round(4))


results_df.to_csv('your_path/subgroup_analysis_results.csv', index=True)

pivot_r2.to_csv('your_path/r2_model_comparison.csv', index=True)
pivot_rmse.to_csv('your_path/rmse_model_comparison2.csv', index=True)


##########################################

def plot_male_r2_with_sample_size(results_df, save_path=None):
    """Plot R² line chart for males with sample size bars and save the figure"""
    
    # Filter male data
    male_data = results_df[results_df['gender'] == 'Male'].copy()
    
    # Define age group order
    age_order = ['20-30', '30-40', '40-50', '50-60', '60+']
    male_data['age_group'] = pd.Categorical(male_data['age_group'], categories=age_order, ordered=True)
    
    # Calculate average sample size per age group and filter out groups with 0 samples
    sample_sizes = male_data.groupby('age_group')['n_samples'].mean().fillna(0)
    
    # Filter out age groups with 0 samples
    valid_age_groups = sample_sizes[sample_sizes > 0].index.tolist()
    
    if not valid_age_groups:
        print("Warning: No valid age group data (all age groups have 0 samples)")
        return None
    
    # Filter data to only include valid age groups
    male_data = male_data[male_data['age_group'].isin(valid_age_groups)]
    sample_sizes = sample_sizes[sample_sizes > 0]
    
    # Create figure and primary axis for R²
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # Create secondary axis for sample size
    ax2 = ax1.twinx()
    
    # Get all model names
    models = male_data['model'].unique()
    
    # Color palette for models
    colors = plt.cm.Set3(np.linspace(0, 1, len(models)))
    
    # Plot R² = 0.5 reference line (thinner)
    ax1.axhline(y=0.5, color='red', linestyle='--', linewidth=1, alpha=0.7, label='R² = 0.5')
    
    # Plot R² LINES on primary axis (left)
    lines = []
    for i, model in enumerate(models):
        model_data = male_data[male_data['model'] == model].sort_values('age_group')
        if not model_data.empty:
            line, = ax1.plot(model_data['age_group'], model_data['r2'], 
                            marker='o', linewidth=2.5, markersize=8, 
                            label=model, color=colors[i])
            lines.append(line)
    
    # Plot sample size BARS on secondary axis (right)
    bars = ax2.bar(range(len(sample_sizes)), sample_sizes.values, 
                   alpha=0.3, color='gray', width=0.6,
                   label='Test sample Size')

    # Set primary axis properties (R²)
    ax1.set_xlabel('Age Group', fontsize=12, fontweight='bold')
    ax1.set_ylabel('R² Score', fontsize=12, fontweight='bold')
    ax1.set_title('R² Performance with Sample Size - Male', fontsize=14, fontweight='bold')
    
    # Remove grid
    ax1.grid(False)
    ax1.set_ylim(bottom=0)
    
    # Add "0.5" label on the left y-axis
    ax1.axhline(y=0.5, color='red', linestyle='--', linewidth=1, alpha=0.7)
    
    # Set secondary axis properties (Sample Size)
    ax2.set_ylabel('Test sample Size', fontsize=12, fontweight='bold')
    ax2.set_ylim(0, 258)
    
    # Remove grid for secondary axis
    ax2.grid(False)
    
    # Add sample size values on bars
    for i, (bar, size) in enumerate(zip(bars, sample_sizes.values)):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 2,
                f'{int(size)}', ha='center', va='bottom', 
                fontweight='bold', fontsize=9)
    
    # Set all text to black
    ax1.title.set_color('black')
    ax1.xaxis.label.set_color('black')
    ax1.yaxis.label.set_color('black')
    ax1.tick_params(axis='x', colors='black')
    ax1.tick_params(axis='y', colors='black')
    
    ax2.yaxis.label.set_color('black')
    ax2.tick_params(axis='y', colors='black')
    
    # Combine all legends (models + sample size + reference line)
    reference_line = plt.Line2D([0], [0], color='red', linestyle='--', linewidth=1, label='R² = 0.5')
    all_handles = [reference_line] + lines + [bars[0]]
    all_labels = ['R² = 0.5'] + [line.get_label() for line in lines] + ['Test sample Size']
    
    # Add combined legend outside the plot on the right side
    ax1.legend(handles=all_handles, labels=all_labels, 
               loc='center left', bbox_to_anchor=(1.15, 0.5),
               fontsize=10, frameon=True)
    
    # Adjust layout to make space for the legend
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    
    # Save the figure if save_path is provided
    if save_path:
        import os
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=600, bbox_inches='tight', facecolor='white')
        print(f"Figure saved to: {save_path}")
    
    plt.show()
    
    return male_data

# Call function and save the figure
male_r2_data = plot_male_r2_with_sample_size(results_df, save_path='your_path/male_r2_performance3.png')

def plot_female_r2_with_sample_size(results_df, save_path=None):
    """Plot R² line chart for females with sample size bars and save the figure"""
    
    # Filter female data
    female_data = results_df[results_df['gender'] == 'Female'].copy()
    
    # Define age group order
    age_order = ['20-30', '30-40', '40-50', '50-60', '60+']
    female_data['age_group'] = pd.Categorical(female_data['age_group'], categories=age_order, ordered=True)
    
    # Calculate average sample size per age group and filter out groups with 0 samples
    sample_sizes = female_data.groupby('age_group')['n_samples'].mean().fillna(0)
    
    # Filter out age groups with 0 samples
    valid_age_groups = sample_sizes[sample_sizes > 0].index.tolist()
    
    if not valid_age_groups:
        print("Warning: No valid age group data (all age groups have 0 samples)")
        return None
    
    # Filter data to only include valid age groups
    female_data = female_data[female_data['age_group'].isin(valid_age_groups)]
    sample_sizes = sample_sizes[sample_sizes > 0]
    
    # Create figure and primary axis for R²
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # Create secondary axis for sample size
    ax2 = ax1.twinx()
    
    # Get all model names
    models = female_data['model'].unique()
    
    # Color palette for models
    colors = plt.cm.Set3(np.linspace(0, 1, len(models)))
    
    # Plot R² = 0.5 reference line (thinner)
    ax1.axhline(y=0.5, color='red', linestyle='--', linewidth=1, alpha=0.7, label='R² = 0.5')
    
    # Plot R² LINES on primary axis (left)
    lines = []
    for i, model in enumerate(models):
        model_data = female_data[female_data['model'] == model].sort_values('age_group')
        if not model_data.empty:
            line, = ax1.plot(model_data['age_group'], model_data['r2'], 
                            marker='s', linewidth=2.5, markersize=8, 
                            label=model, color=colors[i])
            lines.append(line)
    
    # Plot sample size BARS on secondary axis (right)
    bars = ax2.bar(range(len(sample_sizes)), sample_sizes.values, 
                   alpha=0.3, color='gray', width=0.6,
                   label='Test sample Size')
    
    # Set primary axis properties (R²)
    ax1.set_xlabel('Age Group', fontsize=12, fontweight='bold')
    ax1.set_ylabel('R² Score', fontsize=12, fontweight='bold')
    ax1.set_title('R² Performance with Sample Size - Female', fontsize=14, fontweight='bold')
    
    # Remove grid
    ax1.grid(False)
    ax1.set_ylim(bottom=-0.2)
    
    # Add "0.5" label on the left y-axis
    ax1.axhline(y=0.5, color='red', linestyle='--', linewidth=1, alpha=0.7)
    
    # Set secondary axis properties (Sample Size)
    ax2.set_ylabel('Test sample Size', fontsize=12, fontweight='bold')
    ax2.set_ylim(0, 258)
    
    # Remove grid for secondary axis
    ax2.grid(False)
    
    # Add sample size values on bars
    for i, (bar, size) in enumerate(zip(bars, sample_sizes.values)):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 2,
                f'{int(size)}', ha='center', va='bottom', 
                fontweight='bold', fontsize=9)
    
    # Set all text to black
    ax1.title.set_color('black')
    ax1.xaxis.label.set_color('black')
    ax1.yaxis.label.set_color('black')
    ax1.tick_params(axis='x', colors='black')
    ax1.tick_params(axis='y', colors='black')
    
    ax2.yaxis.label.set_color('black')
    ax2.tick_params(axis='y', colors='black')
    
    # Combine all legends (models + sample size + reference line)
    reference_line = plt.Line2D([0], [0], color='red', linestyle='--', linewidth=1, label='R² = 0.5')
    all_handles = [reference_line] + lines + [bars[0]]
    all_labels = ['R² = 0.5'] + [line.get_label() for line in lines] + ['Test sample Size']
    
    # Add combined legend outside the plot on the right side
    ax1.legend(handles=all_handles, labels=all_labels, 
               loc='center left', bbox_to_anchor=(1.15, 0.5),
               fontsize=10, frameon=True)
    
    # Adjust layout to make space for the legend
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    
    # Save the figure if save_path is provided
    if save_path:
        import os
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=600, bbox_inches='tight', facecolor='white')
        print(f"Figure saved to: {save_path}")
    
    plt.show()
    
    return female_data

# Call function and save the figure
female_r2_data = plot_female_r2_with_sample_size(results_df, save_path='your_path/female_r2_performance3.png')

def plot_male_rmse_with_sample_size(results_df, save_path=None):
    """Plot R² line chart for males with sample size bars and save the figure"""
    
    # Filter male data
    male_data = results_df[results_df['gender'] == 'Male'].copy()
    
    # Define age group order
    age_order = ['20-30', '30-40', '40-50', '50-60', '60+']
    male_data['age_group'] = pd.Categorical(male_data['age_group'], categories=age_order, ordered=True)
    
    # Calculate average sample size per age group and filter out groups with 0 samples
    sample_sizes = male_data.groupby('age_group')['n_samples'].mean().fillna(0)
    
    # Filter out age groups with 0 samples
    valid_age_groups = sample_sizes[sample_sizes > 0].index.tolist()
    
    if not valid_age_groups:
        print("Warning: No valid age group data (all age groups have 0 samples)")
        return None
    
    # Filter data to only include valid age groups
    male_data = male_data[male_data['age_group'].isin(valid_age_groups)]
    sample_sizes = sample_sizes[sample_sizes > 0]
    
    # Create figure and primary axis for R²
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # Create secondary axis for sample size
    ax2 = ax1.twinx()
    
    # Get all model names
    models = male_data['model'].unique()
    
    # Color palette for models
    colors = plt.cm.Set3(np.linspace(0, 1, len(models)))
    
    # Plot R² = 0.5 reference line (thinner)
    # ax1.axhline(y=0.5, color='red', linestyle='--', linewidth=1, alpha=0.7, label='R² = 0.5')
    
    
    # Plot R² LINES on primary axis (left)
    lines = []
    for i, model in enumerate(models):
        model_data = male_data[male_data['model'] == model].sort_values('age_group')
        if not model_data.empty:
            line, = ax1.plot(model_data['age_group'], model_data['norm_rmse'], 
                            marker='o', linewidth=2.5, markersize=8, 
                            label=model, color=colors[i])
            lines.append(line)
    
    # Plot sample size BARS on secondary axis (right)
    bars = ax2.bar(range(len(sample_sizes)), sample_sizes.values, 
                   alpha=0.3, color='gray', width=0.6,
                   label='Test sample Size')
    
    # Set primary axis properties (R²)
    ax1.set_xlabel('Age Group', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Normalized RMSE Score', fontsize=12, fontweight='bold')
    ax1.set_title('Normalized RMSE Performance with Sample Size - Male', fontsize=14, fontweight='bold')
    
    # Remove grid
    ax1.grid(False)
    ax1.set_ylim(bottom=0)
    
    # Add "0.5" label on the left y-axis
    ax1.axhline(y=0.5, color='red', linestyle='--', linewidth=1, alpha=0.7)
    
    # Set secondary axis properties (Sample Size)
    ax2.set_ylabel('Test sample Size', fontsize=12, fontweight='bold')
    ax2.set_ylim(0, 258)
    
    # Remove grid for secondary axis
    ax2.grid(False)
    
    # Add sample size values on bars
    for i, (bar, size) in enumerate(zip(bars, sample_sizes.values)):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 2,
                f'{int(size)}', ha='center', va='bottom', 
                fontweight='bold', fontsize=9)
    
    # Set all text to black
    ax1.title.set_color('black')
    ax1.xaxis.label.set_color('black')
    ax1.yaxis.label.set_color('black')
    ax1.tick_params(axis='x', colors='black')
    ax1.tick_params(axis='y', colors='black')
    
    ax2.yaxis.label.set_color('black')
    ax2.tick_params(axis='y', colors='black')
    
    # Combine all legends (models + sample size + reference line)
    # reference_line = plt.Line2D([0], [0], color='red', linestyle='--', linewidth=1, label='R² = 0.5')
    all_handles = lines + [bars[0]]
    all_labels =  [line.get_label() for line in lines] + ['Test sample Size']
    
    # Add combined legend outside the plot on the right side
    ax1.legend(handles=all_handles, labels=all_labels, 
               loc='center left', bbox_to_anchor=(1.15, 0.5),
               fontsize=10, frameon=True)
    
    # Adjust layout to make space for the legend
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    
    # Save the figure if save_path is provided
    if save_path:
        import os
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=600, bbox_inches='tight', facecolor='white')
        print(f"Figure saved to: {save_path}")
    
    plt.show()
    
    return male_data

# Call function and save the figure
male_rmse_data = plot_male_rmse_with_sample_size(results_df, save_path='your_path/male_rmse_performance3.png')

def plot_female_rmse_with_sample_size(results_df, save_path=None):
    """Plot RMSE line chart for females with sample size bars and save the figure"""
    
    # Filter female data
    female_data = results_df[results_df['gender'] == 'Female'].copy()
    
    # Define age group order
    age_order = ['20-30', '30-40', '40-50', '50-60', '60+']
    female_data['age_group'] = pd.Categorical(female_data['age_group'], categories=age_order, ordered=True)
    
    # Calculate average sample size per age group and filter out groups with 0 samples
    sample_sizes = female_data.groupby('age_group')['n_samples'].mean().fillna(0)
    
    # Filter out age groups with 0 samples
    valid_age_groups = sample_sizes[sample_sizes > 0].index.tolist()
    
    if not valid_age_groups:
        print("Warning: No valid age group data (all age groups have 0 samples)")
        return None
    
    # Filter data to only include valid age groups
    female_data = female_data[female_data['age_group'].isin(valid_age_groups)]
    sample_sizes = sample_sizes[sample_sizes > 0]
    
    # Create figure and primary axis for R²
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # Create secondary axis for sample size
    ax2 = ax1.twinx()
    
    # Get all model names
    models = female_data['model'].unique()
    
    # Color palette for models
    colors = plt.cm.Set3(np.linspace(0, 1, len(models)))
       
    # Plot RMSE LINES on primary axis (left)
    lines = []
    for i, model in enumerate(models):
        model_data = female_data[female_data['model'] == model].sort_values('age_group')
        if not model_data.empty:
            line, = ax1.plot(model_data['age_group'], model_data['norm_rmse'], 
                            marker='s', linewidth=2.5, markersize=8, 
                            label=model, color=colors[i])
            lines.append(line)
    
    # Plot sample size BARS on secondary axis (right)
    bars = ax2.bar(range(len(sample_sizes)), sample_sizes.values, 
                   alpha=0.3, color='gray', width=0.6,
                   label='Test sample Size')
    
    # Set primary axis properties (R²)
    ax1.set_xlabel('Age Group', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Normalized RMSE Score', fontsize=12, fontweight='bold')
    ax1.set_title('Normalized RMSE Performance with Sample Size - Female', fontsize=14, fontweight='bold')
    
    # Remove grid
    ax1.grid(False)
    ax1.set_ylim(bottom=0)
    
    # Add "0.5" label on the left y-axis
    # ax1.axhline(y=0.5, color='red', linestyle='--', linewidth=1, alpha=0.7)
    
    # Set secondary axis properties (Sample Size)
    ax2.set_ylabel('Test sample Size', fontsize=12, fontweight='bold')
    ax2.set_ylim(0, 258)
    
    # Remove grid for secondary axis
    ax2.grid(False)
    
    # Add sample size values on bars
    for i, (bar, size) in enumerate(zip(bars, sample_sizes.values)):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 2,
                f'{int(size)}', ha='center', va='bottom', 
                fontweight='bold', fontsize=9)
    
    # Set all text to black
    ax1.title.set_color('black')
    ax1.xaxis.label.set_color('black')
    ax1.yaxis.label.set_color('black')
    ax1.tick_params(axis='x', colors='black')
    ax1.tick_params(axis='y', colors='black')
    
    ax2.yaxis.label.set_color('black')
    ax2.tick_params(axis='y', colors='black')
    
    # Combine all legends (models + sample size + reference line)
    all_handles = lines + [bars[0]]
    all_labels =  [line.get_label() for line in lines] + ['Test sample Size']
    
    # Add combined legend outside the plot on the right side
    ax1.legend(handles=all_handles, labels=all_labels, 
               loc='center left', bbox_to_anchor=(1.15, 0.5),
               fontsize=10, frameon=True)
    
    # Adjust layout to make space for the legend
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    
    # Save the figure if save_path is provided
    if save_path:
        import os
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=600, bbox_inches='tight', facecolor='white')
        print(f"Figure saved to: {save_path}")
    
    plt.show()
    
    return female_data

# Call function and save the figure
female_rmse_data = plot_female_rmse_with_sample_size(results_df, save_path='your_path/female_rmse_performance3.png')