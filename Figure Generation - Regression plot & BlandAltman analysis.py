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

import warnings
warnings.filterwarnings("ignore")


## ----------------------------------------------------------Import Data---------------------------------------------------------
X_train = pd.read_csv("your_path/X_train_All3D_NoScaled.csv",sep=",")
X_train.set_index(X_train.columns[0], inplace= True)  ## Set the first column as the row index of the dataframe
name = X_train.columns.tolist()    ## Extract column names of the dataframe
name = [item.replace('人体外观测量.三维人体扫描分析系统.', '') for item in name]    ## Remove "人体外观测量-三维人体扫描分析系统:" part from column names
name = [item.replace('.cm.', '') for item in name]    ## Remove ".cm." part from column names
X_train.columns = name   ## Replace dataframe column names with simplified names
# print(X_train)
X_train = X_train.drop(columns = ['height', 'Waist_To_Hip_Ratio.x.100.'])
print(X_train)

X_test = pd.read_csv("your_path/X_test_All3D_NoScaled.csv",sep=",")
X_test.set_index(X_test.columns[0], inplace= True)  ## Set the first column as the row index of the dataframe
X_test.columns = name   ## Replace dataframe column names with simplified names
X_test = X_test.drop(columns = ['height', 'Waist_To_Hip_Ratio.x.100.'])
# print(X_test)

y_train = pd.read_csv("your_path/y_train_All3D_NoScaled.csv",sep=",")
y_train.set_index(y_train.columns[0], inplace= True)  ## Set the first column as the row index of the dataframe

y_test = pd.read_csv("your_path/y_test_All3D_NoScaled.csv",sep=",")
y_test.set_index(y_test.columns[0], inplace= True)  ## Set the first column as the row index of the dataframe


## Load previously trained models and parameters using training set
AA_20240105_All_VIF_models = load('your_path/AA_20240105_All_VIF_models.joblib')


# Define a scaling function
def scale_prediction(prediction_obj, factor):
    # Copy object (avoid modifying original object)
    scaled = prediction_obj.__class__(prediction_obj.predicted_mean / factor,
                                    prediction_obj.var_pred_mean / (factor**2),
                                    prediction_obj.__dict__)
    return scaled


## Get FMI related models and parameters, predict FMI, and extract 95% confidence interval (CI) and prediction interval (PI)
FMI_forward_linear_model = AA_20240105_All_VIF_models['FMI_forward_linear_model']['Forward_linear_model']
FMI_selected_feature = AA_20240105_All_VIF_models['FMI_forward_linear_model']['Selected_feature']
print(FMI_selected_feature)
FMI_X_test_selected = X_test[list(FMI_selected_feature)]
FMI_X_test_selected = add_constant(FMI_X_test_selected)    ## Add intercept term
FMI_Linear_y_pred = FMI_forward_linear_model.get_prediction(FMI_X_test_selected)  # Predict new points and extract CI and PI


## Get A/G related models and parameters, predict A/G
A_G_forward_linear_model = AA_20240105_All_VIF_models['A_G_forward_linear_model']['Forward_linear_model']
A_G_selected_feature = AA_20240105_All_VIF_models['A_G_forward_linear_model']['Selected_feature']
A_G_X_test_selected = X_test[list(A_G_selected_feature)]
A_G_X_test_selected = add_constant(A_G_X_test_selected)     ## Add intercept term
A_G_Linear_y_pred = A_G_forward_linear_model.get_prediction(A_G_X_test_selected)  # Predict new points and extract CI and PI


## Get FM related models and parameters, predict FM
FM_forward_linear_model = AA_20240105_All_VIF_models['FM_forward_linear_model']['Forward_linear_model']
FM_selected_feature = AA_20240105_All_VIF_models['FM_forward_linear_model']['Selected_feature']
FM_X_test_selected = X_test[list(FM_selected_feature)]
FM_X_test_selected = add_constant(FM_X_test_selected)      ## Add intercept term
FM_Linear_y_pred = FM_forward_linear_model.get_prediction(FM_X_test_selected)  # Predict new points and extract CI and PI


## Get LM related models and parameters, predict LM
LM_forward_linear_model = AA_20240105_All_VIF_models['LM_forward_linear_model']['Forward_linear_model']
LM_selected_feature = AA_20240105_All_VIF_models['LM_forward_linear_model']['Selected_feature']
LM_X_test_selected = X_test[list(LM_selected_feature)]
LM_X_test_selected = add_constant(LM_X_test_selected)       ## Add intercept term
LM_Linear_y_pred = LM_forward_linear_model.get_prediction(LM_X_test_selected)  # Predict new points and extract CI and PI


## Get VATmass related models and parameters, predict VATmass
VAT_forward_linear_model = AA_20240105_All_VIF_models['VAT_forward_linear_model']['Forward_linear_model']
VAT_selected_feature = AA_20240105_All_VIF_models['VAT_forward_linear_model']['Selected_feature']
VAT_X_test_selected = X_test[list(VAT_selected_feature)]
VAT_X_test_selected = add_constant(VAT_X_test_selected)      ## Add intercept term
VAT_Linear_y_pred = VAT_forward_linear_model.get_prediction(VAT_X_test_selected)  # Predict new points and extract CI and PI


## Get Android related models and parameters, predict Android
Android_forward_linear_model = AA_20240105_All_VIF_models['Android_forward_linear_model']['Forward_linear_model']
Android_selected_feature = AA_20240105_All_VIF_models['Android_forward_linear_model']['Selected_feature']
Android_X_test_selected = X_test[list(Android_selected_feature)]
Android_X_test_selected = add_constant(Android_X_test_selected)        ## Add intercept term
Android_Linear_y_pred = Android_forward_linear_model.get_prediction(Android_X_test_selected)  # Predict new points and extract CI and PI


## Get Gynoid related models and parameters, predict Gynoid
Gynoid_forward_linear_model = AA_20240105_All_VIF_models['Gynoid_forward_linear_model']['Forward_linear_model']
Gynoid_selected_feature = AA_20240105_All_VIF_models['Gynoid_forward_linear_model']['Selected_feature']
Gynoid_X_test_selected = X_test[list(Gynoid_selected_feature)]
Gynoid_X_test_selected = add_constant(Gynoid_X_test_selected)          ## Add intercept term
Gynoid_Linear_y_pred = Gynoid_forward_linear_model.get_prediction(Gynoid_X_test_selected)  # Predict new points and extract CI and PI


## Get BFP related models and parameters, predict BFP
BFP_forward_linear_model = AA_20240105_All_VIF_models['BFP_forward_linear_model']['Forward_linear_model']
BFP_selected_feature = AA_20240105_All_VIF_models['BFP_forward_linear_model']['Selected_feature']
BFP_X_test_selected = X_test[list(BFP_selected_feature)]
BFP_X_test_selected = add_constant(BFP_X_test_selected)                ## Add intercept term
BFP_Linear_y_pred = BFP_forward_linear_model.get_prediction(BFP_X_test_selected)  # Predict new points and extract CI and PI
# print(BFP_Linear_y_pred)



## Loop to add variables
Y_list = []
predictions_list = []

Y_list.append(y_test['FM'])
Y_list.append(y_test['FMI'])
Y_list.append(y_test['LM'])
Y_list.append(y_test['Android'])
Y_list.append(y_test['Gynoid'])
Y_list.append(y_test['VATmass'])
Y_list.append(y_test['BFP']*100)
Y_list.append(y_test['A_G'])

predictions_list.append(FM_Linear_y_pred)
predictions_list.append(FMI_Linear_y_pred)
predictions_list.append(LM_Linear_y_pred)
predictions_list.append(Android_Linear_y_pred)
predictions_list.append(Gynoid_Linear_y_pred)
predictions_list.append(VAT_Linear_y_pred)
predictions_list.append(BFP_Linear_y_pred)
predictions_list.append(A_G_Linear_y_pred)

## Define plot title list as follows:
titles = ["FM", "FMI", "LM", "Android", "Gynoid", "VAT", "BFP", "AGFMR"]


### Define CCC calculation function
def calculate_ccc(y_true, y_pred):
    "Calculate Concordance Correlation Coefficient CCC"
    # Calculate Pearson correlation coefficient
    corr, _ = pearsonr(y_true, y_pred)

    ## Calculate means
    mean_true = np.mean(y_true)
    mean_pred = np.mean(y_pred)

    ## Calculate variances
    var_true = np.var(y_true)
    var_pred = np.var(y_pred)

    ## Calculate covariance (for denominator)
    covar = np.cov(y_true,y_pred)[0,1]

    ## Calculate CCC
    numerator = 2 * corr * np.sqrt(var_true) * np.sqrt(var_pred)
    denominator = var_true + var_pred + (mean_true - mean_pred) ** 2
    ccc = numerator / denominator

    return ccc


## Loop to plot each set of scatter plots and save
for i in range(8):
    ## Get current Y value and predicted value, R2
    Y = Y_list[i]
    predictions = predictions_list[i]

    ## Get confidence interval & prediction interval
    pred_summary = predictions.summary_frame(alpha = 0.05)
    y_pred = pred_summary["mean"]  ## Predicted mean

    ## Perform linear regression on Y and y_pred, calculate fitted regression line
    reg_model = sm.OLS(y_pred, sm.add_constant(Y)).fit()
    intercept, slope = reg_model.params  ## Calculate slope and intercept
    y_pred_fitted = slope * Y + intercept    ## Calculate fitted regression line
    # print(f"Slope: {slope}, Intercept: {intercept}")
    CCC = calculate_ccc(y_pred, y_pred_fitted)

    ## Calculate Residual Standard Error (RSE)
    n = len(Y)
    residuals = y_pred - y_pred_fitted
    RSE = np.sqrt(np.sum(residuals**2) / (n- 2))

    ## Calculate error margin for confidence interval
    t_crit = t.ppf(0.975, df = n-2)   ## 95% confidence interval
    conf_interval = t_crit * RSE / np.sqrt(n)

    ## Calculate confidence interval for fitted regression line
    conf_lower_fit = y_pred_fitted - conf_interval
    conf_upper_fit = y_pred_fitted + conf_interval

    ## Calculate prediction interval
    sigma_pred = np.std(y_pred - y_pred_fitted)   ## Standard error of prediction interval
    pred_upper_line = y_pred_fitted + t_crit * sigma_pred
    pred_lower_line = y_pred_fitted - t_crit * sigma_pred

    ## Bland-Altman calculation
    mean_values = (Y + y_pred) / 2  ## Mean of true and predicted values
    differences = y_pred - Y  ## Error (predicted - true)
    mean_diff = np.mean(differences)  ## Mean difference
    std_diff = np.std(differences)    ## Standard deviation of errors
    upper_limit = mean_diff + 1.96 * std_diff    ## 95% upper limit
    lower_limit = mean_diff - 1.96 * std_diff    ## 95% lower limit

    plt.rcParams['font.family'] = 'DejaVu Serif'      ## Set global font to "Times New Roman"
    ## Plot Predicted vs True graph (left column)
    # ax1 = axes[i,0]
    fig1, ax1 = plt.subplots(figsize = (6,4))
    ax1.scatter(Y, y_pred, color = '#1f77b4', alpha = 0.7, s = 15)   ## Observed data
    if intercept > 0 :
        equation = rf"$Y_{{pred}} = {intercept:.2f} + {slope:.2f}Y$"
    elif intercept < 0 :
        equation = rf"$Y_{{pre d}} = -{-intercept:.2f} + {slope:.2f}Y$"
    else :
        equation = rf"$Y_{{pre d}} = {slope:.2f}Y$"
    ax1.plot(Y, y_pred_fitted, label = equation, color = '#1f77b4', linewidth = 1)   ## Regression line
    ax1.fill_between(Y, conf_lower_fit, conf_upper_fit, color = '#1f77b4', alpha = 0.2, label = "95% Confidence Limits")   ##
    ax1.plot(Y, pred_upper_line, linestyle = ":", color = '#1f77b4', linewidth = 0.5, label = "95% Prediction Limits")
    ax1.plot(Y, pred_lower_line, linestyle = ":", color = '#1f77b4', linewidth = 0.5)
    ax1.plot(Y, Y, color = 'darkblue', linewidth = 0.8, label = "Y = X")
    ax1.set_xlabel(f"DAX True Values of {titles[i]} Equation ", fontsize=10, color = 'black')
    ax1.set_ylabel(f"Predicted Values of {titles[i]} Equation ", fontsize=10, color='black')
    ax1.legend(loc = 'upper left', fontsize = 10, markerscale = 1, labelspacing = 0.5, frameon = False)
    ax1.text(0.95, 0.05, f"$CCC = {CCC:.2f}$", transform = ax1.transAxes, fontsize = 10, color = "black", ha ="right", va = "bottom", fontfamily = 'DejaVu Serif')
    ax1.grid(True, linestyle = '--', linewidth = 0.5)
    plt.savefig(f"your_path/Pred_vs_True_{titles[i]}.png", dpi = 600, bbox_inches = 'tight')   ## Save image
    plt.close(fig1)   ## Close image to avoid memory leaks

    ## Plot Bland-Altman graph (right column)
    # ax2 = axes[i, 1]
    fig2, ax2 = plt.subplots(figsize = (6,4))
    ax2.scatter(mean_values, differences, color = '#1f77b4', alpha = 0.7, s = 15)
    ax2.axhline(mean_diff, color = '#1f77b4', linewidth = 1, label = 'Mean Difference')
    ax2.axhline(upper_limit, color = 'black', linestyle = '--', linewidth = 1, label = '95% Upper Limit')
    ax2.axhline(lower_limit, color = 'black', linestyle = '--', linewidth = 1, label = '95% Lower Limit')
    ax2.set_xlabel( f"Mean of {titles[i]} Equation", fontsize=10, color='black')
    ax2.set_ylabel( f"Difference of {titles[i]} Equation", fontsize=10, color='black')
    ax2.grid(True, linestyle = '--', linewidth = 0.5)
    # ax2.legend(loc = 'lower right', fontsize = 7, markerscale = 0.7, labelspacing = 0.5, frameon = False)
    ax2.grid(False)

    print(titles[i])
    print("95% Upper Limit:", mean_diff, "+", 1.96 * std_diff)
    print("95% Lower Limit:", mean_diff, "-", 1.96 * std_diff)
    plt.savefig(f"your_path/Bland_Altman{titles[i]}.png", dpi = 600, bbox_inches = 'tight')   ## Save image
    plt.close(fig2)    ## Close image to avoid memory leaks
    

################# Plot updated scatter regression plot and Bland-Altman analysis plot for VATmass model ########################

## Get VATmass interaction model
VAT_forward_linear_model = AA_20240105_All_VIF_models['VAT_inetraction_FSLR']['Forward_linear_model']
VAT_selected_feature = AA_20240105_All_VIF_models['VAT_inetraction_FSLR']['Selected_feature']

## Prepare test set data
VAT_X_test_selected = X_test[list(VAT_selected_feature)]
VAT_X_test_selected = add_constant(VAT_X_test_selected)
VAT_y_test = y_test['VATmass']

VAT_Linear_y_pred = VAT_forward_linear_model.get_prediction(VAT_X_test_selected)

### Define function for CCC calculation
def calculate_ccc(y_true, y_pred):
    "Calculate Concordance Correlation Coefficient CCC"
    # Calculate Pearson correlation coefficient
    corr, _ = pearsonr(y_true, y_pred)

    ## Calculate means
    mean_true = np.mean(y_true)
    mean_pred = np.mean(y_pred)

    ## Calculate variances
    var_true = np.var(y_true)
    var_pred = np.var(y_pred)

    ## Calculate covariance (for denominator)
    covar = np.cov(y_true,y_pred)[0,1]

    ## Calculate CCC
    numerator = 2 * corr * np.sqrt(var_true) * np.sqrt(var_pred)
    denominator = var_true + var_pred + (mean_true - mean_pred) ** 2
    ccc = numerator / denominator

    return ccc


## Get confidence interval & prediction interval
pred_summary = VAT_Linear_y_pred.summary_frame(alpha = 0.05)
y_pred = pred_summary["mean"]  ## Prediction mean

Y = VAT_y_test

# Add before regression calculation:
print("=== Index Check ===")
print(f"Y index type: {type(Y.index)}, length: {len(Y)}")
print(f"y_pred index type: {type(y_pred.index)}, length: {len(y_pred)}")
print(f"Index consistency: {Y.index.equals(y_pred.index)}")

# If indices are inconsistent, reset to numeric indices
if not Y.index.equals(y_pred.index):
    print("Indices inconsistent, resetting...")
    Y = Y.reset_index(drop=True)
    y_pred = y_pred.reset_index(drop=True)


## Perform linear regression on Y and y_pred, calculate fitted regression line
reg_model = sm.OLS(y_pred, sm.add_constant(Y)).fit()
intercept, slope = reg_model.params  ##   Calculate slope and intercept
y_pred_fitted = slope * Y + intercept    ##  Calculate fitted regression line
# print(f"Slope: {slope}, Intercept: {intercept}")
CCC = calculate_ccc(y_pred, y_pred_fitted)


##  Calculate Residual Standard Error
n = len(Y)
residuals = y_pred - y_pred_fitted
RSE = np.sqrt(np.sum(residuals**2) / (n- 2))

##  Calculate error margin for confidence interval
t_crit = t.ppf(0.975, df = n-2)   ##  95% confidence interval
conf_interval = t_crit * RSE / np.sqrt(n)

## Calculate confidence interval for fitted regression line
conf_lower_fit = y_pred_fitted - conf_interval
conf_upper_fit = y_pred_fitted + conf_interval

##  Calculate prediction interval
sigma_pred = np.std(y_pred - y_pred_fitted)   ## Standard error for prediction interval
pred_upper_line = y_pred_fitted + t_crit * sigma_pred
pred_lower_line = y_pred_fitted - t_crit * sigma_pred

## Bland-Altman calculation
mean_values = (Y + y_pred) / 2  ## Mean of true and predicted values
differences = y_pred - Y  ## Error (predicted - true)
mean_diff = np.mean(differences)  ## Mean difference
std_diff = np.std(differences)    ## Standard deviation of errors
upper_limit = mean_diff + 1.96 * std_diff    ## 95% upper limit
lower_limit = mean_diff - 1.96 * std_diff    ## 95% lower limit

plt.rcParams['font.family'] = 'DejaVu Serif'      ##  Set global font to "Times New Roman"
## Plot predicted VS true values (left column)
# ax1 = axes[i,0]
fig1, ax1 = plt.subplots(figsize = (6,4))
ax1.scatter(Y, y_pred, color = '#1f77b4', alpha = 0.7, s = 15)   ## Observed data
if intercept > 0 :
    equation = rf"$Y_{{pred}} = {intercept:.2f} + {slope:.2f}Y$"
elif intercept < 0 :
    equation = rf"$Y_{{pre d}} = -{-intercept:.2f} + {slope:.2f}Y$"
else :
    equation = rf"$Y_{{pre d}} = {slope:.2f}Y$"
ax1.plot(Y, y_pred_fitted, label = equation, color = '#1f77b4', linewidth = 1)   ## Regression line
ax1.fill_between(Y, conf_lower_fit, conf_upper_fit, color = '#1f77b4', alpha = 0.2, label = "95% Confidence Limits")   ##
ax1.plot(Y, pred_upper_line, linestyle = ":", color = '#1f77b4', linewidth = 0.5, label = "95% Prediction Limits")
ax1.plot(Y, pred_lower_line, linestyle = ":", color = '#1f77b4', linewidth = 0.5)
ax1.plot(Y, Y, color = 'darkblue', linewidth = 0.8, label = "Y = X")
ax1.set_xlabel(f"DAX True Values of {titles[i]} Equation ", fontsize=10, color = 'black')
ax1.set_ylabel(f"Predicted Values of {titles[i]} Equation ", fontsize=10, color='black')
ax1.legend(loc = 'upper left', fontsize = 10, markerscale = 1, labelspacing = 0.5, frameon = False)
ax1.text(0.95, 0.05, f"$CCC = {CCC:.2f}$", transform = ax1.transAxes, fontsize = 10, color = "black", ha ="right", va = "bottom", fontfamily = 'DejaVu Serif')

ax1.grid(True, linestyle = '--', linewidth = 0.5)
plt.savefig(f"YourPath/RegressionPlot_BlandAltman2/Pred_vs_True_VAT2.png", dpi = 600, bbox_inches = 'tight')   ## Save image
plt.close(fig1)   ##  Close image to avoid memory leaks

## Plot Bland-Altman diagram (right column)
# ax2 = axes[i, 1]
fig2, ax2 = plt.subplots(figsize = (6,4))
ax2.scatter(mean_values, differences, color = '#1f77b4', alpha = 0.7, s = 15)
ax2.axhline(mean_diff, color = '#1f77b4', linewidth = 1, label = 'Mean Difference')
ax2.axhline(upper_limit, color = 'black', linestyle = '--', linewidth = 1, label = '95% Upper Limit')
ax2.axhline(lower_limit, color = 'black', linestyle = '--', linewidth = 1, label = '95% Lower Limit')
ax2.set_xlabel( f"Mean of {titles[i]} Equation", fontsize=10, color='black')
ax2.set_ylabel( f"Difference of {titles[i]} Equation", fontsize=10, color='black')
ax2.grid(True, linestyle = '--', linewidth = 0.5)
# ax2.legend(loc = 'lower right', fontsize = 7, markerscale = 0.7, labelspacing = 0.5, frameon = False)
ax2.grid(False)

print("VAT:")
print("95% Upper Limit:", mean_diff, "+", 1.96 * std_diff)
print("95% Lower Limit:", mean_diff, "-", 1.96 * std_diff)
plt.savefig(f"YourPath/RegressionPlot_BlandAltman2/Bland_AltmanVAT2.png", dpi = 600, bbox_inches = 'tight')   ##  Save image
plt.close(fig2)    ##  Close image to avoid memory leaks
