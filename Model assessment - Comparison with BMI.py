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
from scipy.stats import spearmanr
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
from joblib import dump
from joblib import load
import seaborn as sns
from scipy.stats import t
from matplotlib import font_manager

import warnings
warnings.filterwarnings("ignore")

## Import data
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



def Index_Extract(dataframe):
    # Extract waist circumference WC, hip circumference HC, neck circumference NC, waist-hip ratio WHR
    WC = dataframe["标准站姿腰围"] / 100
    WC.name = "WC"
    WC = WC.to_frame()
    HC = dataframe["臀部围度"] / 100
    HC.name = "HC"
    HC = HC.to_frame()
    NC = dataframe["中间的脖子围度"] / 100
    NC.name = "NC"
    NC = NC.to_frame()
    WHR = dataframe["标准站姿腰围"] / dataframe["臀部围度"]
    WHR.name = "WHR"
    WHR = WHR.to_frame()
    # Calculate BMI
    height = dataframe["身体高度"] / 100
    weight = dataframe["weight"]
    BMI = (weight) / ((height ) ** 2)
    BMI.name = "BMI"
    BMI = BMI.to_frame()
    # Calculate waist-height ratio WHtR, ABSI
    WHtR = ( dataframe["标准站姿腰围"] / 100) / height
    WHtR.name = "WHtR"
    ABSI = (dataframe["标准站姿腰围"] / 100 ) / (((weight) / ((height ) ** 2)) ** (2 / 3) * (height ) ** (1 / 2))
    ABSI.name = "ABSI"
    ABSI = ABSI.to_frame()

    data = pd.concat([BMI, height, weight, WC, HC, NC, WHR, WHtR, ABSI], axis = 1)
    data.index = dataframe.iloc[:,0]
    print(data)
    return data

CommonAnthroIndex_train = Index_Extract(X_train)
CommonAnthroIndex_test = Index_Extract(X_test)


def Index_Comparison(Y, dataframe):
    corr_results = []
    ## Get regression coefficients, standard errors, confidence intervals
    name = ['Android', 'Gynoid', 'A_G', 'FM', 'FMI', 'BFP', 'LM', 'VATmass']
    corr_data = {var: {'Variable':var, 'Height': 0, 'Weight': 0, 'BMI': 0, 'WC': 0, 'HC': 0, 'WHR': 0, 'WHtR': 0, 'NC': 0, 'ABSI': 0}
                  for var in name}
    for column_name in name:
        column_data = Y[column_name]
        # corr_height, p_height = pearsonr(column_data, dataframe['height'])
        # corr_weight, p_weight = pearsonr(column_data, dataframe['weight'])
        corr_BMI, p_BMI = pearsonr(column_data, dataframe['BMI'])
        corr_WC, p_WC = pearsonr(column_data, dataframe['WC'])
        corr_HC, p_HC = pearsonr(column_data, dataframe['HC'])
        corr_WHR, p_WHR = pearsonr(column_data, dataframe['WHR'])
        corr_WHtR, p_WHtR = pearsonr(column_data, dataframe['WHtR'])
        corr_NC, p_NC = pearsonr(column_data, dataframe['NC'])
        corr_ABSI, p_ABSI = pearsonr(column_data, dataframe['ABSI'])

        corr_data[column_name] = {
            'Variable': column_name,
            # 'Height': corr_height,
            # 'Weight': corr_weight,
            'BMI': corr_BMI,
            'WC': corr_WC,
            'HC': corr_HC,
            'WHR': corr_WHR,
            'WHtR': corr_WHtR,
            'ABSI': corr_ABSI,
            'NC': corr_NC
        }

        ## Add all results of this model to result list
    for var, data in corr_data.items():
            corr_results.append(data)

    df = pd.DataFrame(corr_results)
    print(df)
    return df


## Organize all results into a DataFrame
Test_IndexComparison = Index_Comparison(y_test, CommonAnthroIndex_test)


## Load previously trained models and parameters using training set
AA_20240105_All_VIF_models = load('your_path/AA_20240105_All_VIF_models.joblib')

## Get FMI related models and parameters, predict FMI, and extract 95% confidence interval (CI) and prediction interval (PI)
FMI_forward_linear_model = AA_20240105_All_VIF_models['FMI_forward_linear_model']['Forward_linear_model']
FMI_selected_feature = AA_20240105_All_VIF_models['FMI_forward_linear_model']['Selected_feature']
FMI_X_test_selected = X_test[list(FMI_selected_feature)]
FMI_X_test_selected = add_constant(FMI_X_test_selected)    ## Add intercept term
FMI_Linear_y_pred = FMI_forward_linear_model.predict(FMI_X_test_selected)  # Predict new points
corr_FMI, p_FMI = pearsonr(FMI_Linear_y_pred, y_test['FMI'])

## Get A/G related models and parameters, predict A/G
A_G_forward_linear_model = AA_20240105_All_VIF_models['A_G_forward_linear_model']['Forward_linear_model']
A_G_selected_feature = AA_20240105_All_VIF_models['A_G_forward_linear_model']['Selected_feature']
A_G_X_test_selected = X_test[list(A_G_selected_feature)]
A_G_X_test_selected = add_constant(A_G_X_test_selected)     ## Add intercept term
A_G_Linear_y_pred = A_G_forward_linear_model.predict(A_G_X_test_selected)  # Predict new points
corr_A_G, p_A_G = pearsonr(A_G_Linear_y_pred, y_test['A_G'])

## Get FM related models and parameters, predict FM
FM_forward_linear_model = AA_20240105_All_VIF_models['FM_forward_linear_model']['Forward_linear_model']
FM_selected_feature = AA_20240105_All_VIF_models['FM_forward_linear_model']['Selected_feature']
FM_X_test_selected = X_test[list(FM_selected_feature)]
FM_X_test_selected = add_constant(FM_X_test_selected)      ## Add intercept term
FM_Linear_y_pred = FM_forward_linear_model.predict(FM_X_test_selected)  # Predict new points
corr_FM, p_FM = pearsonr(FM_Linear_y_pred, y_test['FM'])

## Get LM related models and parameters, predict LM
LM_forward_linear_model = AA_20240105_All_VIF_models['LM_forward_linear_model']['Forward_linear_model']
LM_selected_feature = AA_20240105_All_VIF_models['LM_forward_linear_model']['Selected_feature']
LM_X_test_selected = X_test[list(LM_selected_feature)]
LM_X_test_selected = add_constant(LM_X_test_selected)       ## Add intercept term
LM_Linear_y_pred = LM_forward_linear_model.predict(LM_X_test_selected)  # Predict new points
corr_LM, p_LM = pearsonr(LM_Linear_y_pred, y_test['LM'])

## Get VATmass related models and parameters, predict VATmass
# VAT_forward_linear_model = AA_20240105_All_VIF_models['VAT_forward_linear_model']['Forward_linear_model']
# VAT_selected_feature = AA_20240105_All_VIF_models['VAT_forward_linear_model']['Selected_feature']
# VAT_X_test_selected = X_test[list(VAT_selected_feature)]
# VAT_X_test_selected = add_constant(VAT_X_test_selected)      ## Add intercept term
# VAT_Linear_y_pred = VAT_forward_linear_model.predict(VAT_X_test_selected)  # Predict new points
# corr_VAT, p_VAT = pearsonr(VAT_Linear_y_pred, y_test['VATmass'])

VAT_forward_linear_model = AA_20240105_All_VIF_models['VAT_inetraction_FSLR']['Forward_linear_model']
VAT_selected_feature = AA_20240105_All_VIF_models['VAT_inetraction_FSLR']['Selected_feature']
VAT_X_test_selected = X_test[list(VAT_selected_feature)]
VAT_X_test_selected = add_constant(VAT_X_test_selected)      ## 添加截距项
VAT_Linear_y_pred = VAT_forward_linear_model.predict(VAT_X_test_selected)  # 预测新的点
corr_VAT, p_VAT = pearsonr(VAT_Linear_y_pred, y_test['VATmass'])

## Get Android related models and parameters, predict Android
Android_forward_linear_model = AA_20240105_All_VIF_models['Android_forward_linear_model']['Forward_linear_model']
Android_selected_feature = AA_20240105_All_VIF_models['Android_forward_linear_model']['Selected_feature']
Android_X_test_selected = X_test[list(Android_selected_feature)]
Android_X_test_selected = add_constant(Android_X_test_selected)        ## Add intercept term
Android_Linear_y_pred = Android_forward_linear_model.predict(Android_X_test_selected)  # Predict new points
corr_Android, p_Android = pearsonr(Android_Linear_y_pred, y_test['Android'])

## Get Gynoid related models and parameters, predict Gynoid
Gynoid_forward_linear_model = AA_20240105_All_VIF_models['Gynoid_forward_linear_model']['Forward_linear_model']
Gynoid_selected_feature = AA_20240105_All_VIF_models['Gynoid_forward_linear_model']['Selected_feature']
Gynoid_X_test_selected = X_test[list(Gynoid_selected_feature)]
Gynoid_X_test_selected = add_constant(Gynoid_X_test_selected)          ## Add intercept term
Gynoid_Linear_y_pred = Gynoid_forward_linear_model.predict(Gynoid_X_test_selected)  # Predict new points
corr_Gynoid, p_Gynoid = pearsonr(Gynoid_Linear_y_pred, y_test['Gynoid'])

## Get BFP related models and parameters, predict BFP
BFP_forward_linear_model = AA_20240105_All_VIF_models['BFP_forward_linear_model']['Forward_linear_model']
BFP_selected_feature = AA_20240105_All_VIF_models['BFP_forward_linear_model']['Selected_feature']
BFP_X_test_selected = X_test[list(BFP_selected_feature)]
BFP_X_test_selected = add_constant(BFP_X_test_selected)                ## Add intercept term
BFP_Linear_y_pred = BFP_forward_linear_model.predict(BFP_X_test_selected)  # Predict new points
corr_BFP, p_BFP = pearsonr(BFP_Linear_y_pred, y_test['BFP'])

# ['Android', 'Gynoid', 'A_G', 'FM', 'FMI', 'BFP', 'LM', 'VATmass']

Test_IndexComparison['Ours'] = [corr_Android, corr_Gynoid, corr_A_G, corr_FM, corr_FMI, corr_BFP, corr_LM, corr_VAT]
Test_IndexComparison = Test_IndexComparison.set_index('Variable')

## Save as CSV file
Test_IndexComparison.to_csv("your_path/IndexComparison_Pearson2.csv", index = True)

print(Test_IndexComparison.T)

## Plot heatmap
plt.figure(figsize = (15, 5))   ## Set image size


g = sns.clustermap(Test_IndexComparison, cmap = 'coolwarm', annot = True, fmt = ".2f", figsize = (10, 6), center = 0)

## Adjust row and column label font size and boldness
plt.setp(g.ax_heatmap.get_xticklabels(), fontsize = 12, fontweight = 'bold')
plt.setp(g.ax_heatmap.get_yticklabels(), fontsize = 12, fontweight = 'bold')

## Get clustering results
row_cluster_order = g.dendrogram_row.reordered_ind
col_cluster_order = g.dendrogram_col.reordered_ind

print("Row cluster order: ", row_cluster_order)
print("Column cluster order: ", col_cluster_order)

## Store clustering order
row_df = pd.DataFrame(row_cluster_order, columns= ["Row Cluster Order"])
col_df = pd.DataFrame(col_cluster_order, columns = ["Column Cluster Order"])

## Save as CSV file
row_df.to_csv("your_path/Index_Cluster_Pearson2.csv")
col_df.to_csv("your_path/DXA_Cluster_Pearson2.csv")

## Save entire image
plt.savefig("your_path/IndexComparison_Cluster_Pearson2.png", dpi = 300, bbox_inches = "tight")
plt.show()
