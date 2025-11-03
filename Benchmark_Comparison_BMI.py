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
from sklearn.model_selection import cross_validate    ## 导入交叉验证包
from sklearn.model_selection import  cross_val_score   ## 导入交叉验证包
from sklearn.model_selection import KFold   ## 导入KFold分离器
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
from sklearn.svm import SVR      ## 导入支持向量机包
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

## 导入数据
X_train = pd.read_csv("C:/Users/zjl__/Desktop/output_zhaojialu/output/SummaryStatistic/X_train_All3D_NoScaled.csv",sep=",")
X_train.set_index(X_train.columns[0], inplace= True)  ## 将数据框的第一列作为数据框的行名
name = X_train.columns.tolist()    ## 提取数据框的列名
name = [item.replace('人体外观测量.三维人体扫描分析系统.', '') for item in name]    ## 删除列名中“人体外观测量-三维人体扫描分析系统:”的部分
name = [item.replace('.cm.', '') for item in name]    ## 删除列名中“.cm.”的部分
X_train.columns = name   ## 替换数据框的列名为最简单名字
# print(X_train)
X_train = X_train.drop(columns = ['height', 'Waist_To_Hip_Ratio.x.100.'])
print(X_train)

X_test = pd.read_csv("C:/Users/zjl__/Desktop/output_zhaojialu/output/SummaryStatistic/X_test_All3D_NoScaled.csv",sep=",")
X_test.set_index(X_test.columns[0], inplace= True)  ## 将数据框的第一列作为数据框的行名
X_test.columns = name   ## 替换数据框的列名为最简单名字
X_test = X_test.drop(columns = ['height', 'Waist_To_Hip_Ratio.x.100.'])
# print(X_test)

y_train = pd.read_csv("C:/Users/zjl__/Desktop/output_zhaojialu/output/SummaryStatistic/y_train_All3D_NoScaled.csv",sep=",")
y_train.set_index(y_train.columns[0], inplace= True)  ## 将数据框的第一列作为数据框的行名

y_test = pd.read_csv("C:/Users/zjl__/Desktop/output_zhaojialu/output/SummaryStatistic/y_test_All3D_NoScaled.csv",sep=",")
y_test.set_index(y_test.columns[0], inplace= True)  ## 将数据框的第一列作为数据框的行名



def Index_Extract(dataframe):
    # 提取 腰围WC、臀围HC、颈围NC、腰臀比WHR
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
    # 计算 BMI
    height = dataframe["身体高度"] / 100
    weight = dataframe["weight"]
    BMI = (weight) / ((height ) ** 2)
    BMI.name = "BMI"
    BMI = BMI.to_frame()
    # 计算腰高比 WHtR, ABSI
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
    ## 获取回归系数、标准误差、置信区间
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

        ## 添加该模型的所有结果到结果列表
    for var, data in corr_data.items():
            corr_results.append(data)

    df = pd.DataFrame(corr_results)
    print(df)
    return df


"""
def Index_Comparison(Y, dataframe):
    corr_results = []
    ## 获取回归系数、标准误差、置信区间
    corr_data = {var: {'Variable':var, 'Height': 0, 'Weight': 0, 'BMI': 0, 'WC': 0, 'HC': 0, 'WHR': 0, 'WHtR': 0, 'NC': 0, 'ABSI': 0}
                  for var in Y.columns}
    for column_name in Y.columns:
        column_data = Y[column_name]
        # corr_height, p_height = pearsonr(column_data, dataframe['height'])
        # corr_weight, p_weight = pearsonr(column_data, dataframe['weight'])
        corr_BMI, p_BMI = spearmanr(column_data, dataframe['BMI'])
        corr_WC, p_WC = spearmanr(column_data, dataframe['WC'])
        corr_HC, p_HC = spearmanr(column_data, dataframe['HC'])
        corr_WHR, p_WHR = spearmanr(column_data, dataframe['WHR'])
        corr_WHtR, p_WHtR = spearmanr(column_data, dataframe['WHtR'])
        corr_NC, p_NC = spearmanr(column_data, dataframe['NC'])
        corr_ABSI, p_ABSI = spearmanr(column_data, dataframe['ABSI'])

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

        ## 添加该模型的所有结果到结果列表
    for var, data in corr_data.items():
            corr_results.append(data)

    df = pd.DataFrame(corr_results)
    print(df)
    return df
"""

## 将所有结果整理为一个DataFrame
Test_IndexComparison = Index_Comparison(y_test, CommonAnthroIndex_test)


## 加载之前用训练集训练的多个模型及参数
AA_20240105_All_VIF_models = load('C:/Users/zjl__/Desktop/output_zhaojialu/output/ForwardRegression2/AA_20240105_All_VIF_models.joblib')

## 获取 FMI 的相关模型及参数, 并预测 FMI，并提取95%置信区间（CI）和预测区间（PI）
FMI_forward_linear_model = AA_20240105_All_VIF_models['FMI_forward_linear_model']['Forward_linear_model']
FMI_selected_feature = AA_20240105_All_VIF_models['FMI_forward_linear_model']['Selected_feature']
FMI_X_test_selected = X_test[list(FMI_selected_feature)]
FMI_X_test_selected = add_constant(FMI_X_test_selected)    ## 添加截距项
FMI_Linear_y_pred = FMI_forward_linear_model.predict(FMI_X_test_selected)  # 预测新的点
corr_FMI, p_FMI = pearsonr(FMI_Linear_y_pred, y_test['FMI'])

## 获取 A/G 的相关模型及参数, 并预测 A/G
A_G_forward_linear_model = AA_20240105_All_VIF_models['A_G_forward_linear_model']['Forward_linear_model']
A_G_selected_feature = AA_20240105_All_VIF_models['A_G_forward_linear_model']['Selected_feature']
A_G_X_test_selected = X_test[list(A_G_selected_feature)]
A_G_X_test_selected = add_constant(A_G_X_test_selected)     ## 添加截距项
A_G_Linear_y_pred = A_G_forward_linear_model.predict(A_G_X_test_selected)  # 预测新的点
corr_A_G, p_A_G = pearsonr(A_G_Linear_y_pred, y_test['A_G'])

## 获取 FM 的相关模型及参数, 并预测 FM
FM_forward_linear_model = AA_20240105_All_VIF_models['FM_forward_linear_model']['Forward_linear_model']
FM_selected_feature = AA_20240105_All_VIF_models['FM_forward_linear_model']['Selected_feature']
FM_X_test_selected = X_test[list(FM_selected_feature)]
FM_X_test_selected = add_constant(FM_X_test_selected)      ## 添加截距项
FM_Linear_y_pred = FM_forward_linear_model.predict(FM_X_test_selected)  # 预测新的点
corr_FM, p_FM = pearsonr(FM_Linear_y_pred, y_test['FM'])

## 获取 LM 的相关模型及参数, 并预测 LM
LM_forward_linear_model = AA_20240105_All_VIF_models['LM_forward_linear_model']['Forward_linear_model']
LM_selected_feature = AA_20240105_All_VIF_models['LM_forward_linear_model']['Selected_feature']
LM_X_test_selected = X_test[list(LM_selected_feature)]
LM_X_test_selected = add_constant(LM_X_test_selected)       ## 添加截距项
LM_Linear_y_pred = LM_forward_linear_model.predict(LM_X_test_selected)  # 预测新的点
corr_LM, p_LM = pearsonr(LM_Linear_y_pred, y_test['LM'])

## 获取 VATmass 的相关模型及参数, 并预测 VATmass
VAT_forward_linear_model = AA_20240105_All_VIF_models['VAT_forward_linear_model']['Forward_linear_model']
VAT_selected_feature = AA_20240105_All_VIF_models['VAT_forward_linear_model']['Selected_feature']
VAT_X_test_selected = X_test[list(VAT_selected_feature)]
VAT_X_test_selected = add_constant(VAT_X_test_selected)      ## 添加截距项
VAT_Linear_y_pred = VAT_forward_linear_model.predict(VAT_X_test_selected)  # 预测新的点
corr_VAT, p_VAT = pearsonr(VAT_Linear_y_pred, y_test['VATmass'])

## 获取  Android 的相关模型及参数, 并预测 Android
Android_forward_linear_model = AA_20240105_All_VIF_models['Android_forward_linear_model']['Forward_linear_model']
Android_selected_feature = AA_20240105_All_VIF_models['Android_forward_linear_model']['Selected_feature']
Android_X_test_selected = X_test[list(Android_selected_feature)]
Android_X_test_selected = add_constant(Android_X_test_selected)        ## 添加截距项
Android_Linear_y_pred = Android_forward_linear_model.predict(Android_X_test_selected)  # 预测新的点
corr_Android, p_Android = pearsonr(Android_Linear_y_pred, y_test['Android'])

## 获取  Gynoid 的相关模型及参数, 并预测 Gynoid
Gynoid_forward_linear_model = AA_20240105_All_VIF_models['Gynoid_forward_linear_model']['Forward_linear_model']
Gynoid_selected_feature = AA_20240105_All_VIF_models['Gynoid_forward_linear_model']['Selected_feature']
Gynoid_X_test_selected = X_test[list(Gynoid_selected_feature)]
Gynoid_X_test_selected = add_constant(Gynoid_X_test_selected)          ## 添加截距项
Gynoid_Linear_y_pred = Gynoid_forward_linear_model.predict(Gynoid_X_test_selected)  # 预测新的点
corr_Gynoid, p_Gynoid = pearsonr(Gynoid_Linear_y_pred, y_test['Gynoid'])

## 获取 BFP 的相关模型及参数, 并预测 BFP
BFP_forward_linear_model = AA_20240105_All_VIF_models['BFP_forward_linear_model']['Forward_linear_model']
BFP_selected_feature = AA_20240105_All_VIF_models['BFP_forward_linear_model']['Selected_feature']
BFP_X_test_selected = X_test[list(BFP_selected_feature)]
BFP_X_test_selected = add_constant(BFP_X_test_selected)                ## 添加截距项
BFP_Linear_y_pred = BFP_forward_linear_model.predict(BFP_X_test_selected)  # 预测新的点
corr_BFP, p_BFP = pearsonr(BFP_Linear_y_pred, y_test['BFP'])

# ['Android', 'Gynoid', 'A_G', 'FM', 'FMI', 'BFP', 'LM', 'VATmass']

Test_IndexComparison['Ours'] = [corr_Android, corr_Gynoid, corr_A_G, corr_FM, corr_FMI, corr_BFP, corr_LM, corr_VAT]
Test_IndexComparison = Test_IndexComparison.set_index('Variable')

## 保存为CSV文件
Test_IndexComparison.to_csv("C:/Users/zjl__/Desktop/output_zhaojialu/output/IndexComparison/IndexComparison_Pearson2.csv", index = True)

print(Test_IndexComparison.T)

## 绘制热图
plt.figure(figsize = (15, 5))   ## 设置图像大小


g = sns.clustermap(Test_IndexComparison, cmap = 'coolwarm', annot = True, fmt = ".2f", figsize = (10, 6), center = 0)

## 调整行标签和列标签的字号和加粗
plt.setp(g.ax_heatmap.get_xticklabels(), fontsize = 12, fontweight = 'bold')
plt.setp(g.ax_heatmap.get_yticklabels(), fontsize = 12, fontweight = 'bold')

## 获取聚类结果
row_cluster_order = g.dendrogram_row.reordered_ind
col_cluster_order = g.dendrogram_col.reordered_ind

print("Row cluster order: ", row_cluster_order)
print("Column cluster order: ", col_cluster_order)

## 将聚类顺序进行存储
row_df = pd.DataFrame(row_cluster_order, columns= ["Row Cluster Order"])
col_df = pd.DataFrame(col_cluster_order, columns = ["Column Cluster Order"])

## 保存为CSV文件
row_df.to_csv("C:/Users/zjl__/Desktop/output_zhaojialu/output/IndexComparison/Index_Cluster_Pearson2.csv")
col_df.to_csv("C:/Users/zjl__/Desktop/output_zhaojialu/output/IndexComparison/DXA_Cluster_Pearson2.csv")

## 保存整张图片
plt.savefig("C:/Users/zjl__/Desktop/output_zhaojialu/output/IndexComparison/IndexComparison_Cluster_Pearson2.png", dpi = 300, bbox_inches = "tight")
plt.show()

