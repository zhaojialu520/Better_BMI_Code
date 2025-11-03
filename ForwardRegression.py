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
from sklearn import linear_model
import matplotlib.pyplot as plt
from numpy import genfromtxt
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.svm import SVR      ## 导入支持向量机包
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

## 导入数据
# AA_20240105_All3D = pd.read_csv("/public/sandbox/workdir/zhaojialu/Documents_About_Data/AA_20240105_without_na3D.txt",sep="\t")
# AA_20240105_Female3D = pd.read_csv("/public/sandbox/workdir/zhaojialu/Documents_About_Data/AA_20240105_female3D_NoNA.txt",sep="\t")
# AA_20240105_Male3D = pd.read_csv("/public/sandbox/workdir/zhaojialu/Documents_About_Data/AA_20240105_male3D_NoNA.txt",sep="\t")

# 将自变量列名进行替换
# name = AA_20240105_All3D.columns.tolist()
# print("name如下所示：",name)
# name = [item.replace('人体外观测量-三维人体扫描分析系统:', '') for item in name]
# AA_20240105_All3D.columns = name

# Labels = AA_20240105_All3D[['FMI', 'A_G', 'FM', 'LM', 'VATmass', 'Android', 'Gynoid', 'BFP']]
# X = AA_20240105_All3D.drop(columns=["FMI", "A_G", "FM", "LM", "VATmass", "Android", "Gynoid", "BFP"], axis=1)

## 按照8：1：1划分数据集
# X_train, X_temp, y_train, y_temp = train_test_split(X, Labels, test_size= 0.2, random_state= 42)
# X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size= 0.5, random_state= 42)

## 导入数据
X_train = pd.read_csv("C:/Users/zjl__/Desktop/output_zhaojialu/output/SummaryStatistic/X_train_All3D_NoScaled.csv",sep=",")
X_train.set_index(X_train.columns[0], inplace= True)  ## 将数据框的第一列作为数据框的行名
name = X_train.columns.tolist()    ## 提取数据框的列名
name = [item.replace('人体外观测量.三维人体扫描分析系统.', '') for item in name]    ## 删除列名中“人体外观测量-三维人体扫描分析系统:”的部分
name = [item.replace('.cm.', '') for item in name]
X_train.columns = name   ## 替换数据框的列名为最简单名字
print(X_train)
X_train = X_train.drop(columns = ['height', 'Waist_To_Hip_Ratio.x.100.'])

X_test = pd.read_csv("C:/Users/zjl__/Desktop/output_zhaojialu/output/SummaryStatistic/X_test_All3D_NoScaled.csv",sep=",")
X_test.set_index(X_test.columns[0], inplace= True)  ## 将数据框的第一列作为数据框的行名
X_test.columns = name   ## 替换数据框的列名为最简单名字
X_test = X_test.drop(columns = ['height', 'Waist_To_Hip_Ratio.x.100.'])

y_train = pd.read_csv("C:/Users/zjl__/Desktop/output_zhaojialu/output/SummaryStatistic/y_train_All3D_NoScaled.csv",sep=",")
y_train.set_index(y_train.columns[0], inplace= True)  ## 将数据框的第一列作为数据框的行名


y_test = pd.read_csv("C:/Users/zjl__/Desktop/output_zhaojialu/output/SummaryStatistic/y_test_All3D_NoScaled.csv",sep=",")
y_test.set_index(y_test.columns[0], inplace= True)  ## 将数据框的第一列作为数据框的行名

## 对训练集进行逐步线性回归，得到FMI，A_G, FM, LM, VATmass, ANdroid, Gynoid, BFP
def forward_stepwise_regression_bic(data, target):
    ## 初始化数据
    initial_features = data.columns.tolist()  ## 所有特征
    selected_features = []   ## 存储选中的特征
    remaining_features = initial_features.copy()  ## 存储剩余特征
    best_bic = float('inf')   ## 初始化最小BIC值
    final_model = None  ## 最终的模型

    while remaining_features:
        bic_with_candidates = []

        ## 遍历每个未选中的特征，逐个加入
        for feature in remaining_features:
            ## 组合已选特征和当前特征
            candidate_features = selected_features + [feature]
            X = data[candidate_features]
            X = add_constant(X)   ## 添加常数项
            y = target

            ## 拟合回归模型
            model = sm.OLS(y, X).fit()

            ## 记录每个模型的BIC值
            bic_with_candidates.append((feature, model.bic))

        ## 按照BIC从小到大排序，选择最小BIC模型
        bic_with_candidates.sort(key = lambda x: x[1])
        best_feature, best_bic_for_feature = bic_with_candidates[0]

        ## 如果该特征的BIC小于当前最优BIC，则选择该特征
        if best_bic_for_feature < best_bic:
            selected_features.append(best_feature)
            remaining_features.remove(best_feature)
            best_bic = best_bic_for_feature
        else:
            ## 如果加入新特征后BIC并没有改善，则停止
            break

    ## 最终模型
    X_final = data[selected_features]
    X_final = add_constant(X_final)
    y_final = target

    final_model = sm.OLS(y_final, X_final).fit()

    params = final_model.params   ## 获取模型参数
    p_values = final_model.pvalues   ## 获取P值
    r_squared = final_model.rsquared   ## 获取R2
    y_pred = final_model.predict(X_final)   ## 获取Y的预测值
    rmse = np.sqrt(mean_squared_error(y, y_pred))  ## 计算RMSE
    results = pd.DataFrame({
        'Parameter': params.index,
        'Coefficient': params.values,
        'P-value': p_values.values
    })
    
    r2_row = pd.DataFrame({
        'Parameter': ['R-Squared'],
        'Coefficient': [r_squared],
        'P-Value':[None]
    })   ## 添加R2
    
    rmse_row = pd.DataFrame({
        'Parameter': ['RMSE'],
        'Coefficient': [rmse],
        'P-Value': [None]
    })   ## 添加RMSE
    
    results = pd.concat([results, r2_row, rmse_row], ignore_index= True)

    results['Coefficient'] = results['Coefficient'].round(3)

    return results, final_model, selected_features


# 多元线性回归模型 _ 逐步向前回归模型
## 预测 FMI
FMI_params, FMI_forward_linear_model, FMI_selected_feature = forward_stepwise_regression_bic(X_train, y_train['FMI'])
FMI_params.to_csv("C:/Users/zjl__/Desktop/output_zhaojialu/output/Coefficient1/FMI_params.csv", index = False)

## 预测 A/G
A_G_params, A_G_forward_linear_model, A_G_selected_feature = forward_stepwise_regression_bic(X_train, y_train['A_G'])
A_G_params.to_csv("C:/Users/zjl__/Desktop/output_zhaojialu/output/Coefficient1/A_G_params.csv", index = False)

## 预测 FM
FM_params, FM_forward_linear_model, FM_selected_feature = forward_stepwise_regression_bic(X_train, y_train['FM'])
FM_params.to_csv("C:/Users/zjl__/Desktop/output_zhaojialu/output/Coefficient1/FM_params.csv", index = False)

## 预测 LM
LM_params, LM_forward_linear_model, LM_selected_feature = forward_stepwise_regression_bic(X_train, y_train['LM'])
LM_params.to_csv("C:/Users/zjl__/Desktop/output_zhaojialu/output/Coefficient1/LM_params.csv", index = False)

## 预测 VATmass
VAT_params, VAT_forward_linear_model, VAT_selected_feature = forward_stepwise_regression_bic(X_train, y_train['VATmass'])
VAT_params.to_csv("C:/Users/zjl__/Desktop/output_zhaojialu/output/Coefficient1/VAT_params.csv", index = False)

## 预测 Android
Android_params, Android_forward_linear_model, Android_selected_feature = forward_stepwise_regression_bic(X_train, y_train['Android'])
Android_params.to_csv("C:/Users/zjl__/Desktop/output_zhaojialu/output/Coefficient1/Android_params.csv", index = False)

## 预测 Gynoid
Gynoid_params, Gynoid_forward_linear_model, Gynoid_selected_feature = forward_stepwise_regression_bic(X_train, y_train['Gynoid'])
Gynoid_params.to_csv("C:/Users/zjl__/Desktop/output_zhaojialu/output/Coefficient1/Gynoid_params.csv", index = False)

## 预测 BFP
BFP_params, BFP_forward_linear_model, BFP_selected_feature = forward_stepwise_regression_bic(X_train, y_train['BFP'])
BFP_params.to_csv("C:/Users/zjl__/Desktop/output_zhaojialu/output/Coefficient1/BFP_params.csv", index = False)

## 将多个模型和参数保存到一个字典中
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

## 保存字典

dump(AA_20240105_All_models, 'C:/Users/zjl__/Desktop/output_zhaojialu/output/ForwardRegression1/AA_20240105_All_models.joblib')
