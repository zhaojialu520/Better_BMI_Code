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
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from joblib import load
import seaborn as sns
from scipy.stats import t

import warnings
warnings.filterwarnings("ignore")


## ----------------------------------------------------------导入数据---------------------------------------------------------
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


## 加载之前用训练集训练的多个模型及参数
AA_20240105_All_VIF_models = load('C:/Users/zjl__/Desktop/output_zhaojialu/output/ForwardRegression2/AA_20240105_All_VIF_models.joblib')


# 定义一个缩放函数
def scale_prediction(prediction_obj, factor):
    # 复制对象（避免修改原始对象）
    scaled = prediction_obj.__class__(prediction_obj.predicted_mean / factor,
                                    prediction_obj.var_pred_mean / (factor**2),
                                    prediction_obj.__dict__)
    return scaled


## 获取 FMI 的相关模型及参数, 并预测 FMI，并提取95%置信区间（CI）和预测区间（PI）
FMI_forward_linear_model = AA_20240105_All_VIF_models['FMI_forward_linear_model']['Forward_linear_model']
FMI_selected_feature = AA_20240105_All_VIF_models['FMI_forward_linear_model']['Selected_feature']
print(FMI_selected_feature)
FMI_X_test_selected = X_test[list(FMI_selected_feature)]
FMI_X_test_selected = add_constant(FMI_X_test_selected)    ## 添加截距项
FMI_Linear_y_pred = FMI_forward_linear_model.get_prediction(FMI_X_test_selected)  # 预测新的点并提取CI和PI


## 获取 A/G 的相关模型及参数, 并预测 A/G
A_G_forward_linear_model = AA_20240105_All_VIF_models['A_G_forward_linear_model']['Forward_linear_model']
A_G_selected_feature = AA_20240105_All_VIF_models['A_G_forward_linear_model']['Selected_feature']
A_G_X_test_selected = X_test[list(A_G_selected_feature)]
A_G_X_test_selected = add_constant(A_G_X_test_selected)     ## 添加截距项
A_G_Linear_y_pred = A_G_forward_linear_model.get_prediction(A_G_X_test_selected)  # 预测新的点并提取CI和PI


## 获取 FM 的相关模型及参数, 并预测 FM
FM_forward_linear_model = AA_20240105_All_VIF_models['FM_forward_linear_model']['Forward_linear_model']
FM_selected_feature = AA_20240105_All_VIF_models['FM_forward_linear_model']['Selected_feature']
FM_X_test_selected = X_test[list(FM_selected_feature)]
FM_X_test_selected = add_constant(FM_X_test_selected)      ## 添加截距项
FM_Linear_y_pred = FM_forward_linear_model.get_prediction(FM_X_test_selected)  # 预测新的点并提取CI和PI


## 获取 LM 的相关模型及参数, 并预测 LM
LM_forward_linear_model = AA_20240105_All_VIF_models['LM_forward_linear_model']['Forward_linear_model']
LM_selected_feature = AA_20240105_All_VIF_models['LM_forward_linear_model']['Selected_feature']
LM_X_test_selected = X_test[list(LM_selected_feature)]
LM_X_test_selected = add_constant(LM_X_test_selected)       ## 添加截距项
LM_Linear_y_pred = LM_forward_linear_model.get_prediction(LM_X_test_selected)  # 预测新的点并提取CI和PI


## 获取 VATmass 的相关模型及参数, 并预测 VATmass
VAT_forward_linear_model = AA_20240105_All_VIF_models['VAT_forward_linear_model']['Forward_linear_model']
VAT_selected_feature = AA_20240105_All_VIF_models['VAT_forward_linear_model']['Selected_feature']
VAT_X_test_selected = X_test[list(VAT_selected_feature)]
VAT_X_test_selected = add_constant(VAT_X_test_selected)      ## 添加截距项
VAT_Linear_y_pred = VAT_forward_linear_model.get_prediction(VAT_X_test_selected)  # 预测新的点并提取CI和PI


## 获取  Android 的相关模型及参数, 并预测 Android
Android_forward_linear_model = AA_20240105_All_VIF_models['Android_forward_linear_model']['Forward_linear_model']
Android_selected_feature = AA_20240105_All_VIF_models['Android_forward_linear_model']['Selected_feature']
Android_X_test_selected = X_test[list(Android_selected_feature)]
Android_X_test_selected = add_constant(Android_X_test_selected)        ## 添加截距项
Android_Linear_y_pred = Android_forward_linear_model.get_prediction(Android_X_test_selected)  # 预测新的点并提取CI和PI


## 获取  Gynoid 的相关模型及参数, 并预测 Gynoid
Gynoid_forward_linear_model = AA_20240105_All_VIF_models['Gynoid_forward_linear_model']['Forward_linear_model']
Gynoid_selected_feature = AA_20240105_All_VIF_models['Gynoid_forward_linear_model']['Selected_feature']
Gynoid_X_test_selected = X_test[list(Gynoid_selected_feature)]
Gynoid_X_test_selected = add_constant(Gynoid_X_test_selected)          ## 添加截距项
Gynoid_Linear_y_pred = Gynoid_forward_linear_model.get_prediction(Gynoid_X_test_selected)  # 预测新的点并提取CI和PI


## 获取 BFP 的相关模型及参数, 并预测 BFP
BFP_forward_linear_model = AA_20240105_All_VIF_models['BFP_forward_linear_model']['Forward_linear_model']
BFP_selected_feature = AA_20240105_All_VIF_models['BFP_forward_linear_model']['Selected_feature']
BFP_X_test_selected = X_test[list(BFP_selected_feature)]
BFP_X_test_selected = add_constant(BFP_X_test_selected)                ## 添加截距项
BFP_Linear_y_pred = BFP_forward_linear_model.get_prediction(BFP_X_test_selected)  # 预测新的点并提取CI和PI
# print(BFP_Linear_y_pred)



## 循环添加变量
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

## 定义图的标题列表如下：
titles = ["FM", "FMI", "LM", "Android", "Gynoid", "VAT", "BFP", "AGFMR"]

## 创建子图布局
# fig, axes = plt.subplots(8, 2, figsize = (14, 32))   ## 8 行 2 列

### 定义 CCC 计算的函数
def calculate_ccc(y_true, y_pred):
    "计算一致性相关系数CCC"
    # 计算Pearson 相关系数
    corr, _ = pearsonr(y_true, y_pred)

    ## 计算均值
    mean_true = np.mean(y_true)
    mean_pred = np.mean(y_pred)

    ## 计算方差
    var_true = np.var(y_true)
    var_pred = np.var(y_pred)

    ## 计算协方差（用于分母
    covar = np.cov(y_true,y_pred)[0,1]

    ## 计算CCC
    numerator = 2 * corr * np.sqrt(var_true) * np.sqrt(var_pred)
    denominator = var_true + var_pred + (mean_true - mean_pred) ** 2
    ccc = numerator / denominator

    return ccc


## 循环绘制每组散点图， 并保存
for i in range(8):
    ## 获取当前的Y值 和预测值, R2
    Y = Y_list[i]
    predictions = predictions_list[i]

    ## 获取置信区间 & 预测区间
    pred_summary = predictions.summary_frame(alpha = 0.05)
    y_pred = pred_summary["mean"]  ## 预测均值
    # R2 = r2_score(Y, y_pred)     ## 计算原始预测方程的R2
    # conf_lower = pred_summary["mean_ci_lower"]  ## 点的置信区间下界
    # conf_upper = pred_summary["mean_ci_upper"]  ## 点的置信区间上界
    # pred_lower = pred_summary["obs_ci_lower"]   ## 点的预测区间下界
    # pred_upper = pred_summary["obs_ci_upper"]   ## 点的预测区间上界

    ## 对 Y 和 y_pred进行线性回归，计算拟合回归直线
    reg_model = sm.OLS(y_pred, sm.add_constant(Y)).fit()
    intercept, slope = reg_model.params  ##   计算斜率和截距
    y_pred_fitted = slope * Y + intercept    ##  计算拟合的回归线
    # print(f"Slope: {slope}, Intercept: {intercept}")
    CCC = calculate_ccc(y_pred, y_pred_fitted)
    # R2 = r2_score(predictions, y_pred_fitted)     ##  计算R2
    # RMSE = np.sqrt(mean_squared_error(predictions, y_pred_fitted))   ##  计算RMSE

    ##  计算残差标准误差（Residual Standard Error）
    n = len(Y)
    residuals = y_pred - y_pred_fitted
    RSE = np.sqrt(np.sum(residuals**2) / (n- 2))

    ##  计算置信区间的误差范围
    t_crit = t.ppf(0.975, df = n-2)   ##  95% 置信区间
    conf_interval = t_crit * RSE / np.sqrt(n)

    ## 计算拟合回归线的置信区间
    conf_lower_fit = y_pred_fitted - conf_interval
    conf_upper_fit = y_pred_fitted + conf_interval

    ##  计算预测区间
    sigma_pred = np.std(y_pred - y_pred_fitted)   ## 预测区间的标准误差
    pred_upper_line = y_pred_fitted + t_crit * sigma_pred
    pred_lower_line = y_pred_fitted - t_crit * sigma_pred

    ## 计算预测区间的动态偏移量
    # sigma_pred_upper = np.std(pred_upper - y_pred)     ##  计算预测区间的标准误差（上）
    # sigma_pred_lower = np.std(pred_lower - y_pred)     ##  计算预测区间的标准误差（下）
    # k = 1.96    ## 95% 置信水平
    # offset_upper = k * sigma_pred_upper        ##  计算动态偏移量(上)
    # offset_lower = k * sigma_pred_lower        ##  计算动态偏移量（下）

    ## 计算预测区间的上界和下界，使它们与拟合回归线平行
    # pred_upper_line = slope * Y + intercept + offset_upper
    # pred_lower_line = slope * Y + intercept - offset_lower

    ## Bland-Altman 计算
    mean_values = (Y + y_pred) / 2  ## 真实值和预测值的均值
    differences = y_pred - Y  ## 误差（预测 - 真实）
    mean_diff = np.mean(differences)  ## 均值差
    std_diff = np.std(differences)    ## 误差标准差
    upper_limit = mean_diff + 1.96 * std_diff    ## 95% 上限
    lower_limit = mean_diff - 1.96 * std_diff    ## 95% 下限

    plt.rcParams['font.family'] = 'DejaVu Serif'      ##  设置全局字体为“Times New Roman”
    ## 绘制 预测 VS 真实图 （左列）
    # ax1 = axes[i,0]
    fig1, ax1 = plt.subplots(figsize = (6,4))
    ax1.scatter(Y, y_pred, color = '#1f77b4', alpha = 0.7, s = 15)   ## 观测数据
    if intercept > 0 :
        equation = rf"$Y_{{pred}} = {intercept:.2f} + {slope:.2f}Y$"
    elif intercept < 0 :
        equation = rf"$Y_{{pre d}} = -{-intercept:.2f} + {slope:.2f}Y$"
    else :
        equation = rf"$Y_{{pre d}} = {slope:.2f}Y$"
    ax1.plot(Y, y_pred_fitted, label = equation, color = '#1f77b4', linewidth = 1)   ## 回归线
    ax1.fill_between(Y, conf_lower_fit, conf_upper_fit, color = '#1f77b4', alpha = 0.2, label = "95% Confidence Limits")   ##
    ax1.plot(Y, pred_upper_line, linestyle = ":", color = '#1f77b4', linewidth = 0.5, label = "95% Prediction Limits")
    ax1.plot(Y, pred_lower_line, linestyle = ":", color = '#1f77b4', linewidth = 0.5)
    ax1.plot(Y, Y, color = 'darkblue', linewidth = 0.8, label = "Y = X")
    ax1.set_xlabel(f"DAX True Values of {titles[i]} Equation ", fontsize=10, color = 'black')
    ax1.set_ylabel(f"Predicted Values of {titles[i]} Equation ", fontsize=10, color='black')
    ax1.legend(loc = 'upper left', fontsize = 10, markerscale = 1, labelspacing = 0.5, frameon = False)
    ax1.text(0.95, 0.05, f"$CCC = {CCC:.2f}$", transform = ax1.transAxes, fontsize = 10, color = "black", ha ="right", va = "bottom", fontfamily = 'DejaVu Serif')
    # ax1.text(0.95, 0.05,     ## 添加R2 和 RMSE 的文本到图的右下角（设置文本相对坐标）
    #       f"$R^{{2}} = {R2:.2f}$ \n $RMSE = {RMSE:.2f}$",
    #      transform = ax1.transAxes,    ##  使用相对坐标系
    #         fontsize = 10, color = "black", ha ="right", va = "bottom",
    #        fontfamily = 'DejaVu Serif'
    #        )
    ax1.grid(True, linestyle = '--', linewidth = 0.5)
    plt.savefig(f"C:/Users/zjl__/Desktop/output_zhaojialu/output/RegressionPlot_BlandAltman2/Pred_vs_True_{titles[i]}.png", dpi = 600, bbox_inches = 'tight')   ## 保存图像
    plt.close(fig1)   ##  关闭图像，避免内存泄露

    ## 绘制 Bland - Altman 图 （右列）
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
    plt.savefig(f"C:/Users/zjl__/Desktop/output_zhaojialu/output/RegressionPlot_BlandAltman2/Bland_Altman{titles[i]}.png", dpi = 600, bbox_inches = 'tight')   ##  保存图像
    plt.close(fig2)    ##  关闭图像，避免内存泄露

## 调整子图之间的间距
# plt.tight_layout()

## 保存整张图片
# plt.savefig("/public/sandbox/workdir/zhaojialu/Result/Test_line_plot.png", dpi = 300, bbox_inches = "tight")
# plt.show()

