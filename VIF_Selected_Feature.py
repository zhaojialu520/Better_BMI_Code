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
from matplotlib import font_manager
from matplotlib.colors import TwoSlopeNorm, ListedColormap, LinearSegmentedColormap
from statsmodels.stats.outliers_influence import variance_inflation_factor
from joblib import dump

from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
from statsmodels.stats.stattools import durbin_watson

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

## 加载之前用训练集训练的多个模型及参数
AA_20240105_All_models = load('C:/Users/zjl__/Desktop/output_zhaojialu/output/ForwardRegression1/AA_20240105_All_models.joblib')

"""
def feature_selection_process(data, target_col, significance_level=0.05, vif_threshold=10):
    """
    特征选择流程：先过滤特定关键词，然后循环进行显著性检验和VIF检验
    
    参数:
    data: 包含特征和目标变量的DataFrame
    target_col: 目标变量列名
    significance_level: 显著性水平，默认0.05
    vif_threshold: VIF阈值，默认10
    
    返回:
    最终选择的特征列表和最终的回归模型
    """
    
    # 1. 筛选掉包含特定关键词的特征
    exclude_keywords = ['3D', '腰带', '垂直', '内部']
    initial_features = [col for col in data.columns 
                       if col != target_col and 
                       not any(keyword in col for keyword in exclude_keywords)]
    
    print(f"初始特征数量: {len(initial_features)}")
    print(f"排除的特征: {[col for col in data.columns if col != target_col and col not in initial_features]}")
    
    current_features = initial_features.copy()
    iteration = 1
    
    while True:
        print(f"\n=== 第 {iteration} 次迭代 ===")
        
        # 准备数据
        X = data[current_features]
        y = data[target_col]
        X_with_const = add_constant(X)
        
        # 拟合回归模型
        try:
            model = sm.OLS(y, X_with_const).fit()
        except Exception as e:
            print(f"回归模型拟合失败: {e}")
            break
            
        # 2. 检查自变量显著性
        pvalues = model.pvalues.drop('const')
        insignificant_features = pvalues[pvalues > significance_level].index.tolist()
        
        # 3. 计算VIF值
        vif_data = pd.DataFrame()
        vif_data["feature"] = X_with_const.columns
        vif_data["VIF"] = [variance_inflation_factor(X_with_const.values, i) 
                          for i in range(X_with_const.shape[1])]
        

        vif_data = vif_data[vif_data['feature'] != 'const']
        
        high_vif_features = vif_data[vif_data["VIF"] > vif_threshold]
        
        print(f"当前特征数量: {len(current_features)}")
        print(f"不显著的特征 (p > {significance_level}): {insignificant_features}")
        print(f"VIF > {vif_threshold} 的特征:")
        print(high_vif_features)
        
        # 检查终止条件
        if len(insignificant_features) == 0 and len(high_vif_features) == 0:
            print("\n✅ 达到终止条件: 所有特征都显著且VIF < 10")
            break
            
        # 决定要移除的特征
        features_to_remove = []
        
        # 优先移除不显著的特征
        if len(insignificant_features) > 0:
            # 移除p值最大的不显著特征
            max_p_feature = pvalues.loc[insignificant_features].idxmax()
            features_to_remove.append(max_p_feature)
            print(f"移除不显著特征: {max_p_feature} (p-value: {pvalues[max_p_feature]:.4f})")
        
        # 如果没有不显著特征但有高VIF特征，移除VIF最大的特征
        elif len(high_vif_features) > 0:
            max_vif_feature = high_vif_features.loc[high_vif_features["VIF"].idxmax(), "feature"]
            features_to_remove.append(max_vif_feature)
            print(f"移除高VIF特征: {max_vif_feature} (VIF: {high_vif_features.loc[high_vif_features['VIF'].idxmax(), 'VIF']:.2f})")
        
        # 移除特征
        for feature in features_to_remove:
            current_features.remove(feature)
            
        # 检查是否还有特征剩余
        if len(current_features) == 0:
            print("❌ 所有特征都被移除了！")
            break
            
        iteration += 1
        
        # 防止无限循环
        if iteration > 50:
            print("⚠️ 达到最大迭代次数，强制终止")
            break
    
    # 输出最终结果
    print(f"\n🎯 最终选择的特征 ({len(current_features)}个):")
    for feature in current_features:
        p_value = model.pvalues[feature] if feature in model.pvalues else 'N/A'
        vif_value = vif_data[vif_data['feature'] == feature]['VIF'].values[0] if feature in vif_data['feature'].values else 'N/A'
        print(f"  {feature}: p-value={p_value:.4f}, VIF={vif_value}")
    
    return current_features, model

def feature_selection_process(data, target_col, significance_level=0.05, vif_threshold=10, protected_feature='标准站姿腰围'):
    """
    特征选择流程：先过滤特定关键词，然后循环进行显著性检验和VIF检验
    在排除高VIF特征时会保护指定特征
    
    参数:
    data: 包含特征和目标变量的DataFrame
    target_col: 目标变量列名
    significance_level: 显著性水平，默认0.05
    vif_threshold: VIF阈值，默认10
    protected_feature: 要保护的特征名称，不会被VIF筛选移除
    
    返回:
    最终选择的特征列表和最终的回归模型
    """
    
    # 1. 筛选掉包含特定关键词的特征
    exclude_keywords = ['3D', '腰带', '垂直', '内部', '躯干扭', '角度']
    initial_features = [col for col in data.columns 
                       if col != target_col and 
                       not any(keyword in col for keyword in exclude_keywords)]
    
    print(f"初始特征数量: {len(initial_features)}")
    print(f"排除的特征: {[col for col in data.columns if col != target_col and col not in initial_features]}")
    print(f"保护的特征: {protected_feature}")
    
    current_features = initial_features.copy()
    iteration = 1
    
    while True:
        print(f"\n=== 第 {iteration} 次迭代 ===")
        
        # 准备数据
        X = data[current_features]
        y = data[target_col]
        X_with_const = add_constant(X)
        
        # 拟合回归模型
        try:
            model = sm.OLS(y, X_with_const).fit()
        except Exception as e:
            print(f"回归模型拟合失败: {e}")
            break
            
        # 2. 检查自变量显著性
        pvalues = model.pvalues.drop('const')
        insignificant_features = pvalues[pvalues > significance_level].index.tolist()
        
        # 3. 计算VIF值
        vif_data = pd.DataFrame()
        vif_data["feature"] = X_with_const.columns
        vif_data["VIF"] = [variance_inflation_factor(X_with_const.values, i) 
                          for i in range(X_with_const.shape[1])]
        
        vif_data = vif_data[vif_data['feature'] != 'const']
        high_vif_features = vif_data[vif_data["VIF"] > vif_threshold]
        
        print(f"当前特征数量: {len(current_features)}")
        print(f"不显著的特征 (p > {significance_level}): {insignificant_features}")
        print(f"VIF > {vif_threshold} 的特征:")
        print(high_vif_features)
        
        # 检查终止条件
        if len(insignificant_features) == 0 and len(high_vif_features) == 0:
            print("\n✅ 达到终止条件: 所有特征都显著且VIF < 10")
            break
            
        # 决定要移除的特征
        features_to_remove = []
        
        # 优先移除不显著的特征
        if len(insignificant_features) > 0:
            # 移除p值最大的不显著特征
            max_p_feature = pvalues.loc[insignificant_features].idxmax()
            features_to_remove.append(max_p_feature)
            print(f"移除不显著特征: {max_p_feature} (p-value: {pvalues[max_p_feature]:.4f})")
        
        # 如果没有不显著特征但有高VIF特征，移除VIF最大的特征（跳过保护的特征）
        elif len(high_vif_features) > 0:
            # 创建可移除的高VIF特征列表（排除保护的特征）
            removable_high_vif_features = high_vif_features[high_vif_features['feature'] != protected_feature]
            
            if len(removable_high_vif_features) > 0:
                # 从可移除的特征中找到VIF最大的特征
                max_vif_feature = removable_high_vif_features.loc[removable_high_vif_features["VIF"].idxmax(), "feature"]
                features_to_remove.append(max_vif_feature)
                print(f"移除高VIF特征: {max_vif_feature} (VIF: {removable_high_vif_features.loc[removable_high_vif_features['VIF'].idxmax(), 'VIF']:.2f})")
            else:
                # 如果只有保护的特征是高VIF，检查是否有不显著特征可以移除
                if len(insignificant_features) > 0:
                    # 移除p值最大的不显著特征
                    max_p_feature = pvalues.loc[insignificant_features].idxmax()
                    features_to_remove.append(max_p_feature)
                    print(f"只有{protected_feature}是高VIF特征，移除不显著特征: {max_p_feature} (p-value: {pvalues[max_p_feature]:.4f})")
                else:
                    print(f"⚠️ 只有{protected_feature}是高VIF特征且所有特征都显著，无法继续优化VIF")
                    break
        
        # 移除特征
        for feature in features_to_remove:
            if feature in current_features:
                current_features.remove(feature)
                print(f"已移除特征: {feature}")
            else:
                print(f"警告: 特征 {feature} 不在当前特征列表中")
            
        # 检查是否还有特征剩余
        if len(current_features) == 0:
            print("❌ 所有特征都被移除了！")
            break
            
        iteration += 1
        
        # 防止无限循环
        if iteration > 50:
            print("⚠️ 达到最大迭代次数，强制终止")
            break
    
    # 检查保护的特征是否还在最终特征集中
    if protected_feature in current_features:
        print(f"\n✅ 保护的特征 '{protected_feature}' 已保留在最终特征集中")
    else:
        print(f"\n⚠️ 保护的特征 '{protected_feature}' 未在最终特征集中")
    
    # 输出最终结果
    print(f"\n🎯 最终选择的特征 ({len(current_features)}个):")
    # 重新拟合最终模型以确保准确性
    X_final = data[current_features]
    y_final = data[target_col]
    X_final_with_const = add_constant(X_final)
    final_model = sm.OLS(y_final, X_final_with_const).fit()
    
    # 计算最终VIF值
    final_vif_data = pd.DataFrame()
    final_vif_data["feature"] = X_final_with_const.columns
    final_vif_data["VIF"] = [variance_inflation_factor(X_final_with_const.values, i) 
                           for i in range(X_final_with_const.shape[1])]
    final_vif_data = final_vif_data[final_vif_data['feature'] != 'const']
    
    for feature in current_features:
        p_value = final_model.pvalues[feature]
        vif_value = final_vif_data[final_vif_data['feature'] == feature]['VIF'].values[0]
        protected_indicator = " (保护)" if feature == protected_feature else ""
        print(f"  {feature}: p-value={p_value:.4f}, VIF={vif_value:.2f}{protected_indicator}")
    
    return current_features, final_model

## 先VIF，再显著性

def feature_selection_process2(data, target_col, significance_level=0.05, vif_threshold=10):
    """
    特征选择流程：先过滤特定关键词，然后循环进行VIF检验和显著性检验
    
    参数:
    data: 包含特征和目标变量的DataFrame
    target_col: 目标变量列名
    significance_level: 显著性水平，默认0.05
    vif_threshold: VIF阈值，默认10
    
    返回:
    最终选择的特征列表和最终的回归模型
    """
    
    # 1. 筛选掉包含特定关键词的特征
    exclude_keywords = ['3D', '腰带', '垂直', '内部']
    initial_features = [col for col in data.columns 
                       if col != target_col and 
                       not any(keyword in col for keyword in exclude_keywords)]
    
    print(f"初始特征数量: {len(initial_features)}")
    print(f"排除的特征: {[col for col in data.columns if col != target_col and col not in initial_features]}")
    
    current_features = initial_features.copy()
    iteration = 1
    
    while True:
        print(f"\n=== 第 {iteration} 次迭代 ===")
        
        # 准备数据
        X = data[current_features]
        y = data[target_col]
        X_with_const = add_constant(X)
        
        # 拟合回归模型
        try:
            model = sm.OLS(y, X_with_const).fit()
        except Exception as e:
            print(f"回归模型拟合失败: {e}")
            break
            
        # 2. 计算VIF值（先进行VIF筛选）
        vif_data = pd.DataFrame()
        vif_data["feature"] = X_with_const.columns
        vif_data["VIF"] = [variance_inflation_factor(X_with_const.values, i) 
                          for i in range(X_with_const.shape[1])]
        
        vif_data = vif_data[vif_data['feature'] != 'const']
        high_vif_features = vif_data[vif_data["VIF"] > vif_threshold]
        
        # 3. 检查自变量显著性（后进行显著性筛选）
        pvalues = model.pvalues.drop('const')
        insignificant_features = pvalues[pvalues > significance_level].index.tolist()
        
        print(f"当前特征数量: {len(current_features)}")
        print(f"VIF > {vif_threshold} 的特征:")
        print(high_vif_features)
        print(f"不显著的特征 (p > {significance_level}): {insignificant_features}")
        
        # 检查终止条件
        if len(high_vif_features) == 0 and len(insignificant_features) == 0:
            print("\n✅ 达到终止条件: 所有特征VIF < 10且都显著")
            break
            
        # 决定要移除的特征（优先移除高VIF特征）
        features_to_remove = []
        
        # 优先移除高VIF的特征
        if len(high_vif_features) > 0:
            # 移除VIF值最大的特征
            max_vif_feature = high_vif_features.loc[high_vif_features["VIF"].idxmax(), "feature"]
            features_to_remove.append(max_vif_feature)
            print(f"移除高VIF特征: {max_vif_feature} (VIF: {high_vif_features.loc[high_vif_features['VIF'].idxmax(), 'VIF']:.2f})")
        
        # 如果没有高VIF特征但有不显著特征，移除p值最大的特征
        elif len(insignificant_features) > 0:
            # 移除p值最大的不显著特征
            max_p_feature = pvalues.loc[insignificant_features].idxmax()
            features_to_remove.append(max_p_feature)
            print(f"移除不显著特征: {max_p_feature} (p-value: {pvalues[max_p_feature]:.4f})")
        
        # 移除特征
        for feature in features_to_remove:
            current_features.remove(feature)
            
        # 检查是否还有特征剩余
        if len(current_features) == 0:
            print("❌ 所有特征都被移除了！")
            break
            
        iteration += 1
        
        # 防止无限循环
        if iteration > 50:
            print("⚠️ 达到最大迭代次数，强制终止")
            break
    
    # 输出最终结果
    print(f"\n🎯 最终选择的特征 ({len(current_features)}个):")
    # 拟合最终模型
    X_final = data[current_features]
    y_final = data[target_col]
    X_final_with_const = add_constant(X_final)
    final_model = sm.OLS(y_final, X_final_with_const).fit()
    
    # 计算最终VIF值
    final_vif_data = pd.DataFrame()
    final_vif_data["feature"] = X_final_with_const.columns
    final_vif_data["VIF"] = [variance_inflation_factor(X_final_with_const.values, i) 
                           for i in range(X_final_with_const.shape[1])]
    final_vif_data = final_vif_data[final_vif_data['feature'] != 'const']
    
    for feature in current_features:
        p_value = final_model.pvalues[feature]
        vif_value = final_vif_data[final_vif_data['feature'] == feature]['VIF'].values[0]
        print(f"  {feature}: p-value={p_value:.4f}, VIF={vif_value:.2f}")
    
    return current_features, final_model

def feature_selection_process2(data, target_col, significance_level=0.05, vif_threshold=10, protected_feature='标准站姿腰围'):
    """
    特征选择流程：先过滤特定关键词，然后循环进行VIF检验和显著性检验
    在排除高VIF特征时会保护指定特征
    
    参数:
    data: 包含特征和目标变量的DataFrame
    target_col: 目标变量列名
    significance_level: 显著性水平，默认0.05
    vif_threshold: VIF阈值，默认10
    protected_feature: 要保护的特征名称，不会被VIF筛选移除
    
    返回:
    最终选择的特征列表和最终的回归模型
    """
    
    # 1. 筛选掉包含特定关键词的特征
    exclude_keywords = ['3D', '腰带', '垂直', '内部']
    initial_features = [col for col in data.columns 
                       if col != target_col and 
                       not any(keyword in col for keyword in exclude_keywords)]
    
    print(f"初始特征数量: {len(initial_features)}")
    print(f"排除的特征: {[col for col in data.columns if col != target_col and col not in initial_features]}")
    print(f"保护的特征: {protected_feature}")
    
    current_features = initial_features.copy()
    iteration = 1
    
    while True:
        print(f"\n=== 第 {iteration} 次迭代 ===")
        
        # 准备数据
        X = data[current_features]
        y = data[target_col]
        X_with_const = add_constant(X)
        
        # 拟合回归模型
        try:
            model = sm.OLS(y, X_with_const).fit()
        except Exception as e:
            print(f"回归模型拟合失败: {e}")
            break
            
        # 2. 计算VIF值（先进行VIF筛选）
        vif_data = pd.DataFrame()
        vif_data["feature"] = X_with_const.columns
        vif_data["VIF"] = [variance_inflation_factor(X_with_const.values, i) 
                          for i in range(X_with_const.shape[1])]
        
        vif_data = vif_data[vif_data['feature'] != 'const']
        high_vif_features = vif_data[vif_data["VIF"] > vif_threshold]
        
        # 3. 检查自变量显著性（后进行显著性筛选）
        pvalues = model.pvalues.drop('const')
        insignificant_features = pvalues[pvalues > significance_level].index.tolist()
        
        print(f"当前特征数量: {len(current_features)}")
        print(f"VIF > {vif_threshold} 的特征:")
        print(high_vif_features)
        print(f"不显著的特征 (p > {significance_level}): {insignificant_features}")
        
        # 检查终止条件
        if len(high_vif_features) == 0 and len(insignificant_features) == 0:
            print("\n✅ 达到终止条件: 所有特征VIF < 10且都显著")
            break
            
        # 决定要移除的特征（优先移除高VIF特征）
        features_to_remove = []
        
        # 优先移除高VIF的特征
        if len(high_vif_features) > 0:
            # 创建可移除的高VIF特征列表（排除保护的特征）
            removable_high_vif_features = high_vif_features[high_vif_features['feature'] != protected_feature]
            
            if len(removable_high_vif_features) > 0:
                # 从可移除的特征中找到VIF最大的特征
                max_vif_feature = removable_high_vif_features.loc[removable_high_vif_features["VIF"].idxmax(), "feature"]
                features_to_remove.append(max_vif_feature)
                print(f"移除高VIF特征: {max_vif_feature} (VIF: {removable_high_vif_features.loc[removable_high_vif_features['VIF'].idxmax(), 'VIF']:.2f})")
            else:
                # 如果只有保护的特征是高VIF，检查是否有不显著特征可以移除
                if len(insignificant_features) > 0:
                    # 移除p值最大的不显著特征
                    max_p_feature = pvalues.loc[insignificant_features].idxmax()
                    features_to_remove.append(max_p_feature)
                    print(f"只有{protected_feature}是高VIF特征，移除不显著特征: {max_p_feature} (p-value: {pvalues[max_p_feature]:.4f})")
                else:
                    print(f"⚠️ 只有{protected_feature}是高VIF特征且所有特征都显著，无法继续优化VIF")
                    break
        
        # 如果没有高VIF特征但有不显著特征，移除p值最大的特征
        elif len(insignificant_features) > 0:
            # 移除p值最大的不显著特征
            max_p_feature = pvalues.loc[insignificant_features].idxmax()
            features_to_remove.append(max_p_feature)
            print(f"移除不显著特征: {max_p_feature} (p-value: {pvalues[max_p_feature]:.4f})")
        
        # 移除特征
        for feature in features_to_remove:
            if feature in current_features:
                current_features.remove(feature)
                print(f"已移除特征: {feature}")
            else:
                print(f"警告: 特征 {feature} 不在当前特征列表中")
            
        # 检查是否还有特征剩余
        if len(current_features) == 0:
            print("❌ 所有特征都被移除了！")
            break
            
        iteration += 1
        
        # 防止无限循环
        if iteration > 50:
            print("⚠️ 达到最大迭代次数，强制终止")
            break
    
    # 检查保护的特征是否还在最终特征集中
    if protected_feature in current_features:
        print(f"\n✅ 保护的特征 '{protected_feature}' 已保留在最终特征集中")
    else:
        print(f"\n⚠️ 保护的特征 '{protected_feature}' 未在最终特征集中")
    
    # 输出最终结果
    print(f"\n🎯 最终选择的特征 ({len(current_features)}个):")
    # 拟合最终模型
    X_final = data[current_features]
    y_final = data[target_col]
    X_final_with_const = add_constant(X_final)
    final_model = sm.OLS(y_final, X_final_with_const).fit()
    
    # 计算最终VIF值
    final_vif_data = pd.DataFrame()
    final_vif_data["feature"] = X_final_with_const.columns
    final_vif_data["VIF"] = [variance_inflation_factor(X_final_with_const.values, i) 
                           for i in range(X_final_with_const.shape[1])]
    final_vif_data = final_vif_data[final_vif_data['feature'] != 'const']
    
    for feature in current_features:
        p_value = final_model.pvalues[feature]
        vif_value = final_vif_data[final_vif_data['feature'] == feature]['VIF'].values[0]
        protected_indicator = " (保护)" if feature == protected_feature else ""
        print(f"  {feature}: p-value={p_value:.4f}, VIF={vif_value:.2f}{protected_indicator}")
    
    return current_features, final_model


def feature_selection_process3(data, target_col, significance_level=0.05, vif_threshold=10, r2_tolerance=0.02):
    """
    优化版特征选择流程：在VIF和显著性筛选的同时，最大化保持模型R²准确性
    
    参数:
    data: 包含特征和目标变量的DataFrame
    target_col: 目标变量列名
    significance_level: 显著性水平，默认0.05
    vif_threshold: VIF阈值，默认10
    r2_tolerance: R²下降容忍度，默认0.02（2%）
    
    返回:
    最终选择的特征列表和最终的回归模型
    """
    
    # 1. 筛选掉包含特定关键词的特征
    exclude_keywords = ['3D', '腰带', '垂直', '内部', '躯干扭', '角度']
    initial_features = [col for col in data.columns 
                       if col != target_col and 
                       not any(keyword in col for keyword in exclude_keywords)]
    
    print(f"初始特征数量: {len(initial_features)}")
    print(f"排除的特征: {[col for col in data.columns if col != target_col and col not in initial_features]}")
    print(f"R²容忍度: {r2_tolerance}")
    
    current_features = initial_features.copy()
    iteration = 1
    best_r2 = 0
    best_features = current_features.copy()
    
    # 计算初始模型的R²作为基准
    X_initial = data[initial_features]
    y_initial = data[target_col]
    X_initial_with_const = add_constant(X_initial)
    initial_model = sm.OLS(y_initial, X_initial_with_const).fit()
    baseline_r2 = initial_model.rsquared
    best_r2 = baseline_r2
    print(f"基准模型R²: {baseline_r2:.4f}")
    
    while True:
        print(f"\n=== 第 {iteration} 次迭代 ===")
        
        # 准备数据
        X = data[current_features]
        y = data[target_col]
        X_with_const = add_constant(X)
        
        # 拟合回归模型
        try:
            model = sm.OLS(y, X_with_const).fit()
            current_r2 = model.rsquared
            current_adj_r2 = model.rsquared_adj
        except Exception as e:
            print(f"回归模型拟合失败: {e}")
            break
            
        # 2. 检查自变量显著性
        pvalues = model.pvalues.drop('const')
        insignificant_features = pvalues[pvalues > significance_level].index.tolist()
        
        # 3. 计算VIF值
        vif_data = pd.DataFrame()
        vif_data["feature"] = X_with_const.columns
        vif_data["VIF"] = [variance_inflation_factor(X_with_const.values, i) 
                          for i in range(X_with_const.shape[1])]
        
        vif_data = vif_data[vif_data['feature'] != 'const']
        high_vif_features = vif_data[vif_data["VIF"] > vif_threshold]
        
        print(f"当前特征数量: {len(current_features)}")
        print(f"当前R²: {current_r2:.4f}, 调整后R²: {current_adj_r2:.4f}")
        print(f"不显著的特征 (p > {significance_level}): {insignificant_features}")
        print(f"VIF > {vif_threshold} 的特征:")
        print(high_vif_features)
        
        # 检查终止条件
        if len(insignificant_features) == 0 and len(high_vif_features) == 0:
            print("\n✅ 达到终止条件: 所有特征都显著且VIF < 10")
            break
            
        # 记录当前最佳模型（如果R²下降在容忍范围内且满足其他条件）
        r2_drop = baseline_r2 - current_r2
        if (len(insignificant_features) == 0 and 
            r2_drop <= r2_tolerance and 
            current_r2 > best_r2 - 0.01):  # 允许轻微下降
            best_r2 = current_r2
            best_features = current_features.copy()
            print(f"更新最佳模型: R² = {current_r2:.4f}, 特征数 = {len(current_features)}")
        
        # 生成候选移除特征列表（考虑对R²的影响）
        candidate_features_to_remove = []
        
        # 优先考虑不显著的特征
        if len(insignificant_features) > 0:
            # 对不显著特征按p值排序，并评估移除对R²的影响
            for feature in sorted(insignificant_features, key=lambda x: pvalues[x], reverse=True):
                candidate_features_to_remove.append({
                    'feature': feature,
                    'reason': 'insignificant',
                    'p_value': pvalues[feature],
                    'vif': vif_data[vif_data['feature'] == feature]['VIF'].values[0] if feature in vif_data['feature'].values else 0
                })
        
        # 然后考虑高VIF特征
        if len(high_vif_features) > 0:
            for _, row in high_vif_features.sort_values('VIF', ascending=False).iterrows():
                candidate_features_to_remove.append({
                    'feature': row['feature'],
                    'reason': 'high_vif',
                    'p_value': pvalues[row['feature']] if row['feature'] in pvalues else 1.0,
                    'vif': row['VIF']
                })
        
        # 智能选择要移除的特征
        features_to_remove = []
        
        if candidate_features_to_remove:
            # 评估移除每个候选特征对R²的影响
            removal_impact = []
            for candidate in candidate_features_to_remove:
                feature = candidate['feature']
                # 测试移除该特征后的R²
                test_features = [f for f in current_features if f != feature]
                if test_features:  # 确保还有特征剩余
                    X_test = data[test_features]
                    X_test_with_const = add_constant(X_test)
                    try:
                        test_model = sm.OLS(y, X_test_with_const).fit()
                        r2_after_removal = test_model.rsquared
                        r2_drop = current_r2 - r2_after_removal
                    except:
                        r2_drop = float('inf')  # 如果拟合失败，认为影响很大
                else:
                    r2_drop = float('inf')
                
                removal_impact.append({
                    'feature': feature,
                    'reason': candidate['reason'],
                    'p_value': candidate['p_value'],
                    'vif': candidate['vif'],
                    'r2_drop': r2_drop
                })
            
            # 选择对R²影响最小的特征进行移除
            if removal_impact:
                # 优先选择对R²影响小且不显著的特征
                low_impact_insignificant = [x for x in removal_impact if x['reason'] == 'insignificant' and x['r2_drop'] <= r2_tolerance]
                if low_impact_insignificant:
                    best_candidate = min(low_impact_insignificant, key=lambda x: x['r2_drop'])
                else:
                    # 如果没有对R²影响小的不显著特征，选择对R²影响小的高VIF特征
                    low_impact_high_vif = [x for x in removal_impact if x['reason'] == 'high_vif' and x['r2_drop'] <= r2_tolerance]
                    if low_impact_high_vif:
                        best_candidate = min(low_impact_high_vif, key=lambda x: x['r2_drop'])
                    else:
                        # 如果所有移除都会导致R²大幅下降，选择影响相对最小的
                        best_candidate = min(removal_impact, key=lambda x: x['r2_drop'])
                
                features_to_remove.append(best_candidate['feature'])
                print(f"移除特征: {best_candidate['feature']} ({best_candidate['reason']})")
                print(f"  p-value: {best_candidate['p_value']:.4f}, VIF: {best_candidate['vif']:.2f}, R²下降: {best_candidate['r2_drop']:.4f}")
        
        # 如果没有合适的候选特征（所有移除都会导致R²大幅下降）
        if not features_to_remove:
            print("⚠️ 没有合适的特征可以移除（所有移除都会导致R²大幅下降）")
            # 检查是否满足放宽的条件
            if len(insignificant_features) == 0:
                print("✅ 所有特征都显著，接受当前模型")
                break
            else:
                # 强制移除p值最大的特征
                if insignificant_features:
                    max_p_feature = max(insignificant_features, key=lambda x: pvalues[x])
                    features_to_remove.append(max_p_feature)
                    print(f"强制移除最不显著特征: {max_p_feature}")
        
        # 移除特征
        for feature in features_to_remove:
            if feature in current_features:
                current_features.remove(feature)
                print(f"已移除特征: {feature}")
            
        # 检查是否还有特征剩余
        if len(current_features) == 0:
            print("❌ 所有特征都被移除了！")
            current_features = best_features  # 恢复到最佳特征集
            break
            
        iteration += 1
        
        # 防止无限循环
        if iteration > 30:
            print("⚠️ 达到最大迭代次数，强制终止")
            break
    
    # 最终模型评估
    print(f"\n🎯 最终选择的特征 ({len(current_features)}个):")
    X_final = data[current_features]
    y_final = data[target_col]
    X_final_with_const = add_constant(X_final)
    final_model = sm.OLS(y_final, X_final_with_const).fit()
    
    # 计算最终指标
    final_vif_data = pd.DataFrame()
    final_vif_data["feature"] = X_final_with_const.columns
    final_vif_data["VIF"] = [variance_inflation_factor(X_final_with_const.values, i) 
                           for i in range(X_final_with_const.shape[1])]
    final_vif_data = final_vif_data[final_vif_data['feature'] != 'const']
    
    final_r2 = final_model.rsquared
    final_adj_r2 = final_model.rsquared_adj
    r2_reduction = baseline_r2 - final_r2
    
    print(f"\n📊 模型性能总结:")
    print(f"初始R²: {baseline_r2:.4f}")
    print(f"最终R²: {final_r2:.4f}")
    print(f"R²下降: {r2_reduction:.4f} ({r2_reduction/baseline_r2*100:.1f}%)")
    print(f"调整后R²: {final_adj_r2:.4f}")
    
    for feature in current_features:
        p_value = final_model.pvalues[feature]
        vif_value = final_vif_data[final_vif_data['feature'] == feature]['VIF'].values[0]
        significance_indicator = " ✓" if p_value <= significance_level else " ✗"
        vif_indicator = " ✓" if vif_value <= vif_threshold else " ✗"
        print(f"  {feature}: p-value={p_value:.4f}{significance_indicator}, VIF={vif_value:.2f}{vif_indicator}")
    
    return current_features, final_model
"""

def feature_selection_process4(data, target_col, significance_level=0.05, vif_threshold=10, r2_tolerance=0.02):
    """
    优化版特征选择流程：在VIF和显著性筛选的同时，最大化保持模型R²准确性
    优先移除对模型性能影响最小的特征
    
    参数:
    data: 包含特征和目标变量的DataFrame
    target_col: 目标变量列名
    significance_level: 显著性水平，默认0.05
    vif_threshold: VIF阈值，默认10
    r2_tolerance: R²下降容忍度，默认0.02（2%）
    
    返回:
    最终选择的特征列表和最终的回归模型
    """
    
    # 1. 筛选掉包含特定关键词的特征
    exclude_keywords = ['3D', '腰带', '垂直', '内部', '躯干扭', '角度']
    initial_features = [col for col in data.columns 
                       if col != target_col and 
                       not any(keyword in col for keyword in exclude_keywords)]
    
    print(f"初始特征数量: {len(initial_features)}")
    print(f"排除的特征: {[col for col in data.columns if col != target_col and col not in initial_features]}")
    print(f"R²容忍度: {r2_tolerance}")
    
    current_features = initial_features.copy()
    iteration = 1
    best_r2 = 0
    best_features = current_features.copy()
    
    # 计算初始模型的R²作为基准
    X_initial = data[initial_features]
    y_initial = data[target_col]
    X_initial_with_const = add_constant(X_initial)
    initial_model = sm.OLS(y_initial, X_initial_with_const).fit()
    baseline_r2 = initial_model.rsquared
    best_r2 = baseline_r2
    print(f"基准模型R²: {baseline_r2:.4f}")
    
    while True:
        print(f"\n=== 第 {iteration} 次迭代 ===")
        
        # 准备数据
        X = data[current_features]
        y = data[target_col]
        X_with_const = add_constant(X)
        
        # 拟合回归模型
        try:
            model = sm.OLS(y, X_with_const).fit()
            current_r2 = model.rsquared
            current_adj_r2 = model.rsquared_adj
        except Exception as e:
            print(f"回归模型拟合失败: {e}")
            break
            
        # 2. 检查自变量显著性
        pvalues = model.pvalues.drop('const')
        insignificant_features = pvalues[pvalues > significance_level].index.tolist()
        
        # 3. 计算VIF值
        vif_data = pd.DataFrame()
        vif_data["feature"] = X_with_const.columns
        vif_data["VIF"] = [variance_inflation_factor(X_with_const.values, i) 
                          for i in range(X_with_const.shape[1])]
        
        vif_data = vif_data[vif_data['feature'] != 'const']
        high_vif_features = vif_data[vif_data["VIF"] > vif_threshold]
        
        print(f"当前特征数量: {len(current_features)}")
        print(f"当前R²: {current_r2:.4f}, 调整后R²: {current_adj_r2:.4f}")
        print(f"不显著的特征 (p > {significance_level}): {insignificant_features}")
        print(f"VIF > {vif_threshold} 的特征:")
        print(high_vif_features)
        
        # 检查终止条件
        if len(insignificant_features) == 0 and len(high_vif_features) == 0:
            print("\n✅ 达到终止条件: 所有特征都显著且VIF < 10")
            break
            
        # 记录当前最佳模型（如果R²下降在容忍范围内且满足其他条件）
        r2_drop = baseline_r2 - current_r2
        if (len(insignificant_features) == 0 and 
            r2_drop <= r2_tolerance and 
            current_r2 > best_r2 - 0.01):  # 允许轻微下降
            best_r2 = current_r2
            best_features = current_features.copy()
            print(f"更新最佳模型: R² = {current_r2:.4f}, 特征数 = {len(current_features)}")
        
        # 生成所有候选移除特征列表（包括不显著和高VIF特征）
        candidate_features_to_remove = []
        
        # 收集所有需要移除的候选特征（不区分优先级）
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
                # 避免重复添加（如果某个特征既高VIF又不显著）
                if not any(candidate['feature'] == row['feature'] for candidate in candidate_features_to_remove):
                    candidate_features_to_remove.append({
                        'feature': row['feature'],
                        'reason': 'high_vif',
                        'p_value': pvalues[row['feature']] if row['feature'] in pvalues else 1.0,
                        'vif': row['VIF']
                    })
        
        # 智能选择要移除的特征 - 基于对R²的影响
        features_to_remove = []
        
        if candidate_features_to_remove:
            # 评估移除每个候选特征对R²的影响
            removal_impact = []
            for candidate in candidate_features_to_remove:
                feature = candidate['feature']
                # 测试移除该特征后的R²
                test_features = [f for f in current_features if f != feature]
                if test_features:  # 确保还有特征剩余
                    X_test = data[test_features]
                    X_test_with_const = add_constant(X_test)
                    try:
                        test_model = sm.OLS(y, X_test_with_const).fit()
                        r2_after_removal = test_model.rsquared
                        r2_drop = current_r2 - r2_after_removal
                    except:
                        r2_drop = float('inf')  # 如果拟合失败，认为影响很大
                else:
                    r2_drop = float('inf')
                
                removal_impact.append({
                    'feature': feature,
                    'reason': candidate['reason'],
                    'p_value': candidate['p_value'],
                    'vif': candidate['vif'],
                    'r2_drop': r2_drop
                })
            
            # 按R²影响从小到大排序
            removal_impact_sorted = sorted(removal_impact, key=lambda x: x['r2_drop'])
            
            # 选择对R²影响最小的特征进行移除
            best_candidate = None
            
            # 首先寻找在容忍度范围内的影响最小的特征
            for candidate in removal_impact_sorted:
                if candidate['r2_drop'] <= r2_tolerance:
                    best_candidate = candidate
                    break
            
            # 如果没有在容忍度范围内的特征，选择影响相对最小的
            if best_candidate is None and removal_impact_sorted:
                best_candidate = removal_impact_sorted[0]
                print(f"⚠️ 所有移除都会导致R²下降超过容忍度，选择影响最小的")
            
            if best_candidate:
                features_to_remove.append(best_candidate['feature'])
                print(f"移除特征: {best_candidate['feature']} ({best_candidate['reason']})")
                print(f"  p-value: {best_candidate['p_value']:.4f}, VIF: {best_candidate['vif']:.2f}, R²下降: {best_candidate['r2_drop']:.4f}")
                
                # 显示其他候选特征的R²影响（用于调试）
                if len(removal_impact_sorted) > 1:
                    print(f"  其他候选特征R²影响:")
                    for i, candidate in enumerate(removal_impact_sorted[1:4]):  # 显示前几个
                        print(f"    {candidate['feature']}: R²下降 {candidate['r2_drop']:.4f} ({candidate['reason']})")
        
        # 如果没有合适的候选特征（所有移除都会导致R²大幅下降）
        if not features_to_remove:
            print("⚠️ 没有合适的特征可以移除（所有移除都会导致R²大幅下降）")
            # 检查是否满足放宽的条件
            if len(insignificant_features) == 0:
                print("✅ 所有特征都显著，接受当前模型")
                break
            else:
                # 强制移除p值最大的特征
                if insignificant_features:
                    max_p_feature = max(insignificant_features, key=lambda x: pvalues[x])
                    features_to_remove.append(max_p_feature)
                    print(f"强制移除最不显著特征: {max_p_feature}")
        
        # 移除特征
        for feature in features_to_remove:
            if feature in current_features:
                current_features.remove(feature)
                print(f"已移除特征: {feature}")
            
        # 检查是否还有特征剩余
        if len(current_features) == 0:
            print("❌ 所有特征都被移除了！")
            current_features = best_features  # 恢复到最佳特征集
            break
            
        iteration += 1
        
        # 防止无限循环
        if iteration > 30:
            print("⚠️ 达到最大迭代次数，强制终止")
            break
    
    # 最终模型评估
    print(f"\n🎯 最终选择的特征 ({len(current_features)}个):")
    X_final = data[current_features]
    y_final = data[target_col]
    X_final_with_const = add_constant(X_final)
    final_model = sm.OLS(y_final, X_final_with_const).fit()
    
    # 计算最终指标
    final_vif_data = pd.DataFrame()
    final_vif_data["feature"] = X_final_with_const.columns
    final_vif_data["VIF"] = [variance_inflation_factor(X_final_with_const.values, i) 
                           for i in range(X_final_with_const.shape[1])]
    final_vif_data = final_vif_data[final_vif_data['feature'] != 'const']
    
    final_r2 = final_model.rsquared
    final_adj_r2 = final_model.rsquared_adj
    r2_reduction = baseline_r2 - final_r2
    
    print(f"\n📊 模型性能总结:")
    print(f"初始R²: {baseline_r2:.4f}")
    print(f"最终R²: {final_r2:.4f}")
    print(f"R²下降: {r2_reduction:.4f} ({r2_reduction/baseline_r2*100:.1f}%)")
    print(f"调整后R²: {final_adj_r2:.4f}")
    
    for feature in current_features:
        p_value = final_model.pvalues[feature]
        vif_value = final_vif_data[final_vif_data['feature'] == feature]['VIF'].values[0]
        significance_indicator = " ✓" if p_value <= significance_level else " ✗"
        vif_indicator = " ✓" if vif_value <= vif_threshold else " ✗"
        print(f"  {feature}: p-value={p_value:.4f}{significance_indicator}, VIF={vif_value:.2f}{vif_indicator}")
    
    return current_features, final_model

# 计算VIF
def calculate_vif(data, include_const=True, threshold=10):
    """
    计算数据集中每个变量的VIF值
    
    参数:
    data: pandas DataFrame, 输入数据
    include_const: bool, 是否在结果中包含常数项
    threshold: float, VIF阈值，用于标记高多重共线性
    
    返回:
    pandas DataFrame, 包含变量名和对应的VIF值
    """
    # 输入验证
    if not isinstance(data, pd.DataFrame):
        raise ValueError("输入数据必须是pandas DataFrame")
    
    if data.empty:
        raise ValueError("输入数据不能为空")
    
    # 检查是否有缺失值
    if data.isnull().any().any():
        raise ValueError("数据中包含缺失值，请先处理缺失值")
    
    # 添加常数项
    X = add_constant(data)
    
    # 计算每个变量的VIF
    vif_data = pd.DataFrame()
    vif_data['Variable'] = X.columns
    vif_data['VIF'] = [variance_inflation_factor(X.values, i) 
                      for i in range(X.shape[1])]
    
    # 添加诊断信息
    vif_data['High_VIF'] = vif_data['VIF'] > threshold
    
    # 是否排除常数项
    if not include_const:
        vif_data = vif_data[vif_data['Variable'] != 'const']
    
    # 按VIF值排序
    vif_data = vif_data.sort_values('VIF', ascending=False)
    
    return vif_data.reset_index(drop=True)

##  --------------------------------  VAT  --------------------------

VAT_feature = AA_20240105_All_models['VAT_forward_linear_model']['Selected_feature'] ## 逐步回归模型的自变量
X_VAT_train = X_train[VAT_feature]  ## 自变量值
Y_VAT_train = y_train["VATmass"]
Y_VAT_train = Y_VAT_train.to_frame().rename(columns={'VATmass': 'VATmass'})

# 合并X和Y
VAT_train_data = pd.concat([X_VAT_train, Y_VAT_train], axis=1)

## 为了可重复，进行VIF和显著性筛选
# 保留 ["标准站姿腰围", "大腿围度左边.水平线.", "对高臀部的腰部后面",   "臂长右", "上臂围度左边", "age"]

VAT_targets, VAT_forward_linear_model = feature_selection_process4(data = VAT_train_data, target_col= "VATmass", significance_level=0.05, vif_threshold=10, r2_tolerance=0.01)

X_VAT_train_Update = X_train[VAT_targets]
vif_VAT_result_Update = calculate_vif(X_VAT_train_Update, include_const= False)  ## 计算自变量的VIF
print(vif_VAT_result_Update)

vif_VAT_result_Update.to_csv("C:/Users/zjl__/Desktop/output_zhaojialu/output/ForwardRegression2/VIF_VAT_params.csv", index=False)

# 计算DW统计量
dw_VAT = durbin_watson(VAT_forward_linear_model.resid)
print(f"VAT-Durbin-Watson统计量: {dw_VAT:.4f}") # VAT-Durbin-Watson统计量: 1.8993

##  --------------------------------  FM  --------------------------

FM_feature = AA_20240105_All_models['FM_forward_linear_model']['Selected_feature']
X_FM_train = X_train[FM_feature]  ## 自变量值
Y_FM_train = y_train["FM"]
Y_FM_train = Y_FM_train.to_frame().rename(columns={'FM': 'FM'})

# 合并X和Y
FM_train_data = pd.concat([X_FM_train, Y_FM_train], axis=1)

# 为了可重复，并结合 VIF < 10 和 回归显著性
# 保留 [ "膝围度左边", "gender", "脚踝围左边", "前臂围度右边","标准站姿腰围", "age", "胯部长度", "头高", ,'腿肚围度左边', "最小腿围度左边"]

FM_targets, FM_forward_linear_model = feature_selection_process4(data = FM_train_data, target_col= "FM", significance_level=0.05, vif_threshold=10, r2_tolerance=0.01)

X_FM_train_Update = X_train[FM_targets]
vif_FM_result_Update = calculate_vif(X_FM_train_Update, include_const= False)  ## 计算自变量的VIF
vif_FM_result_Update.to_csv("C:/Users/zjl__/Desktop/output_zhaojialu/output/ForwardRegression2/VIF_FM_params.csv", index=False)
print(vif_FM_result_Update)

dw_FM = durbin_watson(FM_forward_linear_model.resid)
print(f"FM-Durbin-Watson统计量: {dw_FM:.4f}") # FM-Durbin-Watson统计量: 1.9210

## -------------------------------- FMI ----------------------------------

FMI_feature = AA_20240105_All_models['FMI_forward_linear_model']['Selected_feature']
X_FMI_train = X_train[FMI_feature]  ## 自变量值
Y_FMI_train = y_train["FMI"]
Y_FMI_train = Y_FMI_train.to_frame().rename(columns={'FMI': 'FMI'})

# 合并X和Y
FMI_train_data = pd.concat([X_FMI_train, Y_FMI_train], axis=1)

# 为了可重复，并结合 VIF < 10 和 回归显著性
# 保留 ["中臀围", "身体高度", "膝围度左边", "gender",  "age",  "脚踝围右边", "臂长左", "上臂围度右边"]

FMI_targets, FMI_forward_linear_model = feature_selection_process4(data = FMI_train_data, target_col= "FMI", significance_level=0.05, vif_threshold=10, r2_tolerance=0.01)

X_FMI_train_Update = X_train[FMI_targets]
vif_FMI_result_Update = calculate_vif(X_FMI_train_Update, include_const= False)  ## 计算自变量的VIF
vif_FMI_result_Update.to_csv("C:/Users/zjl__/Desktop/output_zhaojialu/output/ForwardRegression2/VIF_FMI_params.csv", index=False)
print(vif_FMI_result_Update)

dw_FMI = durbin_watson(FMI_forward_linear_model.resid)
print(f"FMI-Durbin-Watson统计量: {dw_FMI:.4f}") # FMI-Durbin-Watson统计量: 1.9920

##  --------------------------------  LM  --------------------------

LM_feature = AA_20240105_All_models['LM_forward_linear_model']['Selected_feature']
X_LM_train = X_train[LM_feature]  ## 自变量值
Y_LM_train = y_train["LM"]
Y_LM_train = Y_LM_train.to_frame().rename(columns={'LM': 'LM'})

# 合并X和Y
LM_train_data = pd.concat([X_LM_train, Y_LM_train], axis=1)

# 为了可重复，并结合 VIF < 10 和 回归显著性
# 保留 ["gender", "高臀部围度", "膝围度左边", "脚踝围左边",  "前臂围度右边",  "身体高度", "腿肚围度左边", "宽度腋窝", "最小腿围度左边"]

LM_targets, LM_forward_linear_model = feature_selection_process4(data = LM_train_data, target_col= "LM", significance_level=0.05, vif_threshold=10, r2_tolerance=0.01)

X_LM_train_Update = X_train[LM_targets]
vif_LM_result_Update = calculate_vif(X_LM_train_Update, include_const= False)  ## 计算自变量的VIF
vif_LM_result_Update.to_csv("C:/Users/zjl__/Desktop/output_zhaojialu/output/ForwardRegression2/VIF_LM_params.csv", index=False)
print(vif_LM_result_Update)

dw_LM = durbin_watson(LM_forward_linear_model.resid)
print(f"LM-Durbin-Watson统计量: {dw_LM:.4f}") # LM-Durbin-Watson统计量: 2.0129


##  --------------------------------  Android  --------------------------

Android_feature = AA_20240105_All_models['Android_forward_linear_model']['Selected_feature']
X_Android_train = X_train[Android_feature]  ## 自变量值
Y_Android_train = y_train["Android"]
Y_Android_train = Y_Android_train.to_frame().rename(columns={'Android': 'Android'})

# 合并X和Y
Android_train_data = pd.concat([X_Android_train, Y_Android_train], axis=1)

# 为了可重复，并结合 VIF < 10 和 回归显著性
# 保留 ["标准站姿腰围",  "gender", "前臂围度右边", "脚踝围左边",   "臀部.大腿围度",  "臂长左",  "膝围度左边", "腰部对臀部高度左边"]

Android_targets, Android_forward_linear_model = feature_selection_process4(data = Android_train_data, target_col= "Android", significance_level=0.05, vif_threshold=10, r2_tolerance=0.01)

X_Android_train_Update = X_train[Android_targets]
vif_Android_result_Update = calculate_vif(X_Android_train_Update, include_const= False)  ## 计算自变量的VIF
vif_Android_result_Update.to_csv("C:/Users/zjl__/Desktop/output_zhaojialu/output/ForwardRegression2/VIF_Android_params.csv", index=False)
print(vif_Android_result_Update)

dw_Android = durbin_watson(Android_forward_linear_model.resid)
print(f"Android-Durbin-Watson统计量: {dw_Android:.4f}")   ## Android-Durbin-Watson统计量: 1.9673

##  --------------------------------  Gynoid  --------------------------

Gynoid_feature = AA_20240105_All_models['Gynoid_forward_linear_model']['Selected_feature']
X_Gynoid_train = X_train[Gynoid_feature]  ## 自变量值
Y_Gynoid_train = y_train["Gynoid"]
Y_Gynoid_train = Y_Gynoid_train.to_frame().rename(columns={'Gynoid': 'Gynoid'})

# 合并X和Y
Gynoid_train_data = pd.concat([X_Gynoid_train, Y_Gynoid_train], axis=1)

# 为了可重复，并结合 VIF < 10 和 回归显著性
## 保留 [ "gender", "膝围度左边","高臀部围度", "age", "前臂围度右边", "大腿围度右边.水平线.", "腿肚围度右边", "宽度腋窝", "横过后面宽度.腋窝水平.", "胯长度.后面", "上臂长度左边"]

Gynoid_targets, Gynoid_forward_linear_model = feature_selection_process4(data = Gynoid_train_data, target_col= "Gynoid", significance_level=0.05, vif_threshold=10, r2_tolerance=0.01)

X_Gynoid_train_Update = X_train[Gynoid_targets]
vif_Gynoid_result_Update = calculate_vif(X_Gynoid_train_Update, include_const= False)  ## 计算自变量的VIF
vif_Gynoid_result_Update.to_csv("C:/Users/zjl__/Desktop/output_zhaojialu/output/ForwardRegression2/VIF_Gynoid_params.csv", index=False)
print(vif_Gynoid_result_Update)

dw_Gynoid = durbin_watson(Gynoid_forward_linear_model.resid)
print(f"Gynoid-Durbin-Watson统计量: {dw_Gynoid:.4f}") # Gynoid-Durbin-Watson统计量: 2.0369


##  --------------------------------  Android / Gynoid  --------------------------

A_G_feature = AA_20240105_All_models['A_G_forward_linear_model']['Selected_feature']
X_A_G_train = X_train[A_G_feature]  ## 自变量值
Y_A_G_train = y_train["A_G"]
Y_A_G_train = Y_A_G_train.to_frame().rename(columns={'A_G': 'A_G'})

# 合并X和Y
A_G_train_data = pd.concat([X_A_G_train, Y_A_G_train], axis=1)

# 为了可重复，并结合 VIF < 10 和 回归显著性
## 删除['高腰围','标准站姿腰围','Underbust圆周.水平线.','腰带.膝距离','腰带.腰的偏移量.前面.','7CV.垂直的距离','脖子前面到垂直的距离','胸部.胸围度.水平线.','脖子到横过后面宽度.腋窝水平.','上面的躯干扭...']
## 保留 ["臀部围度", "对高臀部的腰部后面", "大腿围度左边.水平线.", "脚踝围左边", "中间的脖子围度", "标准站姿腰围", "胸部.胸围度", "脖子到横过后面宽度.腋窝水平", "前臂长度左边"]

A_G_targets, A_G_forward_linear_model = feature_selection_process4(data = A_G_train_data, target_col= "A_G", significance_level=0.05, vif_threshold=10, r2_tolerance=0.01)

X_A_G_train_Update = X_train[A_G_targets]
vif_A_G_result_Update = calculate_vif(X_A_G_train_Update, include_const= False)  ## 计算自变量的VIF
vif_A_G_result_Update.to_csv("C:/Users/zjl__/Desktop/output_zhaojialu/output/ForwardRegression2/VIF_A_G__params.csv", index=False)
print(vif_A_G_result_Update)

dw_A_G = durbin_watson(A_G_forward_linear_model.resid)
print(f"A_G-Durbin-Watson统计量: {dw_A_G:.4f}") # A_G-Durbin-Watson统计量: 1.8479

##  --------------------------------  BFP  --------------------------

BFP_feature = AA_20240105_All_models['BFP_forward_linear_model']['Selected_feature']
X_BFP_train = X_train[BFP_feature]  ## 自变量值
Y_BFP_train = y_train["BFP"] * 100
Y_BFP_train = Y_BFP_train.to_frame().rename(columns={'BFP': 'BFP'})

# 合并X和Y
BFP_train_data = pd.concat([X_BFP_train, Y_BFP_train], axis=1)

# 为了可重复，并结合 VIF < 10 和 回归显著性
# 删除 ['中臀围','腹圆周','脖子对腰部中心向后地','7CV.垂直的距离','脖子前面到垂直的距离','age','中间的脖子围度','腰带.腰的偏移量.前面.','后背到垂直面距离.在胸部水平面','高腰围','脖子正确地对腰部后面','对高臀部的腰部后面','臀部围度','weight','胸部到垂直的距离','肩角度右边','横肩覆盖脖子']
# 保留["gender", "臂长左", "前臂围度右边", "膝围度左边", "脚踝围左边", "臀部.大腿围度",  "头高",  "腿肚围度左边",  "最小腿围度左边", "大腿围度右边.水平线.","宽度腋窝", "脖子左边对腰部后面", "横过后面宽度.腋窝水平.", "标准站姿腰围",  "颈到臀距离","胯长度.后面"]

BFP_targets, BFP_forward_linear_model = feature_selection_process4(data = BFP_train_data, target_col= "BFP", significance_level=0.05, vif_threshold=10, r2_tolerance=0.01)

X_BFP_train_Update = X_train[BFP_targets]
vif_BFP_result_Update = calculate_vif(X_BFP_train_Update, include_const= False)  ## 计算自变量的VIF
vif_BFP_result_Update.to_csv("C:/Users/zjl__/Desktop/output_zhaojialu/output/ForwardRegression2/VIF_BFP__params.csv", index=False)
print(vif_BFP_result_Update)

dw_BFP = durbin_watson(BFP_forward_linear_model.resid)
print(f"BFP-Durbin-Watson统计量: {dw_BFP:.4f}") # BFP-Durbin-Watson统计量: 1.9218

#########  获得模型参数和测试集模型性能 ################

def model_result(X_final, Y_final, final_model):
    params = final_model.params  ## 获取模型参数
    p_values = final_model.pvalues  ## 获取P值
    r_squared = final_model.rsquared  ## 获取R2
    y_pred = final_model.predict(X_final)  ## 获取Y的预测值
    rmse = np.sqrt(mean_squared_error(Y_final, y_pred))  ## 计算RMSE
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
    })  ## 添加R2

    rmse_row = pd.DataFrame({
        'Parameter': ['RMSE'],
        'Coefficient': [rmse],
        'P-Value': [None]
    })  ## 添加RMSE

    results = pd.concat([results, r2_row, rmse_row], ignore_index=True)

    results['Coefficient'] = results['Coefficient'].round(3)
    
    return results

X_VAT_test = X_test[VAT_targets]
X_VAT_test = add_constant(X_VAT_test)
Y_VAT_test = y_test['VATmass']
print("VATmass") # r2: 0.8058659068946479 rmse: 0.23225482234553405
VIF_VAT_params = model_result(X_VAT_test, Y_VAT_test, VAT_forward_linear_model)
VIF_VAT_params.to_csv("C:/Users/zjl__/Desktop/output_zhaojialu/output/Coefficient2/VIF_VAT_params.csv", index=False)

X_FM_test = X_test[FM_targets]
X_FM_test = add_constant(X_FM_test)
Y_FM_test = y_test['FM']
print("FM")  # r2: 0.8766785992784824 rmse: 2.1806469681696576
VIF_FM_params = model_result(X_FM_test, Y_FM_test, FM_forward_linear_model)
VIF_FM_params.to_csv("C:/Users/zjl__/Desktop/output_zhaojialu/output/Coefficient2/VIF_FM_params.csv", index=False)

X_FMI_test = X_test[FMI_targets]
X_FMI_test = add_constant(X_FMI_test)
Y_FMI_test = y_test['FMI']
print("FMI") # r2: 0.8676449370555529 rmse: 0.798729658671095
VIF_FMI_params = model_result(X_FMI_test, Y_FMI_test, FMI_forward_linear_model)
VIF_FMI_params.to_csv("C:/Users/zjl__/Desktop/output_zhaojialu/output/Coefficient2/VIF_FMI_params.csv", index=False)

X_LM_test = X_test[LM_targets]
X_LM_test = add_constant(X_LM_test)
Y_LM_test = y_test['LM']
print("LM")  # 0.9210978254059097 rmse: 2.488947923219758
VIF_LM_params = model_result(X_LM_test, Y_LM_test, LM_forward_linear_model)
VIF_LM_params.to_csv("C:/Users/zjl__/Desktop/output_zhaojialu/output/Coefficient2/VIF_LM_params.csv", index=False)

X_Android_test = X_test[Android_targets]
X_Android_test = add_constant(X_Android_test)
Y_Android_test = y_test['Android']
print("Android")  # r2: 0.8852250182118173 rmse: 0.2596830447532017
VIF_Android_params = model_result(X_Android_test, Y_Android_test, Android_forward_linear_model)
VIF_Android_params.to_csv("C:/Users/zjl__/Desktop/output_zhaojialu/output/Coefficient2/VIF_Android_params.csv", index=False)

X_Gynoid_test = X_test[Gynoid_targets]
X_Gynoid_test = add_constant(X_Gynoid_test)
Y_Gynoid_test = y_test['Gynoid']
print("Gynoid")  # 0.8660905559562025 rmse: 0.37364140066743995
VIF_Gynoid_params = model_result(X_Gynoid_test, Y_Gynoid_test, Gynoid_forward_linear_model)
VIF_Gynoid_params.to_csv("C:/Users/zjl__/Desktop/output_zhaojialu/output/Coefficient2/VIF_Gynoid_params.csv", index=False)

X_A_G_test = X_test[A_G_targets]
X_A_G_test = add_constant(X_A_G_test)
Y_A_G_test = y_test['A_G']
print("AGFMR")  # r2: 0.8438780901807932 rmse: 0.07848113372038507
VIF_A_G_params = model_result(X_A_G_test, Y_A_G_test, A_G_forward_linear_model)
VIF_A_G_params.to_csv("C:/Users/zjl__/Desktop/output_zhaojialu/output/Coefficient2/VIF_A_G_params.csv", index=False)

X_BFP_test = X_test[BFP_targets]
X_BFP_test = add_constant(X_BFP_test)
Y_BFP_test = y_test['BFP'] * 100
print("BFP")   # r2: 0.7872491935850359 rmse: 3.374100525171791
VIF_BFP_params = model_result(X_BFP_test, Y_BFP_test, BFP_forward_linear_model)
VIF_BFP_params.to_csv("C:/Users/zjl__/Desktop/output_zhaojialu/output/Coefficient2/VIF_BFP_params.csv", index=False)


## 将多个模型和参数保存到一个字典中
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

## 保存字典

dump(AA_20240105_All_VIF_models, 'C:/Users/zjl__/Desktop/output_zhaojialu/output/ForwardRegression2/AA_20240105_All_VIF_models.joblib')


####################### 比较模型在训练集和测试集的性能指标 R2，RMSE，ME(95%CI) #########################33

## 加载之前用训练集训练的多个模型及参数
AA_20240105_All_VIF_models = load('C:/Users/zjl__/Desktop/output_zhaojialu/output/ForwardRegression2/AA_20240105_All_VIF_models.joblib')


##定义计算模型性能的函数
def calculate_metrics(y_true, y_pred):
    """
    计算并返回 R², RMSE, ME 和 ME 的 95% CI
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
    se_me = stats.sem(errors) # stats.sem 直接计算标准误
    ci_low, ci_high = stats.t.interval(0.95, df=n-1, loc=me, scale=se_me)
    
    return r2, rmse, me, (ci_low, ci_high)

################################   FMI  ######################################
FMI_FSLR_VIF_model = AA_20240105_All_VIF_models['FMI_forward_linear_model']['Forward_linear_model']
FMI_Selected_feature = AA_20240105_All_VIF_models['FMI_forward_linear_model']['Selected_feature']
FMI_train_pred = FMI_FSLR_VIF_model.predict(add_constant(X_train[FMI_Selected_feature]))
FMI_test_pred = FMI_forward_linear_model.predict(add_constant(X_test[FMI_Selected_feature]))


#  对训练集和测试集分别计算指标
FMI_train_r2, FMI_train_rmse, FMI_train_me, FMI_train_ci = calculate_metrics(y_train['FMI'], FMI_train_pred)
FMI_test_r2, FMI_test_rmse, FMI_test_me, FMI_test_ci = calculate_metrics(y_test['FMI'], FMI_test_pred)

################################   A_G  ######################################
A_G_FSLR_VIF_model = AA_20240105_All_VIF_models['A_G_forward_linear_model']['Forward_linear_model']
A_G_Selected_feature = AA_20240105_All_VIF_models['A_G_forward_linear_model']['Selected_feature']
A_G_train_pred = A_G_FSLR_VIF_model.predict(add_constant(X_train[A_G_Selected_feature]))
A_G_test_pred = A_G_forward_linear_model.predict(add_constant(X_test[A_G_Selected_feature]))


#  对训练集和测试集分别计算指标
A_G_train_r2, A_G_train_rmse, A_G_train_me, A_G_train_ci = calculate_metrics(y_train['A_G'], A_G_train_pred)
A_G_test_r2, A_G_test_rmse, A_G_test_me, A_G_test_ci = calculate_metrics(y_test['A_G'], A_G_test_pred)

################################   FM  ######################################
FM_FSLR_VIF_model = AA_20240105_All_VIF_models['FM_forward_linear_model']['Forward_linear_model']
FM_Selected_feature = AA_20240105_All_VIF_models['FM_forward_linear_model']['Selected_feature']
FM_train_pred = FM_FSLR_VIF_model.predict(add_constant(X_train[FM_Selected_feature]))
FM_test_pred = FM_forward_linear_model.predict(add_constant(X_test[FM_Selected_feature]))

#  对训练集和测试集分别计算指标
FM_train_r2, FM_train_rmse, FM_train_me, FM_train_ci = calculate_metrics(y_train['FM'], FM_train_pred)
FM_test_r2, FM_test_rmse, FM_test_me, FM_test_ci = calculate_metrics(y_test['FM'], FM_test_pred)

################################   LM  ######################################
LM_FSLR_VIF_model = AA_20240105_All_VIF_models['LM_forward_linear_model']['Forward_linear_model']
LM_Selected_feature = AA_20240105_All_VIF_models['LM_forward_linear_model']['Selected_feature']
LM_train_pred = LM_FSLR_VIF_model.predict(add_constant(X_train[LM_Selected_feature]))
LM_test_pred = LM_forward_linear_model.predict(add_constant(X_test[LM_Selected_feature]))

#  对训练集和测试集分别计算指标
LM_train_r2, LM_train_rmse, LM_train_me, LM_train_ci = calculate_metrics(y_train['LM'], LM_train_pred)
LM_test_r2, LM_test_rmse, LM_test_me, LM_test_ci = calculate_metrics(y_test['LM'], LM_test_pred)

################################  VAT  ######################################
VAT_FSLR_VIF_model = AA_20240105_All_VIF_models['VAT_forward_linear_model']['Forward_linear_model']
VAT_Selected_feature = AA_20240105_All_VIF_models['VAT_forward_linear_model']['Selected_feature']
VAT_train_pred = VAT_FSLR_VIF_model.predict(add_constant(X_train[VAT_Selected_feature]))
VAT_test_pred = VAT_forward_linear_model.predict(add_constant(X_test[VAT_Selected_feature]))

#  对训练集和测试集分别计算指标
VAT_train_r2, VAT_train_rmse, VAT_train_me, VAT_train_ci = calculate_metrics(y_train['VATmass'], VAT_train_pred)
VAT_test_r2, VAT_test_rmse, VAT_test_me, VAT_test_ci = calculate_metrics(y_test['VATmass'], VAT_test_pred)

################################  Android  ######################################
Android_FSLR_VIF_model = AA_20240105_All_VIF_models['Android_forward_linear_model']['Forward_linear_model']
Android_Selected_feature = AA_20240105_All_VIF_models['Android_forward_linear_model']['Selected_feature']
Android_train_pred = Android_FSLR_VIF_model.predict(add_constant(X_train[Android_Selected_feature]))
Android_test_pred = Android_forward_linear_model.predict(add_constant(X_test[Android_Selected_feature]))

#  对训练集和测试集分别计算指标
Android_train_r2, Android_train_rmse, Android_train_me, Android_train_ci = calculate_metrics(y_train['Android'], Android_train_pred)
Android_test_r2, Android_test_rmse, Android_test_me, Android_test_ci = calculate_metrics(y_test['Android'], Android_test_pred)


################################  Gynoid  ######################################
Gynoid_FSLR_VIF_model = AA_20240105_All_VIF_models['Gynoid_forward_linear_model']['Forward_linear_model']
Gynoid_Selected_feature = AA_20240105_All_VIF_models['Gynoid_forward_linear_model']['Selected_feature']
Gynoid_train_pred = Gynoid_FSLR_VIF_model.predict(add_constant(X_train[Gynoid_Selected_feature]))
Gynoid_test_pred = Gynoid_forward_linear_model.predict(add_constant(X_test[Gynoid_Selected_feature]))

#  对训练集和测试集分别计算指标
Gynoid_train_r2, Gynoid_train_rmse, Gynoid_train_me, Gynoid_train_ci = calculate_metrics(y_train['Gynoid'], Gynoid_train_pred)
Gynoid_test_r2, Gynoid_test_rmse, Gynoid_test_me, Gynoid_test_ci = calculate_metrics(y_test['Gynoid'], Gynoid_test_pred)

################################  BFP ######################################
BFP_FSLR_VIF_model = AA_20240105_All_VIF_models['BFP_forward_linear_model']['Forward_linear_model']
BFP_Selected_feature = AA_20240105_All_VIF_models['BFP_forward_linear_model']['Selected_feature']
BFP_train_pred = BFP_FSLR_VIF_model.predict(add_constant(X_train[BFP_Selected_feature]))
BFP_test_pred = BFP_forward_linear_model.predict(add_constant(X_test[BFP_Selected_feature]))

#  对训练集和测试集分别计算指标
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

# 将字典转换为DataFrame
df = pd.DataFrame(df)
df.to_csv("C:/Users/zjl__/Desktop/output_zhaojialu/output/ForwardRegression2/Model_evaluate_indices.csv", index=False)


