
#################################长宽格式数据转换 ##########################
# 加载必要的包
library(tidyr)
library(dplyr)
library(broom)
library(tidyverse)

# 假设你的数据框名为 df
# 如果你的数据是从文件读取的，请先确保列名正确

df <- read.csv("C:/Users/zjl__/Desktop/output_zhaojialu/output/RegressionModel_Comparision_Coefficient/VAT_model_results.csv", header = TRUE)

# 增强版：处理完整数据集
# 假设你的完整数据框名为 df_full

# 步骤2：定义模型顺序
model_order <- c("FSLR", "Lasso", "Ridge", "ElasticNet", "RF", "GBDT", "XGBoost")

df_renamed_full <- df %>%
  mutate(
    model = case_when(
      model == "Linear" ~ "FSLR",
      model == "RandomForest" ~ "RF",
      # 可以根据需要添加其他重命名规则
      TRUE ~ model
    )
  ) %>%
  # 确保模型因子顺序
  mutate(model = factor(model, levels = model_order))

# 转换为宽格式
r2_wide_ordered <- df_renamed_full %>%
  select(fold, model, r2) %>%
  pivot_wider(
    names_from = model,
    values_from = r2
  ) %>%
  select(fold, all_of(model_order))

rmse_wide_ordered <- df_renamed_full %>%
  select(fold, model, rmse) %>%
  pivot_wider(
    names_from = model,
    values_from = rmse
  ) %>%
  select(fold, all_of(model_order))

print("完整排序的R²结果：")
print(r2_wide_ordered)

print("完整排序的RMSE结果：")
print(rmse_wide_ordered)

write.csv(r2_wide_ordered, "C:/Users/zjl__/Desktop/output_zhaojialu/output/RegressionModel_Comparision_Coefficient/origin/VAT_r2.csv", row.names = FALSE)
write.csv(rmse_wide_ordered, "C:/Users/zjl__/Desktop/output_zhaojialu/output/RegressionModel_Comparision_Coefficient/origin/VAT_rmse.csv", row.names = FALSE)

######################  Wilcoxon 秩和检验 ###########################################


# 以VAT为例

df <- read.csv("C:/Users/zjl__/Desktop/output_zhaojialu/output/RegressionModel_Comparision_Coefficient/origin/VAT_rmse.csv", header = TRUE)

# FSLR的MSE进行开平方根，直接替换FSLR列
df$FSLR <- sqrt(df$FSLR)

write.csv(df, "C:/Users/zjl__/Desktop/output_zhaojialu/output/RegressionModel_Comparision_Coefficient/origin/VAT_rmse.csv", row.names = FALSE)


# 进行Wilcoxon秩和检验（独立样本）
models <- c("Lasso", "Ridge", "ElasticNet", "RF", "GBDT", "XGBoost")

# 创建一个空列表来存储结果
results_ranksum <- list()

for(model in models) {
  test_result <- wilcox.test(df$FSLR, df[[model]], 
                             paired = FALSE,   # 关键：使用秩和检验
                             alternative = "two.sided", 
                             conf.int = TRUE)
  
  # 将结果整理成数据框
  results_ranksum[[model]] <- tidy(test_result) %>%
    mutate(
      Model = model,
      FSLR_Median = median(df$FSLR),
      Comparison_Median = median(df[[model]]),
      FSLR_Mean = mean(df$FSLR),
      Comparison_Mean = mean(df[[model]]),
      Test_Type = "Wilcoxon_Rank_Sum_Test"
    ) %>%
    select(Model, Test_Type, everything())
}

# 合并所有结果
final_results_ranksum <- bind_rows(results_ranksum)

# 重命名列使其更易读
final_results_ranksum <- final_results_ranksum %>%
  rename(
    Statistic = statistic,
    P_Value = p.value,
    Method = method,
    Alternative = alternative,
    CI_Low = conf.low,
    CI_High = conf.high
  )

# 计算Bonferroni校正
k <- length(models)  # 比较次数 = 6
bonferroni_threshold <- 0.05 / k

final_results_ranksum <- final_results_ranksum %>%
  mutate(
    # Bonferroni校正的p值
    P_Value_Bonferroni = p.adjust(P_Value, method = "bonferroni"),
    
    # 原始显著性（阈值0.05）
    Significance_Original = case_when(
      P_Value < 0.001 ~ "***",
      P_Value < 0.01 ~ "**",
      P_Value < 0.05 ~ "*",
      TRUE ~ "ns"
    ),
    
    # Bonferroni校正后的显著性（阈值0.05/6）
    Significance_Bonferroni = case_when(
      P_Value_Bonferroni < 0.001 ~ "***",
      P_Value_Bonferroni < 0.01 ~ "**",
      P_Value_Bonferroni < bonferroni_threshold ~ "*",
      TRUE ~ "ns"
    ),
    
    # 显著性说明
    Significant_Original = P_Value < 0.05,
    Significant_Bonferroni = P_Value_Bonferroni < 0.05
  )

# 查看结果对比
comparison_table <- final_results_ranksum %>%
  select(Model, P_Value, P_Value_Bonferroni, 
         Significance_Original, Significance_Bonferroni,
         Significant_Original, Significant_Bonferroni)

print("原始阈值 vs Bonferroni校正阈值对比：")
print(comparison_table)

# 显示阈值信息
cat("\n")
cat("比较次数 (k):", k, "\n")
cat("原始显著性阈值: 0.05\n")
cat("Bonferroni校正阈值:", round(bonferroni_threshold, 5), "\n")
cat("Bonferroni校正阈值 (近似): 0.00833\n")

# 查看结果
print("Wilcoxon秩和检验结果（各模型 vs FSLR）：")
print(final_results_ranksum)

# 保存结果
write.csv(final_results_ranksum, "C:/Users/zjl__/Desktop/output_zhaojialu/output/RegressionModel_Comparision_Coefficient/Wilcoxon_RankSum/VAT_vs_FSLR_RMSE.csv", row.names = FALSE)
