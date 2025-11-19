################################# Long-to-Wide Format Data Transformation ##########################
# Load necessary packages
library(tidyr)
library(dplyr)
library(broom)
library(tidyverse)

# Assume your dataframe is named df
# If your data is read from a file, please ensure column names are correct first

df <- read.csv("YourPath/RegressionModel_Comparision_Coefficient/VAT_model_results.csv", header = TRUE)

# Enhanced version: Process complete dataset
# Assume your complete dataframe is named df_full

# Step 2: Define model order
model_order <- c("FSLR", "Lasso", "Ridge", "ElasticNet", "RF", "GBDT", "XGBoost")

df_renamed_full <- df %>%
  mutate(
    model = case_when(
      model == "Linear" ~ "FSLR",
      model == "RandomForest" ~ "RF",
      # Add other renaming rules as needed
      TRUE ~ model
    )
  ) %>%
  # Ensure model factor order
  mutate(model = factor(model, levels = model_order))

# Convert to wide format
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

print("Complete sorted R² results:")
print(r2_wide_ordered)

print("Complete sorted RMSE results:")
print(rmse_wide_ordered)

write.csv(r2_wide_ordered, "YourPath/RegressionModel_Comparision_Coefficient/origin/VAT_r2.csv", row.names = FALSE)
write.csv(rmse_wide_ordered, "YourPath/RegressionModel_Comparision_Coefficient/origin/VAT_rmse.csv", row.names = FALSE)

######################  Wilcoxon Rank Sum Test ###########################################

# Using VAT as example

df <- read.csv("YourPath/RegressionModel_Comparision_Coefficient/origin/VAT_rmse.csv", header = TRUE)

# Take square root of FSLR's MSE, directly replace FSLR column
df$FSLR <- sqrt(df$FSLR)

write.csv(df, "YourPath/RegressionModel_Comparision_Coefficient/origin/VAT_rmse.csv", row.names = FALSE)

# Perform Wilcoxon rank sum test (independent samples)
models <- c("Lasso", "Ridge", "ElasticNet", "RF", "GBDT", "XGBoost")

# Create an empty list to store results
results_ranksum <- list()

for(model in models) {
  test_result <- wilcox.test(df$FSLR, df[[model]], 
                             paired = FALSE,   # Key: use rank sum test
                             alternative = "two.sided", 
                             conf.int = TRUE)
  
  # Organize results into dataframe
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

# Combine all results
final_results_ranksum <- bind_rows(results_ranksum)

# Rename columns for better readability
final_results_ranksum <- final_results_ranksum %>%
  rename(
    Statistic = statistic,
    P_Value = p.value,
    Method = method,
    Alternative = alternative,
    CI_Low = conf.low,
    CI_High = conf.high
  )

# Calculate Bonferroni correction
k <- length(models)  # Number of comparisons = 6
bonferroni_threshold <- 0.05 / k

final_results_ranksum <- final_results_ranksum %>%
  mutate(
    # Bonferroni corrected p-value
    P_Value_Bonferroni = p.adjust(P_Value, method = "bonferroni"),
    
    # Original significance (threshold 0.05)
    Significance_Original = case_when(
      P_Value < 0.001 ~ "***",
      P_Value < 0.01 ~ "**",
      P_Value < 0.05 ~ "*",
      TRUE ~ "ns"
    ),
    
    # Bonferroni corrected significance (threshold 0.05/6)
    Significance_Bonferroni = case_when(
      P_Value_Bonferroni < 0.001 ~ "***",
      P_Value_Bonferroni < 0.01 ~ "**",
      P_Value_Bonferroni < bonferroni_threshold ~ "*",
      TRUE ~ "ns"
    ),
    
    # Significance explanation
    Significant_Original = P_Value < 0.05,
    Significant_Bonferroni = P_Value_Bonferroni < 0.05
  )

# View result comparison
comparison_table <- final_results_ranksum %>%
  select(Model, P_Value, P_Value_Bonferroni, 
         Significance_Original, Significance_Bonferroni,
         Significant_Original, Significant_Bonferroni)

print("Original threshold vs Bonferroni corrected threshold comparison:")
print(comparison_table)

# Display threshold information
cat("\n")
cat("Number of comparisons (k):", k, "\n")
cat("Original significance threshold: 0.05\n")
cat("Bonferroni corrected threshold:", round(bonferroni_threshold, 5), "\n")
cat("Bonferroni corrected threshold (approximate): 0.00833\n")

# View results
print("Wilcoxon rank sum test results (each model vs FSLR):")
print(final_results_ranksum)

# Save results
write.csv(final_results_ranksum, "YourPath/RegressionModel_Comparision_Coefficient/Wilcoxon_RankSum/VAT_vs_FSLR_RMSE.csv", row.names = FALSE)