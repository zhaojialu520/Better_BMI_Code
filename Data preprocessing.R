# Data Preprocessing Script with Stratified Sampling

# Load required packages
library(tidyverse)
library(caret)
library(readr)

#############################
# Configuration Parameters
#############################

# File paths
input_file <- "AA_20240105.csv"  # Replace with your actual file path
output_dir <- "YourPath/output"  # Replace with your output directory

# Create output directory if it doesn't exist
if (!dir.exists(output_dir)) {
  dir.create(output_dir, recursive = TRUE)
  cat("Created output directory:", output_dir, "\n")
}

# Parameters
missing_threshold <- 0.05  # 5% missingness threshold
train_ratio <- 0.8         # 80% training, 20% testing
set_seed <- 123            # For reproducible results

#############################
# Step 1: Load and Explore Data
#############################

cat("Step 1: Loading data...\n")

# Read the original data
data <- read.csv(input_file, stringsAsFactors = FALSE)

cat("Original data dimensions:", dim(data), "\n")

#############################
# Step 2: Identify Column Types and Demographics
#############################

cat("\nStep 2: Identifying column types...\n")

# Define patterns to identify different types of columns
anthropometric_patterns <- colnames(data)[1:151]

dxa_patterns <- c(
  "VAT", "FM", "FMI", "LM", "Android", "Gynoid", "A_G", "BFP"
)

# Function to identify columns by pattern
identify_columns_by_pattern <- function(column_names, patterns) {
  matched_columns <- character(0)
  for (pattern in patterns) {
    matches <- grep(pattern, column_names, ignore.case = TRUE, value = TRUE)
    matched_columns <- c(matched_columns, matches)
  }
  return(unique(matched_columns))
}

# Identify columns
anthropometric_cols <- identify_columns_by_pattern(colnames(data), anthropometric_patterns)
dxa_cols <- identify_columns_by_pattern(colnames(data), dxa_patterns)

# Identify demographic columns (sex and age)
demographic_patterns <- c("sex", "age", "weight", "height")
demographic_cols <- identify_columns_by_pattern(colnames(data), demographic_patterns)

cat("Identified demographic columns:", demographic_cols, "\n")

#############################
# Step 3: Handle Missing Values
#############################

cat("\nStep 3: Handling missing values...\n")

# Calculate missing percentage for each sample
missing_percentage <- apply(data, 1, function(x) sum(is.na(x)) / length(x))

# Remove samples with more than 5% missing values
data_clean <- data[missing_percentage <= missing_threshold, ]

cat("After cleaning:", nrow(data_clean), "samples remaining\n")

#############################
# Step 4: Prepare for Stratified Sampling
#############################

cat("\nStep 4: Preparing for stratified sampling...\n")

# Check if sex and age columns exist
if (length(demographic_cols) < 2) {
  cat("Warning: Sex and/or age columns not found. Using random split instead.\n")
  # Fallback to random split
  set.seed(set_seed)
  train_indices <- createDataPartition(
    y = 1:nrow(data_clean), 
    p = train_ratio, 
    list = FALSE
  )
} else {
  # Extract sex and age for stratification
  sex_col <- demographic_cols[grep("sex", demographic_cols, ignore.case = TRUE)[1]]
  age_col <- demographic_cols[grep("age", demographic_cols, ignore.case = TRUE)[1]]
  
  cat("Using columns for stratification:\n")
  cat("  Sex column:", sex_col, "\n")
  cat("  Age column:", age_col, "\n")
  
  # Create age groups for stratification
  data_clean <- data_clean %>%
    mutate(
      # Ensure sex is factor
      sex_factor = as.factor(.[[sex_col]]),
      # Create age groups (you can adjust the breaks as needed)
      age_group = cut(.[[age_col]], 
                      breaks = c(0, 30, 45, 60, 100),
                      labels = c("18-30", "31-45", "46-60", "60+"),
                      include.lowest = TRUE)
    )
  
  # Create stratification variable: combination of sex and age group
  data_clean$strata <- interaction(data_clean$sex_factor, data_clean$age_group, drop = TRUE)
  
  # Check strata distribution
  cat("\nStrata distribution in full dataset:\n")
  print(table(data_clean$strata))
  
  #############################
  # Step 5: Stratified Train-Test Split
  #############################
  
  cat("\nStep 5: Performing stratified train-test split...\n")
  
  set.seed(set_seed)
  
  # Method 1: Using createDataPartition with strata
  train_indices <- createDataPartition(
    y = data_clean$strata,
    p = train_ratio,
    list = FALSE
  )
}

# Split the data
train_data <- data_clean[train_indices, ]
test_data <- data_clean[-train_indices, ]

cat("Training set size:", nrow(train_data), "\n")
cat("Test set size:", nrow(test_data), "\n")

#############################
# Step 6: Validate Stratification
#############################

cat("\nStep 6: Validating stratification balance...\n")

if (exists("sex_col") && exists("age_col")) {
  # Compare sex distribution
  cat("Sex distribution:\n")
  sex_train <- table(train_data[[sex_col]])
  sex_test <- table(test_data[[sex_col]])
  sex_compare <- cbind(Training = sex_train/sum(sex_train),
                       Test = sex_test/sum(sex_test))
  print(round(sex_compare, 3))
  
  # Compare age distribution
  cat("\nAge distribution summary:\n")
  age_summary <- data.frame(
    Dataset = c("Training", "Test"),
    Mean = c(mean(train_data[[age_col]], na.rm = TRUE), 
             mean(test_data[[age_col]], na.rm = TRUE)),
    SD = c(sd(train_data[[age_col]], na.rm = TRUE), 
           sd(test_data[[age_col]], na.rm = TRUE)),
    Min = c(min(train_data[[age_col]], na.rm = TRUE), 
            min(test_data[[age_col]], na.rm = TRUE)),
    Max = c(max(train_data[[age_col]], na.rm = TRUE), 
            max(test_data[[age_col]], na.rm = TRUE))
  )
  print(age_summary)
  
  # Statistical test for age difference
  age_test <- t.test(train_data[[age_col]], test_data[[age_col]])
  cat("\nT-test for age difference (p-value):", round(age_test$p.value, 4), "\n")
  
  # Chi-square test for sex distribution
  sex_table <- table(
    c(rep("Train", nrow(train_data)), rep("Test", nrow(test_data))),
    c(train_data[[sex_col]], test_data[[sex_col]])
  )
  sex_test <- chisq.test(sex_table)
  cat("Chi-square test for sex distribution (p-value):", round(sex_test$p.value, 4), "\n")
}

#############################
# Step 7: Extract Features and Targets
#############################

cat("\nStep 7: Extracting features and targets...\n")

# Remove stratification columns before exporting
cols_to_remove <- c("sex_factor", "age_group", "strata")
export_cols <- setdiff(colnames(data_clean), cols_to_remove)

# For training set
X_train <- train_data %>% 
  select(all_of(export_cols)) %>%
  select(any_of(anthropometric_cols)) %>%
  select(where(~ sum(!is.na(.)) > 0))

y_train <- train_data %>% 
  select(all_of(export_cols)) %>%
  select(any_of(dxa_cols)) %>%
  select(where(~ sum(!is.na(.)) > 0))

# For test set
X_test <- test_data %>% 
  select(all_of(export_cols)) %>%
  select(any_of(anthropometric_cols)) %>%
  select(where(~ sum(!is.na(.)) > 0))

y_test <- test_data %>% 
  select(all_of(export_cols)) %>%
  select(any_of(dxa_cols)) %>%
  select(where(~ sum(!is.na(.)) > 0))

cat("Training features (X_train) dimensions:", dim(X_train), "\n")
cat("Training targets (y_train) dimensions:", dim(y_train), "\n")
cat("Test features (X_test) dimensions:", dim(X_test), "\n")
cat("Test targets (y_test) dimensions:", dim(y_test), "\n")

#############################
# Step 8: Export Files
#############################

cat("\nStep 8: Exporting files...\n")

# Export training set
write.csv(X_train, file.path(output_dir, "X_train_All3D_NoScaled.csv"), row.names = FALSE)
write.csv(y_train, file.path(output_dir, "y_train_All3D_NoScaled.csv"), row.names = FALSE)

# Export test set
write.csv(X_test, file.path(output_dir, "X_test_All3D_NoScaled.csv"), row.names = FALSE)
write.csv(y_test, file.path(output_dir, "y_test_All3D_NoScaled.csv"), row.names = FALSE)

# Also export the demographic information for reference
if (exists("sex_col") && exists("age_col")) {
  demo_train <- train_data %>% select(any_of(c(sex_col, age_col)))
  demo_test <- test_data %>% select(any_of(c(sex_col, age_col)))
  
  write.csv(demo_train, file.path(output_dir, "demographic_train.csv"), row.names = FALSE)
  write.csv(demo_test, file.path(output_dir, "demographic_test.csv"), row.names = FALSE)
}

cat("Files exported successfully to:", output_dir, "\n")

#############################
# Step 9: Summary Report
#############################

cat("\n" + strrep("=", 60) + "\n")
cat("STRATIFIED PREPROCESSING SUMMARY REPORT\n")
cat(strrep("=", 60) + "\n")

cat("Original dataset:", nrow(data), "samples\n")
cat("After cleaning:", nrow(data_clean), "samples\n")
cat("Training set:", nrow(train_data), "samples (", round(nrow(train_data)/nrow(data_clean)*100, 1), "%)\n")
cat("Test set:", nrow(test_data), "samples (", round(nrow(test_data)/nrow(data_clean)*100, 1), "%)\n")

if (exists("sex_col") && exists("age_col")) {
  cat("\nStratification Results:\n")
  cat("Sex balance maintained: p-value =", round(sex_test$p.value, 4), "\n")
  cat("Age balance maintained: p-value =", round(age_test$p.value, 4), "\n")
  
  if (sex_test$p.value > 0.05 && age_test$p.value > 0.05) {
    cat("✓ SUCCESS: Training and test sets are well balanced for sex and age!\n")
  } else {
    cat("⚠ WARNING: Some imbalance detected in sex or age distribution\n")
  }
}

cat("\nFeature dimensions:\n")
cat("Anthropometric features:", ncol(X_train), "columns\n")
cat("DXA targets:", ncol(y_train), "columns\n")

cat("\nPreprocessing completed successfully!\n")