# First install and load necessary packages
install.packages("nortest")  # For Anderson-Darling test
install.packages("tseries")   # For Jarque-Bera test
library(tidyverse)
library(ggplot2)
library(patchwork)
library(moments)
library(nortest)
library(tseries)
library(lmtest)
library(sandwich)
library(car)
library(broom)

## Import data
# Read training set features
X_train <- read.csv("YourPath/X_train_All3D_NoScaled.csv", 
                    row.names = 1)  ## Use first column as row names

## Process column names
name <- colnames(X_train)    ## Extract column names
name <- gsub('人体外观测量.三维人体扫描分析系统.', '', name)    ## Remove specific prefix
name <- gsub('.cm.', '', name)    ## Remove ".cm." part
colnames(X_train) <- name   ## Replace column names with simplified names

# print(X_train)
X_train <- X_train[, !colnames(X_train) %in% c('height', 'Waist_To_Hip_Ratio.x.100.')]
print(X_train)

# Read test set features
X_test <- read.csv("YourPath/X_test_All3D_NoScaled.csv", 
                   row.names = 1)  ## Use first column as row names
colnames(X_test) <- name   ## Replace column names with simplified names
X_test <- X_test[, !colnames(X_test) %in% c('height', 'Waist_To_Hip_Ratio.x.100.')]
# print(X_test)

X_All <- rbind(X_train, X_test)

# Read training set labels
y_train <- read.csv("YourPath/y_train_All3D_NoScaled.csv", 
                    row.names = 1)  ## Use first column as row names

# Read test set labels
y_test <- read.csv("YourPath/y_test_All3D_NoScaled.csv", 
                   row.names = 1)  ## Use first column as row names

########################## Independent Variable Normality Test ###################################

VAT_targets <- c("标准站姿腰围", "大腿围度左边.水平线.", "对高臀部的腰部后面",   "臂长右", "age", "gender")
FM_targets <- c("膝围度左边", "gender", "脚踝围左边", "前臂围度右边", "标准站姿腰围", "age", "胯部长度", "头高",'腿肚围度左边', "最小腿围度左边")
FMI_targets <- c("中臀围", "身体高度", "膝围度左边", "gender",  "age",  "脚踝围右边", "臂长左", "上臂围度右边")  ##  上臂围度右边
LM_targets <- c("gender", "高臀部围度", "膝围度左边", "脚踝围左边",  "前臂围度右边",  "身体高度", "腿肚围度左边", "宽度腋窝", "最小腿围度左边")
Android_targets <- c("标准站姿腰围",  "gender", "前臂围度右边", "脚踝围左边",   "臀部.大腿围度",  "臂长左",  "膝围度左边", "腰部对臀部高度左边")
Gynoid_targets <- c( "gender", "膝围度左边","高臀部围度", "age", "前臂围度右边", "大腿围度右边.水平线.", "腿肚围度右边", "宽度腋窝", "横过后面宽度.腋窝水平.", "胯长度.后面", "上臂长度左边")
A_G_targets <- c("臀部围度", "对高臀部的腰部后面", "大腿围度左边.水平线.", "脚踝围左边", "中间的脖子围度", "标准站姿腰围", "胸部.胸围度", "脖子到横过后面宽度.腋窝水平.", "前臂长度左边")  # 脖子到横过后面宽度.腋窝水平. 
BFP_targets <- c("gender", "臂长左", "前臂围度右边", "膝围度左边", "脚踝围左边", "臀部.大腿围度",  "头高",  "腿肚围度左边", 
                 "最小腿围度左边", "大腿围度右边.水平线.","宽度腋窝", "脖子左边对腰部后面", "横过后面宽度.腋窝水平.", "标准站姿腰围", 
                 "颈到臀距离","胯长度.后面")

outcomes <- colnames(y_train)

# Combine all variable lists into one vector
all_targets <- c(outcomes, VAT_targets, FM_targets, FMI_targets, LM_targets, 
                 Android_targets, Gynoid_targets, A_G_targets, BFP_targets)

# Remove duplicate variable names
unique_targets <- unique(all_targets)

# View results
print(unique_targets)
cat("Total of", length(unique_targets), "unique variables after merging\n")

normality_dir <- "YourPath/Normality_Plot"

# Create a variable name mapping data frame
name_mapping <- data.frame(
  Chinese = c(
    "标准站姿腰围", "大腿围度左边.水平线.", "对高臀部的腰部后面", "臂长右",
    "膝围度左边", "脚踝围左边", "前臂围度右边", "胯部长度", "头高", 
    "腿肚围度左边", "最小腿围度左边", "中臀围", "身体高度", "脚踝围右边",
    "臂长左", "上臂围度右边", "高臀部围度", "宽度腋窝", "臀部.大腿围度",
    "腰部对臀部高度左边", "大腿围度右边.水平线.", "腿肚围度右边", 
    "横过后面宽度.腋窝水平.", "胯长度.后面", "上臂长度左边", "臀部围度",
    "中间的脖子围度", "胸部.胸围度", "脖子到横过后面宽度.腋窝水平.",
    "前臂长度左边", "脖子左边对腰部后面", "颈到臀距离"
  ),
  English = c(
    "Waist girth (cm)", "Thigh girth left (horizontal, cm)", 
    "Waist to high hip back (cm)", "Arm length right (cm)", "Knee girth left (cm)",
    "Ankle girth left (cm)", "Forearm girth right (cm)", "Crotch length (cm)",
    "Head height (cm)", "Calf girth left (cm)", "Min leg girth left (cm)",
    "Buttock girth (cm)", "Body height (cm)", "Ankle girth right (cm)",
    "Arm length left (cm)", "Upper arm girth right (cm)", "High hip grith (cm)",
    "Width armpits (cm)", "Hip/thigh girth (cm)", "Waist to buttock left (cm)",
    "Thigh girth right (horizontal, cm)", "Calf girth right (cm)",
    "Across back width (armpit level, cm)", "Crotch length rear (cm)", "Upper arm length left (cm)",
    "Hip girth (cm)", "Mid neck girth (cm)", "Bust/chest girth (cm)",
    "Neck to across back width (armpit level, cm)", "Forearm length left (cm)",
    "Neck left to waist back (cm)", "Distance neck to buttock (cm)"
  )
)

## Normality test
variable_normality3 <- map(unique_targets, function(chinese_var) {
  
  # Find corresponding English name
  english_var <- name_mapping$English[name_mapping$Chinese == chinese_var]
  if (length(english_var) == 0) {
    english_var <- chinese_var  # If no mapping found, use original name
  }
  
  df <- cbind(X_train, y_train)
  
  # Use Chinese variable name to extract data
  current_data <- na.omit(df[[chinese_var]])
  
  # Calculate key graphical judgment indicators
  skew_val <- skewness(current_data)
  kurt_val <- kurtosis(current_data)
  
  # Calculate Q-Q plot correlation coefficient
  qq_norm <- qqnorm(current_data, plot.it = FALSE)
  qq_correlation <- cor(qq_norm$x, qq_norm$y)
  
  # Kolmogorov-Smirnov test
  ks_test <- ks.test(current_data, "pnorm", mean = mean(current_data), sd = sd(current_data))
  ks_statistic <- ks_test$statistic
  ks_pvalue <- ks_test$p.value
  
  # Comprehensive judgment based on graphs and descriptive statistics
  if (qq_correlation > 0.995 & abs(skew_val) < 0.3 & kurt_val > 2.7 & kurt_val < 3.3) {
    judgment <- "Excellent Normality"
    judgment_color <- "darkgreen"
  } else if (qq_correlation > 0.99 & abs(skew_val) < 0.5 & kurt_val > 2.5 & kurt_val < 3.5) {
    judgment <- "Good Normality"
    judgment_color <- "green"
  } else if (qq_correlation > 0.98 & abs(skew_val) < 1 & kurt_val > 2 & kurt_val < 4) {
    judgment <- "Acceptable for Large Sample"
    judgment_color <- "orange"
  } else if (qq_correlation > 0.95 & abs(skew_val) < 2 & kurt_val > 1 & kurt_val < 5) {
    judgment <- "Mild Deviation"
    judgment_color <- "darkorange"
  } else {
    judgment <- "Substantially Non-Normal"
    judgment_color <- "red"
  }
  
  # Create histogram - display using English name
  p_hist <- ggplot(data.frame(x = current_data), aes(x = x)) +
    geom_histogram(aes(y = after_stat(density)), 
                   bins = min(30, length(current_data)/10), 
                   fill = "lightblue", color = "black", alpha = 0.7) +
    geom_density(color = "blue", linewidth = 0.8, alpha = 0.5) +
    stat_function(fun = dnorm, 
                  args = list(mean = mean(current_data), sd = sd(current_data)),
                  color = "red", linewidth = 1.2) +
    labs( subtitle = paste("Skewness:", round(skew_val, 3), 
                           "| Kurtosis:", round(kurt_val, 3)),
          x = english_var, y = "Density") +
    theme_minimal() +
    theme(plot.subtitle = element_text(size = 10, face = "bold"))
  
  # Create Q-Q plot - display using English name
  p_qq <- ggplot(data.frame(x = current_data), aes(sample = x)) +
    stat_qq(size = 1, alpha = 0.6) +
    stat_qq_line(color = "red", linewidth = 1) +
    labs( subtitle = paste("Q-Q Correlation:", round(qq_correlation, 4)),
          x = "Theoretical Quantiles", y = "Sample Quantiles") +
    theme_minimal() +
    theme(plot.subtitle = element_text(size = 10, face = "bold"))
  
  # Combine plots
  combined_plot <- (p_hist + p_qq) + 
    plot_layout(ncol = 2) +
    plot_annotation(
      title = paste( english_var),
      # subtitle = paste( "Kolmogorov-Smirnov p-value:", round(ks_pvalue, 4)),
      theme = theme(
        plot.title = element_text(hjust = 0.5, face = "bold", size = 16),
        plot.subtitle = element_text(hjust = 0.5, size = 12, color = judgment_color)
      )
    )
  
  # Generate safe filename (using English name)
  safe_var_name <- gsub("[^[:alnum:]]", "_", english_var)
  file_name <- paste0("normality_", safe_var_name, ".png")
  file_path <- file.path(normality_dir, file_name)
  
  # Save image
  ggsave(filename = file_path, 
         plot = combined_plot, 
         width = 6, 
         height = 3, 
         dpi = 600,
         bg = "white")
  
  cat("Saved:", file_path, "-", judgment, "\n")
  cat("Chinese variable:", chinese_var, "-> English display:", english_var, "\n")
  
  # Display image in R
  print(combined_plot)
  
  # Return results (including Chinese-English comparison)
  return(tibble(
    Chinese_Variable = chinese_var,
    English_Variable = english_var,
    Sample_Size = length(current_data),
    Skewness = round(skew_val, 4),
    Kurtosis = round(kurt_val, 4),
    QQ_Correlation = round(qq_correlation, 4),
    Graphical_Judgment = judgment,
    KS_Statistic = round(ks_statistic, 4),
    KS_PValue = round(ks_pvalue, 4),
    Mean = round(mean(current_data), 4),
    SD = round(sd(current_data), 4),
    File_Path = file_path
  ))
})

normality_results <- bind_rows(variable_normality3)

# Save as CSV file
write_csv(normality_results, "YourPath/Normality_Plot/variable_normality_results.csv")

###  Plot gender bar chart

# Calculate percentages
gender_counts <- X_train %>%
  count(gender) %>%
  mutate(percentage = n / sum(n) * 100) %>%
  # Convert gender to meaningful labels
  mutate(gender_label = factor(gender, 
                               levels = c(0, 1),
                               labels = c("Female", "Male")))

# Create plot and save
gender_plot <- ggplot(gender_counts, aes(x = gender_label, y = n)) +
  geom_col(fill = c("pink", "lightblue"), alpha = 0.7) +
  geom_text(aes(label = paste0(n, "\n(", round(percentage, 1), "%)")), 
            vjust = -0.5, size = 4) +
  labs(title = "Sex Distribution",
       x = "Sex", 
       y = "Frequency") +
  theme_minimal() +
  ylim(0, 700)

# Save image
ggsave("YourPath/Normality_Plot/gender_distribution.png", 
       plot = gender_plot,
       width = 8, 
       height = 6, 
       dpi = 600)

###########  VAT

VAT_targets <- c("标准站姿腰围", "大腿围度左边.水平线.", "对高臀部的腰部后面",   "臂长右", "age", "gender")

X_VAT_train <- X_train[, VAT_targets]

VAT_data <- cbind(y_train[,"VATmass"], X_VAT_train)
colnames(VAT_data)[1] <- "VAT"

VAT_data_nozero <- VAT_data[VAT_data$VAT != 0, ]

# Build regression model
VAT_FSLR <- lm(VAT ~ .+ age : 标准站姿腰围 + gender : 标准站姿腰围, data = VAT_data)

summary(VAT_FSLR) ## Multiple R-squared:  0.8477,	Adjusted R-squared:  0.8466

#########   FM

FM_targets <- c("膝围度左边", "gender", "脚踝围左边", "前臂围度右边", "标准站姿腰围", "age", "胯部长度", "头高",'腿肚围度左边', "最小腿围度左边")

X_FM_train <- X_train[, FM_targets]

FM_data <- cbind(y_train[,"FM"], X_FM_train)
colnames(FM_data)[1] <- "FM"

summary(FM_data$FM)

# Build regression model
FM_FSLR <- lm(FM ~ ., data = FM_data)

########   FMI

FMI_targets <- c("中臀围", "身体高度", "膝围度左边", "gender",  "age",  "脚踝围右边", "臂长左", "上臂围度右边")

X_FMI_train <- X_train[, FMI_targets]

FMI_data <- cbind(y_train[,"FMI"], X_FMI_train)
colnames(FMI_data)[1] <- "FMI"

summary(FMI_data$FMI)

# Build regression model
FMI_FSLR <- lm(FMI ~ ., data = FMI_data)

##################  LM  #########################

LM_targets <- c("gender", "高臀部围度", "膝围度左边", "脚踝围左边",  "前臂围度右边",  "身体高度", "腿肚围度左边", "宽度腋窝", "最小腿围度左边")

X_LM_train <- X_train[, LM_targets]

LM_data <- cbind(y_train[,"LM"], X_LM_train)
colnames(LM_data)[1] <- "LM"

summary(LM_data$LM)

# Build regression model
LM_FSLR <- lm(LM ~ ., data = LM_data)

##################  Android  #########################

Android_targets <- c("标准站姿腰围",  "gender", "前臂围度右边", "脚踝围左边",   "臀部.大腿围度",  "臂长左",  "膝围度左边", "腰部对臀部高度左边")

X_Android_train <- X_train[, Android_targets]

Android_data <- cbind(y_train[,"Android"], X_Android_train)
colnames(Android_data)[1] <- "Android"

summary(Android_data$Android)

# Build regression model
Android_FSLR <- lm(Android ~ ., data = Android_data)

##################  Gynoid  #########################

Gynoid_targets <- c( "gender", "膝围度左边","高臀部围度", "age", "前臂围度右边", "大腿围度右边.水平线.", "腿肚围度右边", "宽度腋窝", "横过后面宽度.腋窝水平.", "胯长度.后面", "上臂长度左边")

X_Gynoid_train <- X_train[, Gynoid_targets]

Gynoid_data <- cbind(y_train[,"Gynoid"], X_Gynoid_train)
colnames(Gynoid_data)[1] <- "Gynoid"

summary(Gynoid_data$Gynoid)

# Build regression model
Gynoid_FSLR <- lm(Gynoid ~ ., data = Gynoid_data)

##################  A_G  #########################

A_G_targets <- c("臀部围度", "对高臀部的腰部后面", "大腿围度左边.水平线.", "脚踝围左边", "中间的脖子围度", "标准站姿腰围", "胸部.胸围度", "脖子到横过后面宽度.腋窝水平.", "前臂长度左边")

X_A_G_train <- X_train[, A_G_targets]

A_G_data <- cbind(y_train[,"A_G"], X_A_G_train)
colnames(A_G_data)[1] <- "A_G"

summary(A_G_data$A_G)

# Build regression model
A_G_FSLR <- lm(A_G ~ ., data = A_G_data)

##################  BFP  #########################

BFP_targets <- c("gender", "臂长左", "前臂围度右边", "膝围度左边", "脚踝围左边", "臀部.大腿围度",  "头高",  "腿肚围度左边", 
                 "最小腿围度左边", "大腿围度右边.水平线.","宽度腋窝", "脖子左边对腰部后面", "横过后面宽度.腋窝水平.", "标准站姿腰围", 
                 "颈到臀距离","胯长度.后面")

X_BFP_train <- X_train[, BFP_targets]

BFP_data <- cbind(y_train[,"BFP"], X_BFP_train)
colnames(BFP_data)[1] <- "BFP"

summary(BFP_data$BFP)

# Build regression model
BFP_FSLR <- lm(BFP ~ ., data = BFP_data)

## Residual normality test

residual_normality3 <- function(model, model_name = "Regression Model", dir = normality_dir) {
  
  # Extract model residuals
  current_data <- residuals(model)
  
  # Calculate key graphical judgment indicators
  skew_val <- skewness(current_data)
  kurt_val <- kurtosis(current_data)
  
  # Calculate Q-Q plot correlation coefficient
  qq_norm <- qqnorm(current_data, plot.it = FALSE)
  qq_correlation <- cor(qq_norm$x, qq_norm$y)
  
  # Only perform Kolmogorov-Smirnov test
  ks_test <- ks.test(current_data, "pnorm", mean = mean(current_data), sd = sd(current_data))
  ks_statistic <- ks_test$statistic
  ks_pvalue <- ks_test$p.value
  
  # Create histogram (add KS test P-value)
  p_hist <- ggplot(data.frame(x = current_data), aes(x = x)) +
    geom_histogram(aes(y = after_stat(density)), 
                   bins = min(30, length(current_data)/5), 
                   fill = "lightblue", color = "black", alpha = 0.7) +
    geom_density(color = "blue", linewidth = 0.8, alpha = 0.5) +
    stat_function(fun = dnorm, 
                  args = list(mean = mean(current_data), sd = sd(current_data)),
                  color = "red", linewidth = 1.2) +
    labs(title = paste("Histogram of Residuals -", model_name),
         subtitle = paste("Skewness:", round(skew_val, 3), 
                          "| Kurtosis:", round(kurt_val, 3)),
         x = "Residuals", y = "Density") +
    theme_minimal() +
    theme(plot.subtitle = element_text(size = 10, face = "bold"))
  
  # Create Q-Q plot
  p_qq <- ggplot(data.frame(x = current_data), aes(sample = x)) +
    stat_qq(size = 1, alpha = 0.6) +
    stat_qq_line(color = "red", linewidth = 1) +
    labs(title = paste("Q-Q Plot of Residuals -", model_name),
         subtitle = paste("Q-Q Correlation:", round(qq_correlation, 4)),
         x = "Theoretical Quantiles", y = "Sample Quantiles") +
    theme_minimal() +
    theme(plot.subtitle = element_text(size = 10, face = "bold"))
  
  # Combine plots (using patchwork package)
  if (require(patchwork, quietly = TRUE)) {
    combined_plot <- p_hist + p_qq + 
      plot_layout(ncol = 2) +
      plot_annotation(
        title = paste("Residual Normality Assessment:", model_name),
        # subtitle = paste("Sample size:", length(current_data),
        #                  "| Kolmogorov-Smirnov p-value:", round(ks_pvalue, 4)),
        theme = theme(
          plot.title = element_text(hjust = 0.5, face = "bold", size = 14),
          plot.subtitle = element_text(hjust = 0.5, size = 11, color = "darkgreen")
        )
      )
  } else {
    # If patchwork is not available, use basic grid layout
    combined_plot <- p_hist + p_qq + 
      plot_annotation(
        title = paste("Residual Normality Assessment:", model_name),
        # subtitle = paste("Sample size:", length(current_data),
        #                  "| Kolmogorov-Smirnov p-value:", round(ks_pvalue, 4)),
        theme = theme(
          plot.title = element_text(hjust = 0.5, face = "bold", size = 14),
          plot.subtitle = element_text(hjust = 0.5, size = 12, color = "darkgreen")
        )
      )
  }
  
  # Generate safe filename
  safe_model_name <- gsub("[^[:alnum:]]", "_", model_name)
  file_name <- paste0("residual_normality_", safe_model_name, ".png")
  file_path <- file.path(normality_dir, file_name)
  
  # Save image
  ggsave(filename = file_path, 
         plot = combined_plot, 
         width = 8, 
         height = 4, 
         dpi = 600,
         bg = "white")
  
  cat("Saved:", file_path, "-", "\n")
  
  # Display image in R
  print(combined_plot)
  
  # Return comprehensive judgment results
  return(tibble(
    Model = model_name,
    Sample_Size = length(current_data),
    Skewness = round(skew_val, 4),
    Kurtosis = round(kurt_val, 4),
    QQ_Correlation = round(qq_correlation, 4),
    Residual_Mean = round(mean(current_data), 4),
    Residual_SD = round(sd(current_data), 4),
    KS_Statistic = round(ks_statistic, 4),
    KS_PValue = round(ks_pvalue, 4),
    File_Path = file_path
  ))
}

normality_comparison3 <- map2(list(VAT_FSLR, FM_FSLR, FMI_FSLR, LM_FSLR, 
                                   Android_FSLR, Gynoid_FSLR, A_G_FSLR, BFP_FSLR),
                              c("VAT Model", "FM Model", "FMI Model", "LM Model", 
                                "Android Model", "Gynoid Model", "A_G Model", "BFP Model"),
                              residual_normality3)

normality_comparison_result <- bind_rows(normality_comparison3)

# Save as CSV file
write_csv(normality_comparison_result, "YourPath/Normality_Plot/residual_normality_results.csv")