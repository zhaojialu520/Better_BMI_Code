# Load necessary packages
library(lmtest)
library(car)
library(caret)

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

# Read training set labels
y_train <- read.csv("YourPath/y_train_All3D_NoScaled.csv", 
                    row.names = 1)  ## Use first column as row names

# Read test set labels
y_test <- read.csv("YourPath/y_test_All3D_NoScaled.csv", 
                   row.names = 1)  ## Use first column as row names

##################  VAT  #########################

VAT_targets <- c("标准站姿腰围", "大腿围度左边.水平线.", "对高臀部的腰部后面",   "臂长右", "age", "gender")

X_VAT_train <- X_train[, VAT_targets]

VAT_data <- cbind(y_train[,"VATmass"], X_VAT_train)
colnames(VAT_data)[1] <- "VAT"

# Build regression model
VAT_FSLR <- lm(VAT ~ .+ age : 标准站姿腰围 + gender : 标准站姿腰围, data = VAT_data)

# Perform DW test
dw_VAT_result <- dwtest(VAT_FSLR)
print(dw_VAT_result) 

##################  FM  #########################

FM_targets <- c("膝围度左边", "gender", "脚踝围左边", "前臂围度右边", "标准站姿腰围", "age", "胯部长度", "头高",'腿肚围度左边', "最小腿围度左边")

X_FM_train <- X_train[, FM_targets]

FM_data <- cbind(y_train[,"FM"], X_FM_train)
colnames(FM_data)[1] <- "FM"

# Build regression model
FM_FSLR <- lm(FM ~ ., data = FM_data)

# Perform DW test
dw_FM_result <- dwtest(FM_FSLR)
print(dw_FM_result) 

##################  FMI  #########################

FMI_targets <- c("中臀围", "身体高度", "膝围度左边", "gender",  "age",  "脚踝围右边", "臂长左", "上臂围度右边")

X_FMI_train <- X_train[, FMI_targets]

FMI_data <- cbind(y_train[,"FMI"], X_FMI_train)
colnames(FMI_data)[1] <- "FMI"

# Build regression model
FMI_FSLR <- lm(FMI ~ ., data = FMI_data)

# Perform DW test
dw_FMI_result <- dwtest(FMI_FSLR)
print(dw_FMI_result) 

##################  LM  #########################

LM_targets <- c("gender", "高臀部围度", "膝围度左边", "脚踝围左边",  "前臂围度右边",  "身体高度", "腿肚围度左边", "宽度腋窝", "最小腿围度左边")

X_LM_train <- X_train[, LM_targets]

LM_data <- cbind(y_train[,"LM"], X_LM_train)
colnames(LM_data)[1] <- "LM"

# Build regression model
LM_FSLR <- lm(LM ~ ., data = LM_data)

# Perform DW test
dw_LM_result <- dwtest(LM_FSLR)
print(dw_LM_result) 

##################  Android  #########################

Android_targets <- c("标准站姿腰围",  "gender", "前臂围度右边", "脚踝围左边",   "臀部.大腿围度",  "臂长左",  "膝围度左边", "腰部对臀部高度左边")

X_Android_train <- X_train[, Android_targets]

Android_data <- cbind(y_train[,"Android"], X_Android_train)
colnames(Android_data)[1] <- "Android"

# Build regression model
Android_FSLR <- lm(Android ~ ., data = Android_data)

# Perform DW test
dw_Android_result <- dwtest(Android_FSLR)
print(dw_Android_result) 

##################  Gynoid  #########################

Gynoid_targets <- c( "gender", "膝围度左边","高臀部围度", "age", "前臂围度右边", "大腿围度右边.水平线.", "腿肚围度右边", "宽度腋窝", "横过后面宽度.腋窝水平.", "胯长度.后面", "上臂长度左边")

X_Gynoid_train <- X_train[, Gynoid_targets]

Gynoid_data <- cbind(y_train[,"Gynoid"], X_Gynoid_train)
colnames(Gynoid_data)[1] <- "Gynoid"

# Build regression model
Gynoid_FSLR <- lm(Gynoid ~ ., data = Gynoid_data)

# Perform DW test
dw_Gynoid_result <- dwtest(Gynoid_FSLR)
print(dw_Gynoid_result) 

##################  A_G  #########################

A_G_targets <- c("臀部围度", "对高臀部的腰部后面", "大腿围度左边.水平线.", "脚踝围左边", "中间的脖子围度", "标准站姿腰围", "胸部.胸围度", "脖子到横过后面宽度.腋窝水平.", "前臂长度左边")

X_A_G_train <- X_train[, A_G_targets]

A_G_data <- cbind(y_train[,"A_G"], X_A_G_train)
colnames(A_G_data)[1] <- "A_G"

# Build regression model
A_G_FSLR <- lm(A_G ~ ., data = A_G_data)

# Perform DW test
dw_A_G_result <- dwtest(A_G_FSLR)
print(dw_A_G_result) 

##################  BFP  #########################

BFP_targets <- c("gender", "臂长左", "前臂围度右边", "膝围度左边", "脚踝围左边", "臀部.大腿围度",  "头高",  "腿肚围度左边", 
                 "最小腿围度左边", "大腿围度右边.水平线.","宽度腋窝", "脖子左边对腰部后面", "横过后面宽度.腋窝水平.", "标准站姿腰围", 
                 "颈到臀距离","胯长度.后面")

X_BFP_train <- X_train[, BFP_targets]

BFP_data <- cbind(y_train[,"BFP"], X_BFP_train)
colnames(BFP_data)[1] <- "BFP"

# Build regression model
BFP_FSLR <- lm(BFP ~ ., data = BFP_data)

# Perform DW test
dw_BFP_result <- dwtest(BFP_FSLR)
print(dw_BFP_result) 