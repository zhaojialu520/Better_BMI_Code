# 加载必要的包
library(lmtest)
library(car)
library(caret)



## 导入数据
# 读取训练集特征
X_train <- read.csv("C:/Users/zjl__/Desktop/output_zhaojialu/output/SummaryStatistic/X_train_All3D_NoScaled.csv", 
                    row.names = 1)  ## 将数据框的第一列作为数据框的行名

## 处理列名
name <- colnames(X_train)    ## 提取数据框的列名
name <- gsub('人体外观测量.三维人体扫描分析系统.', '', name)    ## 删除列名中"人体外观测量-三维人体扫描分析系统:"的部分
name <- gsub('.cm.', '', name)    ## 删除列名中".cm."的部分
colnames(X_train) <- name   ## 替换数据框的列名为最简单名字

# print(X_train)
X_train <- X_train[, !colnames(X_train) %in% c('height', 'Waist_To_Hip_Ratio.x.100.')]
print(X_train)

# 读取测试集特征
X_test <- read.csv("C:/Users/zjl__/Desktop/output_zhaojialu/output/SummaryStatistic/X_test_All3D_NoScaled.csv", 
                   row.names = 1)  ## 将数据框的第一列作为数据框的行名
colnames(X_test) <- name   ## 替换数据框的列名为最简单名字
X_test <- X_test[, !colnames(X_test) %in% c('height', 'Waist_To_Hip_Ratio.x.100.')]
# print(X_test)

# 读取训练集标签
y_train <- read.csv("C:/Users/zjl__/Desktop/output_zhaojialu/output/SummaryStatistic/y_train_All3D_NoScaled.csv", 
                    row.names = 1)  ## 将数据框的第一列作为数据框的行名

# 读取测试集标签
y_test <- read.csv("C:/Users/zjl__/Desktop/output_zhaojialu/output/SummaryStatistic/y_test_All3D_NoScaled.csv", 
                   row.names = 1)  ## 将数据框的第一列作为数据框的行名


##################  VAT  #########################

VAT_targets <- c("标准站姿腰围", "大腿围度左边.水平线.", "对高臀部的腰部后面",   "臂长右", "上臂围度左边", "age")

X_VAT_train <- X_train[, VAT_targets]

VAT_data <- cbind(y_train[,"VATmass"], X_VAT_train)
colnames(VAT_data)[1] <- "VAT"

# 建立回归模型
VAT_FSLR <- lm(VAT ~ ., data = VAT_data)

# 进行DW检验
dw_VAT_result <- dwtest(VAT_FSLR)
print(dw_VAT_result) ## DW = 1.8993, p-value = 0.05131


##################  FM  #########################

FM_targets <- c("膝围度左边", "gender", "脚踝围左边", "前臂围度右边", "标准站姿腰围", "age", "胯部长度", "头高",'腿肚围度左边', "最小腿围度左边")

X_FM_train <- X_train[, FM_targets]

FM_data <- cbind(y_train[,"FM"], X_FM_train)
colnames(FM_data)[1] <- "FM"

# 建立回归模型
FM_FSLR <- lm(FM ~ ., data = FM_data)

# 进行DW检验
dw_FM_result <- dwtest(FM_FSLR)
print(dw_FM_result) ## DW = 1.921, p-value = 0.09921

##################  FMI  #########################

FMI_targets <- c("中臀围", "身体高度", "膝围度左边", "gender",  "age",  "脚踝围右边", "臂长左", "上臂围度右边")

X_FMI_train <- X_train[, FMI_targets]

FMI_data <- cbind(y_train[,"FMI"], X_FMI_train)
colnames(FMI_data)[1] <- "FMI"

# 建立回归模型
FMI_FSLR <- lm(FMI ~ ., data = FMI_data)

# 进行DW检验
dw_FMI_result <- dwtest(FMI_FSLR)
print(dw_FMI_result) ## DW = 1.992, p-value = 0.4438


##################  LM  #########################

LM_targets <- c("gender", "高臀部围度", "膝围度左边", "脚踝围左边",  "前臂围度右边",  "身体高度", "腿肚围度左边", "宽度腋窝", "最小腿围度左边")

X_LM_train <- X_train[, LM_targets]

LM_data <- cbind(y_train[,"LM"], X_LM_train)
colnames(LM_data)[1] <- "LM"

# 建立回归模型
LM_FSLR <- lm(LM ~ ., data = LM_data)

# 进行DW检验
dw_LM_result <- dwtest(LM_FSLR)
print(dw_LM_result) ## DW = 2.0129, p-value = 0.5777

##################  Android  #########################

Android_targets <- c("标准站姿腰围",  "gender", "前臂围度右边", "脚踝围左边",   "臀部.大腿围度",  "臂长左",  "膝围度左边", "腰部对臀部高度左边")

X_Android_train <- X_train[, Android_targets]

Android_data <- cbind(y_train[,"Android"], X_Android_train)
colnames(Android_data)[1] <- "Android"

# 建立回归模型
Android_FSLR <- lm(Android ~ ., data = Android_data)

# 进行DW检验
dw_Android_result <- dwtest(Android_FSLR)
print(dw_Android_result) ## DW = 1.9673, p-value = 0.2959


##################  Gynoid  #########################

Gynoid_targets <- c( "gender", "膝围度左边","高臀部围度", "age", "前臂围度右边", "大腿围度右边.水平线.", "腿肚围度右边", "宽度腋窝", "横过后面宽度.腋窝水平.", "胯长度.后面", "上臂长度左边")

X_Gynoid_train <- X_train[, Gynoid_targets]

Gynoid_data <- cbind(y_train[,"Gynoid"], X_Gynoid_train)
colnames(Gynoid_data)[1] <- "Gynoid"

# 建立回归模型
Gynoid_FSLR <- lm(Gynoid ~ ., data = Gynoid_data)

# 进行DW检验
dw_Gynoid_result <- dwtest(Gynoid_FSLR)
print(dw_Gynoid_result) ## data:  Gynoid_FSLR DW = 2.0369, p-value = 0.7186

##################  A_G  #########################

A_G_targets <- c("臀部围度", "对高臀部的腰部后面", "大腿围度左边.水平线.", "脚踝围左边", "中间的脖子围度", "标准站姿腰围", "胸部.胸围度", "脖子到横过后面宽度.腋窝水平.", "前臂长度左边")

X_A_G_train <- X_train[, A_G_targets]

A_G_data <- cbind(y_train[,"A_G"], X_A_G_train)
colnames(A_G_data)[1] <- "A_G"

# 建立回归模型
A_G_FSLR <- lm(A_G ~ ., data = A_G_data)

# 进行DW检验
dw_A_G_result <- dwtest(A_G_FSLR)
print(dw_A_G_result) ## DW = 1.8485, p-value = 0.007192


##################  BFP  #########################

BFP_targets <- c("gender", "臂长左", "前臂围度右边", "膝围度左边", "脚踝围左边", "臀部.大腿围度",  "头高",  "腿肚围度左边", 
                 "最小腿围度左边", "大腿围度右边.水平线.","宽度腋窝", "脖子左边对腰部后面", "横过后面宽度.腋窝水平.", "标准站姿腰围", 
                 "颈到臀距离","胯长度.后面")

X_BFP_train <- X_train[, BFP_targets]

BFP_data <- cbind(y_train[,"BFP"], X_BFP_train)
colnames(BFP_data)[1] <- "BFP"

# 建立回归模型
BFP_FSLR <- lm(BFP ~ ., data = BFP_data)

# 进行DW检验
dw_BFP_result <- dwtest(BFP_FSLR)
print(dw_BFP_result) ## data:  BFP_FSLR DW = 1.9218, p-value = 0.1013



