df <- read.csv('C:/Users/jesse/OneDrive/Desktop/Portfolio/Portfolio/NYCitySchools.csv')

library(mltools) 
library(caret)
library(dplyr)
library(car)
library(tidyverse)
library(magrittr)
library(visdat)
library(ggplot2)
library(tidyr)
library(mice)
library(e1071)
library(DescTools)
library(randomForest)
library(themis)
library(mlbench)
library(corrplot)
library(data.table)
library(kknn)
library(xgboost)
library(Matrix)
library(class)

set.seed(1)

# House cleaning. 

# The size of the data set is such I can save the dataframe
# on a rolling basis (df1, df2, d3, etc.)

df1 <- df

df2 <- df1 %>%
  dplyr::select(
    average_grade_8_english_proficiency, 
    percent_students_with_disabilities, 
    economic_need_index, 
    percent_in_temp_housing, 
    years_of_principal_experience_at_this_school,
    teacher_attendance_rate, 
    federal, 
    program
  ) %>%
  mutate(target = case_when(
    average_grade_8_english_proficiency >= 3.0 ~ 'pass',
    average_grade_8_english_proficiency < 3.0 ~ 'fail'
  )) %>%
  dplyr::select(-average_grade_8_english_proficiency)

df2$target <- as.factor(df2$target)

# Basic Data Assessment 

summary(df2)

numeric_data <- df2[sapply(df2, is.numeric)]

means <- sapply(numeric_data, mean, na.rm = TRUE)
sds <- sapply(numeric_data, sd, na.rm = TRUE)

report <- data.frame(
  Variable = names(means),
  Mean = means,
  Standard_Deviation = sds
)

print(report)

vis_miss(df2)

df2 %>%
  pull(target) %>%
  table() %>%
  pie()

df2 %>%
  pull(program) %>%
  table() %>%
  pie()

df_numeric <- df2[, sapply(df2, is.numeric)]

df_numeric <- df_numeric %>% select(-federal)

df_long <- pivot_longer(df_numeric, cols = everything(), names_to = "key", values_to = "value")

ggplot(df_long, aes(value)) +
  geom_histogram(bins = 20, fill = 'blue', color = 'black') +
  facet_wrap(~key, scales = 'free_x') +
  theme_minimal() 

Skew(df$economic_need_index, na.rm = TRUE )
Skew(df2$percent_in_temp_housing, na.rm = TRUE)
Skew(df2$percent_students_with_disabilities, na.rm = TRUE)
Skew(df2$teacher_attendance_rate, na.rm = TRUE)
Skew(df2$years_of_principal_experience_at_this_school, na.rm = TRUE)

df25 <- df2

# Detect and Dummy Code Outliers

percent_in_temp_housing_out <- boxplot.stats(df25$percent_in_temp_housing
                                             , coef = 2.5, do.conf = TRUE, do.out = TRUE)$out # 10 outliers

percent_students_with_disabilities_out <- boxplot.stats(df25$percent_students_with_disabilities
                                                        , coef = 2.5, do.conf = TRUE, do.out = TRUE)$out # 2 outliers

teacher_attendance_rate_out <- boxplot.stats(df25$teacher_attendance_rate
                                             , coef = 2.5, do.conf = TRUE, do.out = TRUE)$out # 4 outliers

years_of_principal_experience_at_this_school <- boxplot.stats(df25$years_of_principal_experience_at_this_school
                                                              , coef = 2.5, do.conf = TRUE, do.out = TRUE)$out # 0 outliers

df25$percent_in_temp_housing_out <- ifelse(df25$percent_in_temp_housing %in% 
                                              percent_in_temp_housing_out, 1, 0)

df25$percent_students_with_disabilities_out<- ifelse(df25$percent_students_with_disabilities %in% 
                                                        percent_students_with_disabilities_out, 1, 0)

df25$teacher_attendance_rate_out <- ifelse(df25$teacher_attendance_rate %in% 
                                              teacher_attendance_rate_out, 1, 0)

# Set Dummy Codes for missing data

df25_miss <- df25[, c("years_of_principal_experience_at_this_school","teacher_attendance_rate")]

na_df <- as.data.frame(is.na(df25_miss))
na_df <- lapply(na_df, as.integer)
na_df <- as.data.frame(na_df)
col.names <- colnames(na_df)
colnames(na_df) <- paste(col.names, "na", sep = "_")

df25 <- cbind(df25, na_df)

df3 <- df25 %>% mutate_if(is.integer, as.numeric)

dummy_model <- dummyVars(~ program, data = df3)
one_hot <- predict(dummy_model, newdata = df3)
df325 <- cbind(df3, one_hot)

df325 <- df325 %>% select( -program)

# EDA

train_test_indices <- createDataPartition(df325$target, p = 0.75, list = FALSE)
train_df <- df325[train_test_indices, ]
test_df <- df325[-train_test_indices, ]

df35 <- train_df

df4 <- df35

df4$percent_in_temp_housing <- sqrt(df4$percent_in_temp_housing) ### load into CV
hist(df35$percent_in_temp_housing, breaks = 20)
hist(df4$percent_in_temp_housing, breaks = 20)

lower_bound <- quantile(df35$teacher_attendance_rate, 0.01, na.rm = TRUE)  
upper_bound <- quantile(df35$teacher_attendance_rate, 0.99, na.rm = TRUE)  

df4$teacher_attendance_rate <- pmax(pmin(df35$teacher_attendance_rate, upper_bound), lower_bound)

hist(df35$teacher_attendance_rate, breaks = 20)
hist(df4$teacher_attendance_rate, breaks = 20)

lower_bound <- quantile(test_df$teacher_attendance_rate, 0.01, na.rm = TRUE)  
upper_bound <- quantile(test_df$teacher_attendance_rate, 0.99, na.rm = TRUE)  

test_df$teacher_attendance_rate <- pmax(pmin(test_df$teacher_attendance_rate, upper_bound), lower_bound)

upper_bound <- quantile(df35$percent_students_with_disabilities, 0.99, na.rm = TRUE) 

df4$percent_students_with_disabilities <- pmin(df35$percent_students_with_disabilities, upper_bound)

hist(df35$percent_students_with_disabilities, breaks = 20)
hist(df4$percent_students_with_disabilities, breaks = 20)

df4$years_of_principal_experience_at_this_school <- sqrt(df4$years_of_principal_experience_at_this_school) ### cv
hist(df35$years_of_principal_experience_at_this_school, breaks = 20)
hist(df4$years_of_principal_experience_at_this_school, breaks = 20)

# missing value assessment

df45 <- df4

vis_miss(df45)

# years_of_principal_experience_at_this_school

df_principal <- select(df45, -target, -years_of_principal_experience_at_this_school)

df_principal$teacher_attendance_rate <- ifelse(is.na(df_principal$teacher_attendance_rate),
                                               runif(sum(is.na(df_principal$teacher_attendance_rate)), min = 0, max = 1),
                                               df_principal$teacher_attendance_rate)

df_principal$years_of_principal_experience_at_this_school_na <- factor(ifelse(df_principal$years_of_principal_experience_at_this_school_na == 1
                                                                      , "missing", "not_missing"), levels = c("missing", "not_missing"))

### Cross Validation

train_index <- createDataPartition(df_principal$years_of_principal_experience_at_this_school_na, p = 0.8, list = FALSE)
training <- df_principal[train_index, ]
testing <- df_principal[-train_index, ]

cv_control <- trainControl(
  method = "cv",          
  number = 5,
  sampling = "smote",
  classProbs = TRUE,
  summaryFunction = twoClassSummary
)

rf_model <- train(
  years_of_principal_experience_at_this_school_na ~ .,               
  data = df_principal,                
  method = "rf",            
  trControl = cv_control,   
  metric = "ROC",           
  tuneGrid = expand.grid(mtry = c(1, 2, 3)),
  weights = ifelse(df_principal$years_of_principal_experience_at_this_school_na == "minority_class", 10, 1)
)

pred.rf <- predict(rf_model, newdata = testing)

pred.rf <- predict(rf_model, testing)
confusionMatrix(table(testing$years_of_principal_experience_at_this_school_na, pred.rf))

# teacher

df_teacher <- select(df45, -target, -teacher_attendance_rate)

df_teacher$years_of_principal_experience_at_this_school <- ifelse(is.na(df_teacher$years_of_principal_experience_at_this_school),
                                               runif(sum(is.na(df_teacher$years_of_principal_experience_at_this_school)), min = 0, max = 1),
                                               df_teacher$years_of_principal_experience_at_this_school)

df_teacher$teacher_attendance_rate_na <- factor(ifelse(df_principal$teacher_attendance_rate_na == 1
                                                       , "missing", "not_missing"), levels = c("missing", "not_missing"))

# Cross Validation

train_index <- createDataPartition(df_teacher$teacher_attendance_rate_na, p = 0.8, list = FALSE)
training <- df_teacher[train_index, ]
testing <- df_teacher[-train_index, ]

cv_control <- trainControl(
  method = "cv",          
  number = 5,
  sampling = "smote",
  classProbs = TRUE,
  summaryFunction = twoClassSummary
)

rf_model <- train(
  teacher_attendance_rate_na ~ .,               
  data = df_teacher,                
  method = "rf",            
  trControl = cv_control,   
  metric = "ROC",           
  tuneGrid = expand.grid(mtry = c(1, 2, 3)),
  weights = ifelse(df_teacher$teacher_attendance_rate_na == "minority_class", 10, 1)
)

pred.rf <- predict(rf_model, newdata = testing)

pred.rf <- predict(rf_model, testing)
confusionMatrix(table(testing$teacher_attendance_rate_na, pred.rf))

# Missing value imputation -- Visual Inspection and multi-variate modeling confirm 
# the missing values are MNAR

df5 <- df45

mice_imputed <- mice(df5, m = 5, method = "pmm", maxit = 200, seed = 123)

df6 <- complete(mice_imputed, 1)

vis_miss(df6) # no missing values

plot(mice_imputed)

# EDA -- Normally I would cross validate most of these measures to avoid a data leak ; however it's a 
# bit resource in R and some don't recommend it.

plot_density(df6)

df6 %>%
  pull(target) %>%
  table() %>%
  pie()

binary_df <- df6 %>% select( -percent_students_with_disabilities, -economic_need_index, -percent_in_temp_housing
                    , -years_of_principal_experience_at_this_school, -teacher_attendance_rate )

for (i in 1:(ncol(binary_df) - 1)) {  
  for (j in (i + 1):ncol(binary_df)) {
    var1 <- colnames(binary_df)[i]  
    var2 <- colnames(binary_df)[j]  
    
    tbl <- table(binary_df[[var1]], binary_df[[var2]])  
 
    mosaicplot(tbl, main = paste(var1, "vs", var2), xlab = var1, ylab = var2)
  }
} # Program type is irrelevant

# Watchlist variables are basically useless
# teacher attendance rate na and years principal na share an odd association (weak)
# year principal na and teacher attendance rate out have an association (weak)
# percent student disability and program have an association (weak)

# categorical correlation table

# Initialize the matrix
var_names <- colnames(binary_df)
n <- length(var_names)
cramers_v_matrix <- matrix(NA, nrow = n, ncol = n, dimnames = list(var_names, var_names))

# Function to compute Cramér's V
compute_cramers_v <- function(var1, var2) {
  tbl <- table(var1, var2)  
  chi2 <- chisq.test(tbl, correct = FALSE)$statistic  
  n <- sum(tbl)  
  k <- min(nrow(tbl), ncol(tbl)) 
  sqrt(chi2 / (n * (k - 1)))  
}

# Compute Cramér's V for all variable pairs
for (i in 1:n) {
  for (j in i:n) {
    if (i == j) {
      cramers_v_matrix[i, j] <- 1  
    } else {
      cramers_v_matrix[i, j] <- compute_cramers_v(binary_df[[i]], binary_df[[j]])
      cramers_v_matrix[j, i] <- cramers_v_matrix[i, j]  # Fill the lower triangle
    }
  }
}

corrplot(cramers_v_matrix, method = "color", type = "upper",
         tl.col = "black", tl.srt = 45,
         col = colorRampPalette(c("blue", "white", "red"))(200),
         title = "Cramér's V Heatmap",
         tl.cex = 0.6)

# Likely predictors: federal, Principal Experience na, teach attendance na 
# Correlation matrix numerics

df65 <- df6

df_numeric <- df65

numeric_vars <- df65 %>% select(-target, -federal, -years_of_principal_experience_at_this_school_na, -percent_in_temp_housing_out, 
                                        -percent_students_with_disabilities_out, -teacher_attendance_rate_out, -teacher_attendance_rate_na,  
                                        -programa, -programb, -programc, -programd, -programe, -programf, -programg, -programh, -programi, 
                                        -programj, -programk)

spearman_cor_matrix <- cor(numeric_vars, use = "pairwise.complete.obs", method = "spearman")

corrplot(spearman_cor_matrix, method = "color", type = "upper",
         tl.col = "black", tl.srt = 45, # Customize text color and angle
         col = colorRampPalette(c("blue", "white", "red"))(200),
         title = "Spearman Correlation Heatmap",
         tl.cex = 0.6) # Adjust text size

df65$needy = rowMeans(df65[, c("economic_need_index", "percent_in_temp_housing")], na.rm = TRUE)

test_df$needy = rowMeans(test_df[, c("economic_need_index", "percent_in_temp_housing")], na.rm = TRUE) 

boxplots <- df65 %>% select(-federal, -years_of_principal_experience_at_this_school_na, -percent_in_temp_housing_out, 
                               -percent_students_with_disabilities_out, -teacher_attendance_rate_out, -teacher_attendance_rate_na,  
                               -programa, -programb, -programc, -programd, -programe, -programf, -programg, -programh, -programi, 
                               -programj, -programk)

plot_boxplot(boxplots, by = "target")

# Economic Need Index, Needy, percent in temp housing, percent students with disabilities

boxplots$target <- as.numeric(boxplots$target)

kendall_cor_matrix <- cor(boxplots, use = "pairwise.complete.obs", method = "kendall")

corrplot(kendall_cor_matrix, method = "color", type = "upper",
         tl.col = "black", tl.srt = 45, 
         col = colorRampPalette(c("blue", "white", "red"))(200),
         title = "Kendall's Tau Heatmap",
         tl.cex = 0.6) # Adjust text size

df7 <- df65 %>% select(federal, years_of_principal_experience_at_this_school_na, teacher_attendance_rate_na
                       , percent_students_with_disabilities, economic_need_index, percent_in_temp_housing, needy)

# Variable Manipulations

train_df <- train_df %>% select(target, federal, years_of_principal_experience_at_this_school_na, 
                                teacher_attendance_rate_na, percent_students_with_disabilities, economic_need_index, percent_in_temp_housing)

train_df$needy = rowMeans(train_df[, c("economic_need_index", "percent_in_temp_housing")], na.rm = TRUE)

test_df <- test_df %>% select(target, federal, years_of_principal_experience_at_this_school_na, 
                                teacher_attendance_rate_na, percent_students_with_disabilities, economic_need_index, percent_in_temp_housing)

test_df$needy = rowMeans(test_df[, c("economic_need_index", "percent_in_temp_housing")], na.rm = TRUE)

df7 <- train_df

# Cross Validation

# Define fold-wise preprocessing function
fold_preprocess <- function(train_fold, test_fold) {

  upper_bound <- quantile(train_fold$percent_students_with_disabilities, 0.99, na.rm = TRUE)
  train_fold$percent_students_with_disabilities <- pmin(train_fold$percent_students_with_disabilities, upper_bound)
  test_fold$percent_students_with_disabilities <- pmin(test_fold$percent_students_with_disabilities, upper_bound)
  

  train_fold$percent_in_temp_housing <- sqrt(train_fold$percent_in_temp_housing)
  test_fold$percent_in_temp_housing <- sqrt(test_fold$percent_in_temp_housing)
  
  return(list(train = train_fold, test = test_fold))
}

# Custom trainControl with stratified sampling
ctrl <- trainControl(
  method = "repeatedcv",
  number = 5,
  repeats = 3,
  classProbs = TRUE,
  summaryFunction = twoClassSummary,
  savePredictions = "final",
  sampling = "smote"  # Handle imbalance with SMOTE
)

# Preprocess data dynamically for cross-validation
dynamic_preprocessing <- function(data, index) {
  processed_folds <- lapply(index, function(train_indices) {
    train_fold <- data[train_indices, ]
    test_fold <- data[-train_indices, ]
    fold_preprocess(train_fold, test_fold)
  })
  return(processed_folds)
}

train_indices <- createFolds(train_df$target, k = 5, returnTrain = TRUE)

processed_folds <- dynamic_preprocessing(train_df, train_indices)

# Diagnostic check for missing values after preprocessing
lapply(processed_folds, function(fold) {
  cat("Missing values in training fold:\n")
  print(colSums(is.na(fold$train)))
  
  cat("Missing values in test fold:\n")
  print(colSums(is.na(fold$test)))
})


# KNN Model
classify.knn <- train(
  target ~ ., 
  data = train_df,  
  method = "kknn",
  metric = "ROC",
  trControl = ctrl,
  preProcess = c("center", "scale"),  
  tuneGrid = expand.grid(
    kmax = seq(5, 10, by = 1),          
    distance = c(1, 2),                 
    kernel = c("rectangular", "triangular", "epanechnikov")
  )
)

plot(classify.knn)

print(classify.knn)

pred.knn <- predict(classify.knn, test_df)
confusionMatrix(table(test_df[,"target"], pred.knn)) 

# 
#      fail pass
# fail  182    6
# pass    0   54

train_predictions <- predict(classify.knn, train_df, type = "prob")  
train_classifications <- predict(classify.knn, train_df)            

train_results_knn <- data.frame(
  actual = train_df$target,               
  knn_predicted_class = train_classifications,  
  knn_predicted_prob = train_predictions[, 2]   
)

test_predictions <- predict(classify.knn, test_df, type = "prob")   
test_classifications <- predict(classify.knn, test_df)             

test_results_knn <- data.frame(
  actual = test_df$target,                
  knn_predicted_class = test_classifications,  
  knn_predicted_prob = test_predictions[, 2]   
)

# View the outputs
head(train_results_knn)
head(test_results_knn)

# SVM Model
svm_grid <- expand.grid(C = seq(0.001, 0.010, length = 30))

classify.svm <- train(
  target ~ ., 
  data = train_df, 
  method = "svmLinear", 
  metric = "ROC", 
  trControl = ctrl,  
  preProcess = c("center", "scale"), 
  tuneGrid = expand.grid(C = seq(0.001, 0.010, length = 30))
)

plot(classify.svm)

print(classify.svm)

pred.svm <- predict(classify.svm, test_df)
confusionMatrix(table(test_df[,"target"], pred.svm)) 

# 
#     fail pass
#fail  149   39
#pass    9   45     

train_predictions <- predict(classify.svm, train_df, type = "prob")  
train_classifications <- predict(classify.svm, train_df)            

# Combine results into a dataframe for train_df
train_results_svm <- data.frame(
  actual = train_df$target,               
  svm_predicted_class = train_classifications,  
  svm_predicted_prob = train_predictions[, 2]  
)

test_predictions <- predict(classify.svm, test_df, type = "prob")   
test_classifications <- predict(classify.svm, test_df)             

test_results_svm <- data.frame(
  actual = test_df$target,               
  svm_predicted_class = test_classifications,  
  svm_predicted_prob = test_predictions[, 2]   
)

# View the outputs
head(train_results_svm)
head(test_results_svm)

# random forest

# Note ROC is used to generate as Kappa cannot be used.

mtry <- sqrt(ncol(train_df))

tunegrid <- expand.grid(.mtry=mtry)

classify.rf <- train(target ~ .,
                     data = train_df,
                     method = 'rf',
                     metric = 'ROC', # Random forest does not have a Kappa option
                     tuneGrid = tunegrid,
                     trControl = ctrl) 

pred.rf <- predict(classify.rf, test_df)
confusionMatrix(table(test_df[,"target"], pred.rf)) 

#      fail pass
# fail  223    2
# pass    9   56
#
# Kappa : 0.9294  

# Predictions and classifications for train_df
train_predictions <- predict(classify.rf, train_df, type = "prob")  
train_classifications <- predict(classify.rf, train_df)            

# Combine results into a dataframe for train_df
train_results_rf <- data.frame(
  actual = train_df$target,              
  rf_predicted_class = train_classifications,  
  rf_predicted_prob = train_predictions[, 2]   
)

# Predictions and classifications for test_df
test_predictions <- predict(classify.rf, test_df, type = "prob")   
test_classifications <- predict(classify.rf, test_df)             

# Combine results into a dataframe for test_df
test_results_rf <- data.frame(
  actual = test_df$target,                
  rf_predicted_class = test_classifications,  
  rf_predicted_prob = test_predictions[, 2]   
)

# CHeck outputs
head(train_results_rf)
head(test_results_rf)

test_cb <- cbind(test_results_rf, test_results_svm, test_results_knn, test_df)

test_cb <- test_cb %>% select(-actual)

test_cb$rf_predicted_class <- as.numeric(test_cb$rf_predicted_class) - 1
test_cb$svm_predicted_class <- as.numeric(test_cb$svm_predicted_class) - 1
test_cb$knn_predicted_class <- as.numeric(test_cb$knn_predicted_class) - 1

train_cb <- cbind(train_results_rf, train_results_svm, train_results_knn, train_df)

train_cb <- train_cb %>% select(-actual)

train_cb$rf_predicted_class <- as.numeric(train_cb$rf_predicted_class) - 1
train_cb$svm_predicted_class <- as.numeric(train_cb$svm_predicted_class) - 1
train_cb$knn_predicted_class <- as.numeric(train_cb$knn_predicted_class) - 1

# The accuracies on the existing models is very good.  But the purpose of this 
# build was to use an ensemble model. Let's see if we can boost the power of the
# KNN, SVM, and rf models using XGBoost and combine the with the original data

tune_grid <- expand.grid(
  nrounds = c(100, 200, 300),       
  max_depth = c(1, 3, 6),           
  eta = c(0.01, 0.1),          
  gamma = c(1),               
  colsample_bytree = c(3, 6), 
  min_child_weight = c(3, 6),    
  subsample = c(0.7, 0.9, 1.1)           
)

xgb_model <- train(
  target ~ ., 
  data = train_cb,                  
  method = "xgbTree",           
  metric = "ROC",                   
  trControl = ctrl,                 
  tuneGrid = tune_grid,             
  preProcess = c("center", "scale") 
)

print(xgb_model)

# Ensure the target column in test_cb is a factor with appropriate levels
test_cb$target <- factor(test_cb$target, levels = c("fail", "pass"))

# Make predictions using the trained XGBoost model
xgb_predictions_prob <- predict(xgb_model, newdata = test_cb, type = "prob")

# Convert probabilities to predicted classes (threshold = 0.5)
xgb_predictions_class <- ifelse(xgb_predictions_prob[, "pass"] > 0.5, "pass", "fail")
xgb_predictions_class <- factor(xgb_predictions_class, levels = c("fail", "pass"))

# Generate the confusion matrix
xgb_confusion <- confusionMatrix(xgb_predictions_class, test_cb$target)

# Print the confusion matrix
print(xgb_confusion)

# Prediction fail pass
# fail  182    2
# pass    6   52
#
# Kappa : 0.9071