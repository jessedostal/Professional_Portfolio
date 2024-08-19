
df <- read.csv('C:/Users/jesse/OneDrive/Desktop/Portfolio/Portfolio/NYCitySchools.csv')

library(mltools) 
library(caret)
library(dplyr)
library(car)

set.seed(1)

### House cleaning. 

### The size of the data set is such I can save the dataframe
### on a rolling basis (df1, df2, d3, etc.)

df1 <- df

df2 <- df1 %>%
  
  select(average_grade_8_english_proficiency, percent_students_with_disabilities, 
             economic_need_index, percent_in_temp_housing, years_of_principal_experience_at_this_school
             , teacher_attendance_rate, federal, watchlist) %>%

  mutate(target = case_when(average_grade_8_english_proficiency >= 3.0 ~ 'pass',
                             average_grade_8_english_proficiency < 3.0 ~ 'fail'))  %>%
          
  select(-average_grade_8_english_proficiency)

df2$watchlist <- as.factor(df2$watchlist)
df2$target <- as.factor(df2$target)

### Basic Dataframe information

str(df2)

### Assess for missing data

library(naniar)

vis_miss(df2)

gg_miss_upset(df2)

na_df <- as.data.frame(is.na(df2))
na_df <- lapply(na_df, as.integer)
na_df <- as.data.frame(na_df)
col.names <- colnames(na_df)
colnames(na_df) <- paste(col.names, "na", sep = "_")

df2 <- cbind(df2, na_df)

df3 <- df2[, !sapply(df2, function(col) all(col == 0))]

### Data is obviouslyat not missing at random for years_of_principal_at_this_school
### teacher_attendance_rate_NA

install.packages("DataExplorer")

library(DataExplorer)

df35 <- df3 %>% mutate_if(is.integer, as.numeric)

summary(df35)

plot_histogram(df35)

## Outliers
##
## percent_in_temp_housing
## percent_students_with_disabilities
## teacher_attendance_rate
## years_of_principal_experience_at_this_school?

plot_density(df35)

df35 %>%
  pull(target) %>%
  table() %>%
  pie()

df35 %>%
  pull(watchlist) %>%
  table() %>%
  pie()

plot_boxplot(df35, by = "target")

plot_boxplot(df35, by = "watchlist")

ggplot(df35) +
  geom_bar(aes(x = watchlist, fill = target), position = "dodge") +
  xlab("watchlist") + ylab("Count")

## outlier analysis
##
## outliers have minimal effect on target
## number or outliers is minimal we'll skip using cross validation.

library(arsenal)

tableby(target ~ .,
        data = df35) %>%
  summary(pfootnote=TRUE)

install.packages('grDevices')

library(grDevices)
 
df357 <- df35

## Mark the outliers in new variable
## remove values from outliers
## missing values will be imputed later

percent_in_temp_housing_out <- boxplot.stats(df357$percent_in_temp_housing, 
                                                  coef = 2.5, do.conf = TRUE, do.out = TRUE)$out ## 10 outliers
percent_students_with_disabilities_out <- boxplot.stats(df357$percent_students_with_disabilities
                                                             , coef = 2.5, do.conf = TRUE, do.out = TRUE)$out ## 2 outliers
teacher_attendance_rate_out <- boxplot.stats(df357$teacher_attendance_rate, 
                                                  coef = 2.5, do.conf = TRUE, do.out = TRUE)$out ## 4 outliers
boxplot.stats(df357$years_of_principal_experience_at_this_school, coef = 2.5, do.conf = TRUE, do.out = TRUE) ## 0 outliers

df357$percent_in_temp_housing_out = 0
df357$percent_students_with_disabilities_out = 0
df357$teacher_attendance_rate_out = 0

df357$percent_in_temp_housing_out <- ifelse(df357$percent_in_temp_housing %in%
                                           percent_in_temp_housing_out, 1, 0)
df357$percent_students_with_disabilities_out<- ifelse(df357$percent_students_with_disabilities %in% 
                                                     percent_students_with_disabilities_out, 1, 0)
df357$teacher_attendance_rate_out <- ifelse(df357$teacher_attendance_rate %in% 
                                           teacher_attendance_rate_out, 1, 0)

df357$percent_in_temp_housing[df357$percent_in_temp_housing_out == 1] <- NA
df357$percent_students_with_disabilities[df357$percent_students_with_disabilities_out == 1] <- NA
df357$teacher_attendance_rate[df357$teacher_attendance_rate_out == 1] <- NA

sum(df357$percent_in_temp_housing_out)
sum(df357$percent_students_with_disabilities_out)
sum(df357$teacher_attendance_rate_out)

vis_miss(df357)

df39 <- select(df357, -percent_in_temp_housing_out
               , -percent_students_with_disabilities_out
               , -teacher_attendance_rate_out)

# Imputation of missing variables

target <- df39$target

df4 <- select(df39, -target)

library(mice)

df.impute <- mice(df4, defaultMthod= c("rf","rf","rf","rf"))

df5 <- complete(df.impute,1)

vis_miss(df5) ## no missing values

df6 <- cbind(target, df5)

### EDA

plot_histogram(df6)

plot_density(df6)

df6 %>%
  pull(target) %>%
  table() %>%
  pie()

df6 %>%
  pull(watchlist) %>%
  table() %>%
  pie()

ggplot(df6) +
  geom_bar(aes(x = watchlist, fill = target), position = "dodge") +
  xlab("watchlist") + ylab("Count")

plot_boxplot(df6, by = "watchlist")
plot_boxplot(df6, by = "target")

## teacher_attendance_rate 
## principal_experience_at_the_school 
## watchlist
## appear to be poor predictors 

## Split train test

smp_size <- floor(0.50 * nrow(df6))

train_ind <- sample(seq_len(nrow(df6)), size = smp_size)

df6train <- df6[train_ind, ]
df6test <- df6[-train_ind, ]

tableby(target ~ .,
        data = df6train) %>%
  summary(pfootnote=TRUE)

## Poor Predictors:
## watchlist
## teacher_attendance_rate

tableby(target ~ .,
        data = df6test) %>%
  summary(pfootnote=TRUE)

## Poor Predictors:
## watchlist
## teacher_attendance_rate
## years_of_principal_experience_at_this_school

## Drop:
## teacher_attendance_rate
## watchlist

## Retain inconclusive: years_of_principal_experience_at_this_school


df65 <- select(df6, -watchlist, -teacher_attendance_rate)

df7 <- mutate(df65, antifederal = case_when (federal == 1 ~ 0,
                                            federal == 0 ~ 1))
### KNN

library(caret)
library(embed)

knn.index <- createDataPartition(df7$target, p = 0.7, list = FALSE)

training <- df7[knn.index, ]  
testing <- df7[-knn.index, ]  

####

ctrl <- trainControl(method = "repeatedcv",  
                     classProbs = TRUE,
                     number = 3,
                     repeats = 7)     

### Initial Runs with kappa running 1:20 (incremented at 1) result in 1 
### as the best fit. Rerun kappa at 1:5 (incremented at 0.25)

classify.knn <- train(target ~ ., 
                  data = training, 
                  method = "knn", 
                  metric = "Kappa",
                  trControl = ctrl,  
                  preProcess = c("center","scale"), 
                  tuneGrid =data.frame(k=seq(1,5,by=0.25)))

pred.knn <- predict(classify.knn, testing) 
confusionMatrix(table(testing[,"target"], pred.knn)) 

print(classify.knn)

plot(classify.knn)

# Run 1
#
#      fail pass
# fail  218    7
# pass   16   49
#
# Kappa : 0.7602

# Generate predicted classes and probabilities for test and train
pred.train.prob <- predict(classify.knn, training, type = "prob")[, 2]  

pred.test.prob <- predict(classify.knn, testing, type = "prob")[, 2]  

# Create an empty vector to hold the combined probabilities
predicted_prob <- numeric(nrow(df7))
predicted_class <- factor(levels = levels(df7$target))

# Fill in the probabilities and classes for test and train
predicted_prob[knn.index] <- pred.train.prob
predicted_prob[-knn.index] <- pred.test.prob

# Create a new data frame with the original index
knn_predictions <- data.frame(
  knn_prob = predicted_prob
)

# View the combined data
head(knn_predictions)

### svm

svm.index <- createDataPartition(df7$target, p = 0.7, list = FALSE)

training <- df7[svm.index, ]  
testing <- df7[-svm.index, ]  

ctrl <- trainControl(method = "repeatedcv",  
                     classProbs = TRUE,
                     number = 5,
                     repeats = 5)    

classify.svm <- train(target ~., 
                  data = training, 
                  method = "svmLinear", 
                  metric = "Kappa",
                  trControl = ctrl,  
                  preProcess = c("center","scale"), 
                  tuneGrid = expand.grid(C = seq(0, 2, length = 30))) 

plot(classify.svm)

print(classify.svm)

pred.svm <- predict(classify.svm, testing)
confusionMatrix(table(testing[,"target"], pred.svm)) 

#       fail pass
#  fail  215   10
#  pass   17   48
# 
# Kappa : 0.7216  

# Generate predicted classes and probabilities for the test and train 
pred.train <- predict(classify.svm, training)
pred.train.prob <- predict(classify.svm, training, type = "prob")[, 2]  

pred.test <- predict(classify.svm, testing)
pred.test.prob <- predict(classify.svm, testing, type = "prob")[, 2]  

# Create an empty vector to hold the combined probabilities and predicted classes
predicted_prob <- numeric(nrow(df7))
predicted_class <- factor(rep(NA, nrow(df7)), levels = levels(df7$target))

# Assign predicted probabilities and classes based on the original indices
predicted_prob[svm.index] <- pred.train.prob
predicted_prob[-svm.index] <- pred.test.prob

predicted_class[svm.index] <- pred.train
predicted_class[-svm.index] <- pred.test

# Combine into a data frame using the original row names from df7
svm_predictions <- data.frame(
  svm_pred = predicted_class,
  svm_prob = predicted_prob,
  row.names = rownames(df7)  # Use the original row names from df7
)

# View the combined data
head(svm_predictions)

### random forest

library(randomForest)
library(mlbench)
library(e1071)

### Note ROC are used to generate the fores as Kappa cannot be used.

rf.index <- createDataPartition(df7$target, p = 0.7, list = FALSE)

training <- df7[rf.index, ]  
testing <- df7[-rf.index, ]  

ctrl <- trainControl(method = "repeatedcv",  
                     number = 5,
                     repeats = 5, 
                     classProbs = TRUE,
                     summaryFunction = twoClassSummary)     

mtry <- sqrt(ncol(training))

tunegrid <- expand.grid(.mtry=mtry)

classify.rf <- train(target ~ .,
                     data = training,
                     method = 'rf',
                     metric = 'ROC', ## Random forest does not have a Kappa option
                     tuneGrid = tunegrid,
                     trControl = ctrl) 

pred.rf <- predict(classify.rf, testing)
confusionMatrix(table(testing[,"target"], pred.rf)) 

#      fail pass
# fail  223    2
# pass    9   56
#
# Kappa : 0.8866  

####

# Generate predicted classes and probabilities for the test and train 
pred.train <- predict(classify.rf, training)
pred.train.prob <- predict(classify.rf, training, type = "prob")[, 2]  

pred.test <- predict(classify.rf, testing)
pred.test.prob <- predict(classify.rf, testing, type = "prob")[, 2]  

# Create an empty vector to hold the combined probabilities and predicted classes
predicted_prob <- numeric(nrow(df7))
predicted_class <- factor(rep(NA, nrow(df7)), levels = levels(df7$target))

# Assign predicted probabilities and classes based on the original indices
predicted_prob[svm.index] <- pred.train.prob
predicted_prob[-svm.index] <- pred.test.prob

predicted_class[svm.index] <- pred.train
predicted_class[-svm.index] <- pred.test

# Combine into a data frame using the original row names from df7
rf_predictions <- data.frame(
  rf_class = predicted_class,
  rf_prob = predicted_prob,
  row.names = rownames(df7)  # Use the original row names from df7
)

# View the combined data
head(rf_predictions)

# Let's see if we can boost the power of the KNN, SVM, and rf models using XGBoost
# and combine the with the original data

df8 <- df7

df9 <- cbind(df8, svm_predictions, knn_predictions, rf_predictions)

# Load the necessary libraries
library(xgboost)
library(Matrix)
library(class)

# Assuming your data is in a DataFrame called df9

# Step 1: Convert the target variable to numeric
df9$target <- as.numeric(df9$target) - 1 # "fail" becomes 0, "pass" becomes 1

unique(df9$target)

# Convert any necessary factors to numeric (if applicable)
df9$svm_pred <- as.numeric(df9$svm_pred) - 1
df9$rf_class <- as.numeric(df9$rf_class) - 1

# Step 2: Use the existing model predictions as features along with original features
# Assuming the original features are already in df9 and you have svm_pred, svm_prob, knn_prob, rf_class, rf_prob in df9

# Identify relevant columns
predictor_columns <- c('percent_students_with_disabilities', 'economic_need_index', 'percent_in_temp_housing',
                       'years_of_principal_experience_at_this_school', 'federal', 
                       'years_of_principal_experience_at_this_school_na', 'teacher_attendance_rate_na', 
                       'antifederal', 'svm_pred', 'svm_prob', 'knn_prob', 'rf_class', 'rf_prob')

# Create the feature matrix X
X <- df9[, predictor_columns]

# Step 3: Convert the feature matrix to a matrix format for XGBoost
X_matrix <- as.matrix(X)

# Step 4: Prepare the DMatrix for XGBoost
dtrain <- xgb.DMatrix(data = X_matrix, label = df9$target)

# Step 5: Train the XGBoost model using the combined features
params <- list(
  objective = "binary:logistic", # Binary classification
  eval_metric = "aucpr",       # Evaluation metric
  max_depth = 6,                 # Maximum depth of a tree
  eta = 0.3,                     # Learning rate
  nthread = 2                    # Number of threads to use
)

# Train the XGBoost model
xgb_model <- xgb.train(
  params = params,
  data = dtrain,
  nrounds = 100,                 # Number of boosting rounds
  watchlist = list(train = dtrain),
  verbose = 1
)

# Step 6: Evaluate the model (optional)
predictions <- predict(xgb_model, X_matrix)
predicted_labels <- ifelse(predictions > 0.5, 1, 0)

# Confusion matrix to evaluate the predictions
table(Predicted = predicted_labels, Actual = df9$target)

#      fail  pass
# fail  752     0
# pass    0   218
#
# Kappa : 1.000