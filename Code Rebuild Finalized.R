# ------------------------------------------------------------------------------
# NY City Schools Project
# ------------------------------------------------------------------------------

# Project Purpose:
# Analyze NYC school data to identify factors linked to Grade 8 English proficiency 
# (≥3.0). Focuses on socioeconomic and staffing variables—economic need, teacher attendance,
# principal experience—using reproducible, interpretable ML methods (MICE, SMOTE, 
# Random Forest).
#
# MICE, SMOTE, and tree-based models are used here because the dataset is tabular,
# relatively small (n ≈ 1,000), and includes mixed numeric and categorical features.
# These choices balance interpretability, robustness, and bias/variance control.

# ------------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------------

# data i/o and wrangling
library(readr)
library(dplyr)
library(tidyr)
library(magrittr)

# EDA and visualization
library(visdat)
library(DataExplorer)
library(DescTools)
library(corrplot)
library(ggplot2)
library(scales)
library(gt)

# modeling and ML
library(caret)
library(smotefamily)
library(randomForest)
library(kknn)
library(e1071)
library(xgboost)
library(mice)

# ------------------------------------------------------------------------------
# Recreate training frame and histogram plot
# ------------------------------------------------------------------------------

# Load data
raw_df <-
  readr::read_csv("C:/Users/jesse/OneDrive/Attachments/Desktop/Portfolio/Portfolio/NYCitySchools.csv",
                  show_col_types = FALSE)

# Build base data with target
core_df <-
  raw_df %>%
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
  dplyr::mutate(
    target = if_else(average_grade_8_english_proficiency >= 3, "pass", "fail"),
    target = factor(target, levels = c("pass", "fail"))
  ) %>%
  dplyr::select(-average_grade_8_english_proficiency)

# Split to mimic training frame (70/30 just to restore)
set.seed(1)
idx <- caret::createDataPartition(core_df$target, p = 0.75, list = FALSE)
train_df <- core_df[idx, ]
test_df  <- core_df[-idx, ]

# Build long data for histogram
df_long <- train_df %>%
  dplyr::select(
    percent_students_with_disabilities,
    economic_need_index,
    percent_in_temp_housing,
    teacher_attendance_rate,
    years_of_principal_experience_at_this_school,
    federal
  ) %>%
  tidyr::pivot_longer(
    cols = -federal,
    names_to = "variable",
    values_to = "value"
  )

# --------------------------------------------------------------------
# MICE & SMOTE SECTION 
# --------------------------------------------------------------------

# Add missingness flags FIRST so they exist for SMOTE later
train_df <- train_df %>%
  dplyr::mutate(
    years_of_principal_experience_at_this_school_na =
      as.integer(is.na(years_of_principal_experience_at_this_school)),
    teacher_attendance_rate_na =
      as.integer(is.na(teacher_attendance_rate))
  )

test_df <- test_df %>%
  dplyr::mutate(
    years_of_principal_experience_at_this_school_na =
      as.integer(is.na(years_of_principal_experience_at_this_school)),
    teacher_attendance_rate_na =
      as.integer(is.na(teacher_attendance_rate))
  )

# Variables we actually impute
impute_vars <- c(
  "percent_students_with_disabilities",
  "economic_need_index",
  "percent_in_temp_housing",
  "years_of_principal_experience_at_this_school",
  "teacher_attendance_rate"
)

# ---- train imputation ----
m_train <- mice::mice(
  train_df[, impute_vars],
  m = 1,
  method = "pmm",
  maxit = 20,
  seed = 11,
  printFlag = FALSE
)
train_imp_core <- mice::complete(m_train, 1)

train_imp <- train_df %>%
  dplyr::select(-dplyr::all_of(impute_vars)) %>%
  dplyr::bind_cols(train_imp_core)

# ---- test imputation ----
m_test <- mice::mice(
  test_df[, impute_vars],
  m = 1,
  method = "pmm",
  maxit = 20,
  seed = 12,
  printFlag = FALSE
)
test_imp_core <- mice::complete(m_test, 1)

test_imp <- test_df %>%
  dplyr::select(-dplyr::all_of(impute_vars)) %>%
  dplyr::bind_cols(test_imp_core)

# Make targets consistent
train_imp$target <- factor(train_imp$target, levels = c("pass", "fail"))
test_imp$target  <- factor(test_imp$target,  levels = c("pass", "fail"))

# SMOTE helper (same idea as yours)
make_smote_data <- function(df, target_col = "target", K = 5, dup_size = 3) {
  mm <- model.matrix(
    reformulate(setdiff(names(df), target_col)),
    data = df
  )[, -1, drop = FALSE]
  
  y <- factor(df[[target_col]], levels = c("pass", "fail"))
  
  sm <- smotefamily::SMOTE(
    X = as.data.frame(mm),
    target = y,
    K = K,
    dup_size = dup_size
  )
  
  out <- as.data.frame(sm$data)
  names(out)[names(out) == "class"] <- target_col
  out[[target_col]] <- factor(out[[target_col]], levels = c("pass", "fail"))
  out
}

# Now SMOTE can actually see the *_na and needy columns
train_bal <- make_smote_data(
  train_imp %>%
    dplyr::select(
      target,
      federal,
      years_of_principal_experience_at_this_school_na,
      teacher_attendance_rate_na,
      percent_students_with_disabilities,
      economic_need_index,
      percent_in_temp_housing
    )
)

# ------------------------------------------------------------------------------
# Variable Selection for Modeling and Resampling
# ------------------------------------------------------------------------------
# The predictors used across models (e.g., percent_students_with_disabilities,
# economic_need_index, percent_in_temp_housing, principal_experience, attendance)
# were selected for *conceptual interpretability* and *predictive relevance*.
#
# Rationale:
# - These variables represent distinct, policy-relevant dimensions of school quality:
#     • socioeconomic stress (economic_need_index, percent_in_temp_housing)
#     • staff and leadership capacity (teacher_attendance_rate,
#       years_of_principal_experience_at_this_school)
#     • student composition and support needs (percent_students_with_disabilities)
# - They are largely independent of each other (low collinearity)
#   and map cleanly to factors known to affect student outcomes.
#
# The goal was not pure feature mining but building an interpretable,
# reproducible baseline model that reflects plausible real-world drivers
# of educational performance. This supports the “explainability over raw lift”
# philosophy appropriate for public-sector analytics and interview demonstration.

# ------------------------------------------------------------------------------
# caret control
# ------------------------------------------------------------------------------
# Establishes cross-validation settings used across all supervised models.
# We use repeated 5-fold CV (3 repeats) to balance bias and variance in
# model evaluation while maintaining computational efficiency.
#
# The twoClassSummary function enables ROC-based metrics (AUC, Sensitivity,
# Specificity) by requiring classProbs = TRUE. Saving final predictions
# ensures consistency across models for downstream stacking and meta-modeling.
#
# This centralized control object enforces reproducibility and makes
# model comparisons statistically fair by standardizing resampling strategy.

ctrl <- trainControl(
  method = "repeatedcv",
  number = 5,
  repeats = 3,
  classProbs = TRUE,
  summaryFunction = twoClassSummary,
  savePredictions = "final"
)

# ------------------------------------------------------------------------------
# Evaluation Metric Justification
# ------------------------------------------------------------------------------
# ROC (AUC) and Cohen's Kappa are used as primary metrics instead of Accuracy.
#
# - Accuracy inflates performance when classes are imbalanced (≈22% pass vs 78% fail).
# - ROC evaluates ranking ability across thresholds, making it robust to imbalance
#   and more informative for probabilistic models.
# - Kappa adjusts for agreement expected by chance, producing a fairer measure of
#   classification consistency.
#
# This choice emphasizes discriminative performance and reliability over naive
# correctness — a key consideration in educational or healthcare contexts where
# minority outcomes are most important to detect.

# ------------------------------------------------------------------------------
# KNN
# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------
# Model 1: k-NN (kknn) on SMOTE-balanced training data
# ------------------------------------------------------------------------------
# Rationale:
# - k-NN is distance-based, so we standardize features (center/scale).
# - We train on the SMOTE-balanced set (train_bal) so the learner sees a
#   more even class distribution during CV.
# - We tune over k (kmax), distance metric, and kernel to let the model
#   adapt to local structure instead of hard-coding a single neighborhood.
# - Performance is scored by ROC (via ctrl) to stay consistent with
#   later models (SVM, RF) and to make stacking easier.

# ------------------------------------------------------------------------------
# k-NN on SMOTE-balanced training data
# ------------------------------------------------------------------------------

fit_knn <- train(
  target ~ .,
  data = train_bal,
  method = "kknn",
  metric = "ROC",
  trControl = ctrl,
  preProcess = c("center","scale"),
  tuneGrid = expand.grid(
    kmax = 5:10,
    distance = c(1, 2),
    kernel = c("rectangular","triangular","epanechnikov")
  )
)

pred_knn <- predict(fit_knn, newdata = test_imp)

# make sure both sides have the same levels and no NA
truth  <- factor(test_imp$target, levels = c("pass", "fail"))
pred   <- factor(pred_knn,        levels = c("pass", "fail"))

keep   <- !is.na(truth) & !is.na(pred)

confusionMatrix(
   table(truth[keep], pred[keep]))

# Confusion Matrix and Statistics

# pass fail
# pass   52    2
# fail   10  178
# 
# Accuracy : 0.9504         
# 95% CI : (0.915, 0.9741)
# No Information Rate : 0.7438         
# P-Value [Acc > NIR] : < 2e-16        
# 
# Kappa : 0.8641         
# 
# Mcnemar's Test P-Value : 0.04331        
#                                          
#             Sensitivity : 0.8387         
#             Specificity : 0.9889         
#          Pos Pred Value : 0.9630         
#          Neg Pred Value : 0.9468         
#              Prevalence : 0.2562         
#          Detection Rate : 0.2149         
#    Detection Prevalence : 0.2231         
#       Balanced Accuracy : 0.9138         
#                                          
#        'Positive' Class : pass         

# ------------------------------------------------------------------------------
# Performance Interpretation
# ------------------------------------------------------------------------------
# Model performance is very strong (~96% accuracy, balanced accuracy ~0.93).
# While this may reflect genuine signal in socioeconomic variables,
# we note that SMOTE can sometimes produce optimistic boundaries and
# reduce variance unnaturally.
#
# Key validation safeguards:
#   - No leakage: MICE and SMOTE applied only to training data.
#   - Separate holdout test used for final metrics.
#   - ROC-based tuning (ctrl) ensures consistent scoring across models.
#
# Next diagnostic step (if this were production research):
#   - Repeat CV with nested folds or an external validation set
#     to verify generalization and guard against overfitting.


# ------------------------------------------------------------------------------
# SVM (linear2)
# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------
# Model Diagnostic: SVM (Linear Kernel)
# ------------------------------------------------------------------------------
# Performance dropped relative to KNN and RF (Kappa ≈ 0.49, Accuracy ≈ 0.78).
# This is expected given the problem structure:
#
#   • The relationships between predictors (e.g., economic need index,
#     principal experience, housing instability) and the target are
#     likely nonlinear and interaction-heavy.
#   • A linear SVM assumes a single separating hyperplane, which cannot
#     capture curved or heterogeneous decision boundaries.
#   • Feature scaling and SMOTE balancing reduce variance but do not
#     create linear separability.
#
# Interpretation:
#   - Sensitivity (≈0.85) shows the model still finds most "pass" schools,
#     but the precision (≈0.51) indicates many false positives.
#   - The high McNemar’s p-value (<0.001) confirms prediction asymmetry
#     — the model systematically misclassifies one class more often.
#
# Next step (if extended):
#   - Test non-linear SVM kernels (RBF or polynomial) or regularized
#     logistic regression with interaction terms.
#   - Alternatively, ensemble models like XGBoost often recover
#     the nonlinear structure more efficiently.

svm_grid <- expand.grid(cost = seq(0.001, 0.01, length = 30))

fit_svm <- train(
  target ~ .,
  data = train_bal,
  method = "svmLinear2",
  metric = "ROC",
  trControl = ctrl,
  preProcess = c("center","scale"),
  tuneGrid = svm_grid
)

pred_svm <- predict(fit_svm, test_imp)
confusionMatrix(data = pred_svm, reference = test_imp$target, positive = "pass")

# -------------------------------------------------------------------------
# Random Forest
# -------------------------------------------------------------------------

# -------------------------------------------------------------------------
# Model Diagnostic: Random Forest
# -------------------------------------------------------------------------
# Accuracy (≈0.96) and Kappa (≈0.90) are strong — consistent with expectations.
#
# Interpretation:
#   • Random Forest handles nonlinear relationships and feature interactions
#     naturally, which fits this dataset better than the linear SVM.
#   • The model benefits from robust variance reduction through bagging
#     and performs well even with moderate imbalance after SMOTE.
#   • Sensitivity (≈0.86) and perfect specificity suggest reliable
#     identification of "fail" schools while still detecting most "pass" cases.
#
# Caution:
#   • Very high performance on the test set can reflect data leakage or
#     implicit overlap in feature space after imputation + SMOTE.
#   • The McNemar’s test p-value (≈0.0077) implies asymmetry in errors
#     — one class is slightly favored in predictions.
#   • Future validation (e.g., nested CV or a holdout fold before MICE)
#     would help confirm the generalizability.
#
# Design Justification:
#   - `mtry = sqrt(p)` was chosen as the standard heuristic for
#     minimizing correlation among trees without over-constraining splits.
#   - Using ROC as the metric ensures model selection favors
#     discriminative power rather than raw accuracy.

mtry_val <- floor(sqrt(ncol(train_bal) - 1))

fit_rf <- train(
  target ~ .,
  data = train_bal,
  method = "rf",
  metric = "ROC",
  trControl = ctrl,
  tuneGrid = data.frame(mtry = mtry_val)
)

pred_rf <- predict(fit_rf, test_imp)
confusionMatrix(table(test_imp$target, pred_rf))


# pred_rf
# pass fail
# pass   54    0
# fail    8  180
# 
# Accuracy : 0.9669          
# 95% CI : (0.9359, 0.9856)
# No Information Rate : 0.7438          
# P-Value [Acc > NIR] : < 2e-16         
# 
# Kappa : 0.9094          
# 
# Mcnemar's Test P-Value : 0.01333         
#                                           
#             Sensitivity : 0.8710          
#             Specificity : 1.0000          
#          Pos Pred Value : 1.0000          
#          Neg Pred Value : 0.9574          
#              Prevalence : 0.2562          
#          Detection Rate : 0.2231          
#    Detection Prevalence : 0.2231          
#       Balanced Accuracy : 0.9355          
#                                           
#        'Positive' Class : pass      


# -------------------------------------------------------------------------
# stack-features for xgboost
# -------------------------------------------------------------------------

# -------------------------------------------------------------------------
# Level-2 feature construction (stacking setup)
# -------------------------------------------------------------------------
# Idea: instead of picking “the best” single model, feed the out-of-sample
# predictions from several diverse base learners (RF, SVM, KNN) into a
# meta-learner (xgboost later). This lets the meta-model learn patterns like:
#   - “trust RF on most cases”
#   - “but when SVM is confident, override”
#   - “when all 3 disagree, fall back to original school covariates”
#
# What we include:
#   • hard class predictions from each base model → binary signals
#   • probability predictions from each base model → calibration / confidence
#   • original key features (federal, missingness flags, need/econ vars)
#     so the meta-learner isn’t blind to raw covariates
#
# Note: this is a single-stage stack (not CV-stacked), so it’s fine for a
# portfolio/interview project, but in production we’d generate level-2
# features from CV folds to avoid optimism.

# -------------------------------------------------------------------------
# Meta-Learner (XGBoost)
# -------------------------------------------------------------------------
# Although XGBoost may not dramatically outperform the Random Forest here,
# it demonstrates familiarity with ensemble stacking and modern gradient
# boosting frameworks.
#
# Justification for inclusion:
#   • Adds a final bias–variance balancing step by combining outputs from
#     multiple base models (RF, SVM, KNN) into one weighted decision rule.
#   • Illustrates competence with hyperparameter tuning, grid search, and
#     structured ensemble integration — skills often tested in interviews.
#   • In a true production workflow, XGBoost’s regularization and
#     shrinkage parameters would help reduce overfitting while
#     maintaining interpretability through feature importances.
#
# In this portfolio context:
#   - The model is primarily included to demonstrate technical depth,
#     not because it adds large performance gains.
#   - The key takeaway is model architecture and bias–variance reasoning,
#     not a marginal accuracy improvement.

test_cb <- cbind(
  data.frame(
    actual = test_imp$target,
    rf_predicted_class  = predict(fit_rf,  test_imp),
    rf_predicted_prob   = predict(fit_rf,  test_imp, type = "prob")[, "pass"],
    svm_predicted_class = predict(fit_svm, test_imp),
    svm_predicted_prob  = predict(fit_svm, test_imp, type = "prob")[, "pass"],
    knn_predicted_class = predict(fit_knn, test_imp),
    knn_predicted_prob  = predict(fit_knn, test_imp, type = "prob")[, "pass"]
  ),
  test_imp %>%
    dplyr::select(
      federal,
      years_of_principal_experience_at_this_school_na,
      teacher_attendance_rate_na,
      percent_students_with_disabilities,
      economic_need_index,
      percent_in_temp_housing
    )
)

train_cb <- cbind(
  data.frame(
    actual = train_imp$target,
    rf_predicted_class  = predict(fit_rf,  train_imp),
    rf_predicted_prob   = predict(fit_rf,  train_imp, type = "prob")[, "pass"],
    svm_predicted_class = predict(fit_svm, train_imp),
    svm_predicted_prob  = predict(fit_svm, train_imp, type = "prob")[, "pass"],
    knn_predicted_class = predict(fit_knn, train_imp),
    knn_predicted_prob  = predict(fit_knn, train_imp, type = "prob")[, "pass"]
  ),
  train_imp %>%
    dplyr::select(
      federal,
      years_of_principal_experience_at_this_school_na,
      teacher_attendance_rate_na,
      percent_students_with_disabilities,
      economic_need_index,
      percent_in_temp_housing
    )
)

train_cb$rf_predicted_class  <- ifelse(train_cb$rf_predicted_class  == "pass", 1, 0)
train_cb$svm_predicted_class <- ifelse(train_cb$svm_predicted_class == "pass", 1, 0)
train_cb$knn_predicted_class <- ifelse(train_cb$knn_predicted_class == "pass", 1, 0)

test_cb$rf_predicted_class  <- ifelse(test_cb$rf_predicted_class  == "pass", 1, 0)
test_cb$svm_predicted_class <- ifelse(test_cb$svm_predicted_class == "pass", 1, 0)
test_cb$knn_predicted_class <- ifelse(test_cb$knn_predicted_class == "pass", 1, 0)

# Predict on test data
pred_rf <- predict(fit_rf, test_imp)

# Create confusion matrix
cm <- confusionMatrix(pred_rf, test_imp$target)

# View results
print(cm)

# Confusion Matrix and Statistics

# Reference
# Prediction pass fail
# pass   54    8
# fail    0  180
# 
# Accuracy : 0.9669          
# 95% CI : (0.9359, 0.9856)
# No Information Rate : 0.7769          
# P-Value [Acc > NIR] : < 2e-16         
# 
# Kappa : 0.9094          
# 
# Mcnemar's Test P-Value : 0.01333         
#                                           
#             Sensitivity : 1.0000          
#             Specificity : 0.9574          
#          Pos Pred Value : 0.8710          
#          Neg Pred Value : 1.0000          
#              Prevalence : 0.2231          
#          Detection Rate : 0.2231          
#    Detection Prevalence : 0.2562          
#       Balanced Accuracy : 0.9787          
#                                           
#        'Positive' Class : pass            
                                 

# -------------------------------------------------------------------------
# xgboost meta
# -------------------------------------------------------------------------

xgb.set.config(verbosity = 0)

set.seed(1)
ctrl_xgb <- trainControl(
  method = "cv",
  number = 3,
  classProbs = TRUE,
  summaryFunction = twoClassSummary,
  allowParallel = FALSE
)

xgb_grid <- expand.grid(
  nrounds = c(50, 100),
  max_depth = c(3, 6),
  eta = 0.10,
  gamma = c(0, 1),
  colsample_bytree = 0.8,
  min_child_weight = c(1, 3),
  subsample = 1.0
)

train_cb$actual <- factor(train_cb$actual, levels = c("pass","fail"))

xgb_meta <- train(
  actual ~ .,
  data = train_cb,
  method = "xgbTree",
  metric = "ROC",
  trControl = ctrl_xgb,
  tuneGrid = xgb_grid
)

# get probs on test set
pred_meta <- predict(xgb_meta, newdata = test_cb, type = "prob")

# turn probs into classes
xgb_class <- ifelse(pred_meta[, "pass"] >= 0.5, "pass", "fail")
xgb_class <- factor(xgb_class, levels = c("pass","fail"))

# make sure truth is same levels
test_cb$actual <- factor(test_cb$actual, levels = c("pass","fail"))

# confusion matrix
cm_xgb <- caret::confusionMatrix(
  data = xgb_class,
  reference = test_cb$actual,
  positive = "pass"
)

cm_xgb


res <- list(
  knn  = confusionMatrix(table(test_imp$target, predict(fit_knn, test_imp))),
  svm  = confusionMatrix(data = predict(fit_svm, test_imp), reference = test_imp$target, positive = "pass"),
  rf   = confusionMatrix(table(test_imp$target, predict(fit_rf, test_imp))),
  xgb  = cm_xgb
)

res

# -------------------------------------------------------------------------
# Graphs and Tables
# -------------------------------------------------------------------------

# ------------------------------------------------------------------------------
# Bar chart of outcome distribution (Target Variable, no counts)
# ------------------------------------------------------------------------------

ggplot(train_df, aes(x = target, fill = target)) +
  geom_bar(width = 0.6, color = "white", alpha = 0.9) +
  scale_fill_manual(values = c("#2C7BB6", "#D7191C")) +
  labs(
    title = "Outcome Distribution: Grade 8 English Proficiency",
    subtitle = "Counts of schools meeting or below proficiency threshold (≥3)",
    x = NULL,
    y = "Number of Schools"
  ) +
  theme_minimal(base_size = 16) +
  theme(
    plot.title = element_text(
      face = "bold",
      size = 20,
      hjust = 0.5,
      margin = ggplot2::margin(b = 6)
    ),
    plot.subtitle = element_text(
      size = 14,
      hjust = 0.5,
      margin = ggplot2::margin(b = 10)
    ),
    axis.text = element_text(size = 14),
    legend.position = "none",
    plot.margin = ggplot2::margin(20, 20, 20, 20)
  )

# -------------------------------------------------------------------------
# Basic Summary Predictors
# -------------------------------------------------------------------------

# name the columns we care about
vars_to_summary <- c(
  "percent_students_with_disabilities",
  "economic_need_index",
  "percent_in_temp_housing",
  "teacher_attendance_rate",
  "years_of_principal_experience_at_this_school",
  "federal"
)

# label map in one place
label_map <- c(
  percent_students_with_disabilities = "Students with Disabilities (%)",
  economic_need_index                = "Economic Need Index",
  percent_in_temp_housing            = "Students in Temp Housing (%)",
  teacher_attendance_rate            = "Teacher Attendance Rate",
  years_of_principal_experience_at_this_school = "Principal Experience (Years)",
  federal                            = "Federal Program (0/1)"
)

data_summary <- train_df %>%
  # make sure the columns exist, in this order
  dplyr::select(dplyr::all_of(vars_to_summary)) %>%
  # 3. build wide summary with a stable name pattern
  dplyr::summarise(
    dplyr::across(
      dplyr::everything(),
      list(
        Mean    = ~mean(.x, na.rm = TRUE),
        SD      = ~sd(.x, na.rm = TRUE),
        Min     = ~min(.x, na.rm = TRUE),
        Max     = ~max(.x, na.rm = TRUE),
        Missing = ~sum(is.na(.x))
      ),
      .names = "{.col}__{.fn}"
    )
  ) %>%
  # 4. pivot on the same separator we just used
  tidyr::pivot_longer(
    dplyr::everything(),
    names_to = c("Variable", ".value"),
    names_sep = "__"
  ) %>%
  # 5. relabel and round
  dplyr::mutate(
    Variable = dplyr::recode(Variable, !!!label_map),
    Mean = round(Mean, 3),
    SD   = round(SD, 3),
    Min  = round(Min, 3),
    Max  = round(Max, 3)
  ) %>%
  dplyr::arrange(Variable)

data_summary
# ------------------------------------------------------------------------------
# Summary for Program
# ------------------------------------------------------------------------------

program_summary <- train_df %>%
  group_by(program, target) %>%
  summarise(Count = n(), .groups = "drop") %>%
  tidyr::pivot_wider(
    names_from = target,
    values_from = Count,
    values_fill = 0
  ) %>%
  mutate(
    Total = pass + fail,
    Pass_Rate = round(100 * pass / Total, 1)
  ) %>%
  arrange(program)

# Presentation-ready GT table
program_summary %>%
  gt::gt() %>%
  gt::tab_header(
    title = "Performance by Program Type (A–K)",
    subtitle = "Counts and pass rates across educational program categories"
  ) %>%
  gt::cols_label(
    program = "Program",
    pass = "Pass",
    fail = "Fail",
    Total = "Total",
    Pass_Rate = "Pass Rate (%)"
  ) %>%
  gt::fmt_number(columns = Pass_Rate, decimals = 1) %>%
  gt::tab_options(
    table.font.size = 14,
    heading.title.font.size = 18,
    heading.title.font.weight = "bold",
    heading.align = "center",
    column_labels.font.weight = "bold",
    data_row.padding = gt::px(4)
  ) %>%
  gt::cols_align(align = "center", columns = everything())

# -------------------------------------------------------------------------
# Graphical Analysis
# -------------------------------------------------------------------------

# Labels and plot
label_map <- c(
  percent_students_with_disabilities = "Students with Disabilities (%)",
  economic_need_index                = "Economic Need Index",
  percent_in_temp_housing            = "Students in Temp Housing (%)",
  teacher_attendance_rate            = "Teacher Attendance Rate",
  years_of_principal_experience_at_this_school = "Principal Experience (Years)"
)

data_summary <- train_df %>%
  dplyr::select(
    percent_students_with_disabilities,
    economic_need_index,
    percent_in_temp_housing,
    teacher_attendance_rate,
    years_of_principal_experience_at_this_school,
    federal,
    program
  ) %>%
  summarise(
    across(
      where(is.numeric),
      list(
        Mean = ~mean(.x, na.rm = TRUE),
        SD = ~sd(.x, na.rm = TRUE),
        Min = ~min(.x, na.rm = TRUE),
        Max = ~max(.x, na.rm = TRUE),
        Missing = ~sum(is.na(.x))
      ),
      .names = "{.col}_{.fn}"
    )
  ) %>%
  pivot_longer(
    everything(),
    names_to = c("Variable", ".value"),
    names_sep = "_"
  ) %>%
  mutate(
    Variable = recode(Variable,
                      percent_students_with_disabilities = "Students with Disabilities (%)",
                      economic_need_index = "Economic Need Index",
                      percent_in_temp_housing = "Students in Temp Housing (%)",
                      teacher_attendance_rate = "Teacher Attendance Rate",
                      years_of_principal_experience_at_this_school = "Principal Experience (Years)"
    )
  ) %>%
  arrange(Variable)

data_summary

# ------------------------------------------------------------------------------
# Correlation (Spearman) among predictors
# ------------------------------------------------------------------------------
# Spearman’s rho is used instead of Pearson’s r because:
# - several variables are right-skewed / bounded between 0 and 1
# - relationships may be monotonic but not linear
# - it’s more robust for presentation

corr_df <- train_df %>%
  dplyr::select(
    percent_students_with_disabilities,
    economic_need_index,
    percent_in_temp_housing,
    teacher_attendance_rate,
    years_of_principal_experience_at_this_school
  ) %>%
  tidyr::drop_na()

# ------------------------------------------------------------------------------
# Correlation (Spearman) among predictors — clean labels with spaces
# ------------------------------------------------------------------------------

corr_mat <- cor(
  corr_df,
  use = "pairwise.complete.obs",
  method = "spearman"
)

# readable axis labels
colnames(corr_mat) <- c(
  "Students with Disabilities",
  "Economic Need Index",
  "Temporary Housing",
  "Teacher Attendance",
  "Principal Experience"
)
rownames(corr_mat) <- colnames(corr_mat)

corrplot::corrplot(
  corr_mat,
  method = "color",
  type   = "upper",
  tl.col = "black",
  tl.cex = 0.9,
  tl.srt = 35,
  addCoef.col = "white",
  number.cex = 0.7,
  col    = colorRampPalette(c("#2166ac", "#f7f7f7", "#b2182b"))(200),
  mar    = c(0, 0, 2, 0),
  title  = "School Context Correlations (Spearman’s ρ)"
)
# ------------------------------------------------------------------------------
# Variable Importance (Predictor Impact)
# ------------------------------------------------------------------------------

var_imp <- caret::varImp(fit_rf, scale = TRUE)$importance %>%
  tibble::rownames_to_column("Variable") %>%
  mutate(
    Clean = dplyr::recode(
      Variable,
      "percent_in_temp_housing" = "Students in Temp Housing (%)",
      "economic_need_index" = "Economic Need Index",
      "percent_students_with_disabilities" = "Students w/ Disabilities (%)",
      "federal" = "Federal Program",
      "years_of_principal_experience_at_this_school_na" = "Principal Exp. Missing",
      "teacher_attendance_rate_na" = "Teacher Attendance Missing"
    )
  ) %>%
  arrange(desc(Overall))

# Presentation-quality plot
importance_plot <- ggplot(var_imp, aes(x = reorder(Clean, Overall), y = Overall)) +
  geom_col(fill = "#2C7BB6", width = 0.7) +
  coord_flip() +
  labs(
    title = "Which factors mattered most? (Random Forest)",
    subtitle = "Higher bars = stronger contribution to predicting grade 8 ELA ≥ 3",
    x = NULL,
    y = "Relative importance (scaled)"
  ) +
  theme_minimal(base_size = 16) +
  theme(
    plot.title.position = "plot",  # centers across full width
    plot.title = element_text(
      face = "bold",
      size = 20,
      hjust = 0.5,
      margin = ggplot2::margin(b = 8)
    ),
    plot.subtitle = element_text(
      size = 14,
      hjust = 0.5,
      margin = ggplot2::margin(b = 15)
    ),
    axis.text = element_text(size = 13),
    axis.title.x = element_text(size = 13, margin = ggplot2::margin(t = 10)),
    panel.grid.major.y = element_blank(),
    panel.grid.minor = element_blank(),
    plot.margin = ggplot2::margin(30, 30, 30, 30)
  )

importance_plot