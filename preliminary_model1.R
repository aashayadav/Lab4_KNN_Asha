
library(tidyverse)
library(tidymodels)
library(doParallel)
library(kknn)


# Load the data

full_train <- read_csv("data/train.csv") %>% # relative path
  mutate(classification = factor(classification, # outcome variable classification into factor
                                 levels = 1:4,
                                 labels = c("far below", "below", "meets", 
                                            "exceeds"),
                                 ordered = TRUE))


full_train <- dplyr::slice_sample(full_train, prop = 0.02) 

# Split data
d <- initial_split(full_train)
d_train <- training(d)           # training set
d_test <- testing(d)             # testing set
cv <- vfold_cv(d_train)


# Recipe
rec <- recipe(classification ~ lat + lon + econ_dsvntg + tag_ed_fg + sp_ed_fg + tst_bnch +
                tst_dt, data = d_train) %>%
  step_mutate(tst_dt = lubridate::mdy_hms(tst_dt)) %>%
  update_role(tst_dt, new_role = "time_index") %>%  # alters an existing role in the recipe
  step_rollimpute(all_numeric()) %>%  # substitute missing values of numeric variables by the measure of location e.g. median
  update_role(tst_dt, new_role = "predictor") %>%  #?
  step_mutate(tst_dt = as.numeric(tst_dt)) %>%   # setting date variable as numeric
  step_medianimpute(all_numeric()) %>% #substitute missing values of numeric variables by the training set median of those variables.
  step_novel(all_nominal(), -all_outcomes()) %>% # assign a previously unseen factor level to a new value.
  step_unknown(all_nominal(), -all_outcomes()) %>% # assign a missing value in a factor level to 'unknown'.
  step_dummy(all_nominal(), -all_outcomes()) %>% # convert nominal data in numeric binary model for the levels of the original data.
  step_nzv(all_predictors()) %>% # removes highly sprarse or unbalanced variables.
  step_normalize(all_predictors()) # normalize numeric variables to have SD of 1 and mean of 0

# parallel processing

all_cores <- parallel::detectCores(logical = FALSE)

cl <- makePSOCKcluster(all_cores)
registerDoParallel(cl)
foreach::getDoParWorkers()
clusterEvalQ(cl, {library(tidymodels)})

# Preliminary kknn model

mod <- nearest_neighbor() %>%
  set_engine("kknn") %>%
  set_mode("classification")

preliminary_fit <- fit_resamples(mod, rec, cv)
saveRDS(preliminary_fit, "models/preliminary_fit.Rds")


