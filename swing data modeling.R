library(tidymodels) 
library(vip) 
library(themis)
library(stacks)
tidymodels_prefer() 

# loading in data 
train <- read.csv(file.choose()) 
test <- read.csv(file.choose())

# basic feature engineering 
train <- train %>% 
  mutate(
    plat_adv = if_else(is_lhp == is_lhb, 1, 0), 
    count = paste(balls, strikes, sep = "-"), 
    count_type = case_when(
      count %in% c('0-1','0-2','1-2') ~ "behind", 
      count %in% c('0-0','1-1','2-2','3-2') ~ "even", 
      .default = "ahead"
    ), 
    launch_speed = bat_speed * 1.2, 
    max_ev = bat_speed * 1.23 + release_speed * 0.2116, 
    est_squared_up_rate = round(launch_speed / max_ev, 3), 
    is_bunt = if_else(swing_length < 4, 1, 0), 
    spray_angle = if_else(is_lhb == 1, -spray_angle, spray_angle), 
    pfx_x = if_else(is_lhb == 1, -pfx_x, pfx_x), 
    plate_x = if_else(is_lhb == 1, -plate_x, plate_x)
  ) 

test <- test %>% 
  mutate(
    plat_adv = if_else(is_lhp == is_lhb, 1, 0), 
    count = paste(balls, strikes, sep = "-"), 
    count_type = case_when(
      count %in% c('0-1','0-2','1-2') ~ "behind", 
      count %in% c('0-0','1-1','2-2','3-2') ~ "even", 
      .default = "ahead"
    ), 
    launch_speed = bat_speed * 1.2, 
    max_ev = bat_speed * 1.23 + release_speed * 0.2116, 
    est_squared_up_rate = round(launch_speed / max_ev, 3), 
    is_bunt = if_else(swing_length < 4, 1, 0), 
    spray_angle = if_else(is_lhb == 1, -spray_angle, spray_angle), 
    pfx_x = if_else(is_lhb == 1, -pfx_x, pfx_x), 
    plate_x = if_else(is_lhb == 1, -plate_x, plate_x) 
  ) 

# "adjusted" swing length using linear regression 
swing_length_lm <- 
  lm(swing_length ~ plate_x + plate_z + bat_speed + spray_angle
     + pfx_x + pfx_z + release_speed, data = train) 
summary(swing_length_lm) 
# pretty good results 
# adding count type makes no meaningful difference 


train$swing_length_adj <- predict(swing_length_lm, train)
train$contact_point <- train$swing_length - train$swing_length_adj 
test$swing_length_adj <- predict(swing_length_lm, test) 
test$contact_point <- test$swing_length - test$swing_length_adj 

# same idea as above, but with bat speed 
bat_speed_lm <- 
  lm(bat_speed ~ plate_x + plate_z + pfx_x + pfx_z + release_speed + 
       factor(count_type) + swing_length, data = train) 
summary(bat_speed_lm)
# note that adding spray angle here makes no difference 
# this is unlike swing length 
# variability accounted for is also less 

train$bat_speed_adj <- predict(bat_speed_lm, train) 
test$bat_speed_adj <- predict(bat_speed_lm, test)
train$speed_added <- train$bat_speed - train$bat_speed_adj 
test$speed_added <- test$bat_speed - test$bat_speed_adj 

# vertical approach angle 
calculate_vaa <- function(data) { 
  yf <- 17/12
  vy0 <- data$vy0 
  vz0 <- data$vz0
  ay <- data$ay
  az <- data$az
  vy_f <- -sqrt(vy0^2 - (2 * ay * (50 - yf))) 
  t <- (vy_f - vy0) / ay
  vz_f = vz0 + (az * t)
  vaa_rad <- atan2(vz_f, vy_f)
  vaa <- (180 + (180/pi)*(vaa_rad))*-1 
  return(vaa)
} 

# vertical release angle
calculate_vra <- function(data) {
  release_extension <- data$release_extension
  vy0 <- data$vy0 
  vz0 <- data$vz0
  ay <- data$ay
  az <- data$az
  vy_s <- -sqrt(vy0**2 - 2 * ay * (60.5 - release_extension - 50))
  t_s <- (vy_s - vy0) / ay
  vz_s <- vz0 - az * t_s
  vra <- -atan(vz_s / vy_s) * (180 / pi) 
  return(vra)
} 

# horizontal release angle 
calculate_hra <- function(data) {
  release_extension <- data$release_extension
  vx0 <- data$vx0
  vy0 <- data$vy0 
  ax <- data$ax
  ay <- data$ay
  vy_s <- -sqrt(vy0**2 - 2 * ay * (60.5 - release_extension - 50))
  t_s <- (vy_s - vy0) / ay
  vx_s <- vx0 - ax * t_s
  hra <- -atan(vx_s / vy_s) * (180 / pi) 
  return(hra)
}

train$vaa <- calculate_vaa(train) 
train$vra <- calculate_vra(train) 
train$hra <- calculate_hra(train) 

test$vaa <- calculate_vaa(test) 
test$vra <- calculate_vra(test) 
test$hra <- calculate_hra(test) 

# change outcome variable to factor 
train$outcome_code <- as.factor(train$outcome_code)

# only the features we want 
model_train <- train %>% 
  select(-c(uid, pitch_type, game_type, is_lhb, is_lhp, balls, strikes, 
            game_year, on_3b, on_2b, on_1b, outs_when_up, inning, sz_top, 
            sz_bot, vx0, vy0, vz0, ax, ay, az, effective_speed, is_bunt, 
            release_pos_y, pitch_number, pitch_name, 
            spin_axis, release_spin_rate, count, outcome)) %>% 
  relocate(outcome_code, .after = hra)

# cross_validation
set.seed(101) 
rf_folds <- vfold_cv(model_train, v = 2, strata = outcome_code) 
  
# model recipe 
rf_recipe <- recipe(outcome_code ~., data = model_train) %>% 
  step_dummy(all_nominal_predictors()) %>% 
  step_smote(over_ratio = 0.1, outcome_code)

# check to see if features have been generated correctly 
rf_recipe %>% 
  prep() %>% 
  bake(new_data = NULL) %>% 
  head() 

# specify the model type 
rf_spec <- 
  rand_forest(min_n = tune(), mtry = tune(), trees = 500) %>% 
  set_engine("ranger", importance = 'impurity') %>% 
  set_mode("classification") 

# join the recipe and model type in on workflow 
rf_wflow <- 
  workflow() %>% 
  add_recipe(rf_recipe) %>% 
  add_model(rf_spec) 

# sample hyperparameters with latin hypercube design 
set.seed(102) 
rf_grid <- grid_latin_hypercube(
  min_n(), 
  finalize(mtry(), select(model_train, -outcome_code)), 
  size = 20
)

# set number of CPU cores to use 
options(mc.cores = parallel::detectCores()) 
# don't need this if using light mode 
options(tidymodels.dark = T) 

# hyperparameter tuning 
set.seed(103)
rf_initial <- tune_grid(
  rf_wflow, 
  resamples = rf_folds, 
  grid = rf_grid, 
  control = control_grid(verbose = T), 
  metrics = metric_set(roc_auc)
) 

# stack different random forest models 
# this will result in an ensemble prediction 
rf_stack <- stacks() %>% 
  add_candidates(rf_tuning) %>% 
  blend_predictions() %>% 
  fit_members() 

# predict on test set 
# warning: this will take a long time and result in 10+ GB object 
# don't stack unless you have sufficient time + memory 
rf_preds <- predict(rf_stack, test, type = "prob") 

# export results to csv file 
test %>% 
  mutate(
    out = rf_preds$.pred_0, 
    single = rf_preds$.pred_1, 
    double = rf_preds$.pred_2, 
    triple = rf_preds$.pred_3, 
    home_run = rf_preds$.pred_4
  ) %>% 
  select(uid, out, single, double, triple, home_run) %>% 
  write.csv("submission_stack.csv", row.names = F) 

# bonus: non-stack single random forest fit 
best_params <- select_best(rf_tuning, metric = "roc_auc")

# fitting final model to training data 
rf_fit <- 
  finalize_workflow(rf_wflow, best_params) %>% 
  fit(data = model_train) 
    
# now we can graph feature importance using vip library 
rf_fit %>% 
  vip(n = 25) 

