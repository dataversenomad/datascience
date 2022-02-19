rm(list=ls())

setwd("/home/analytics/R/Projects/Time Series 2")

#Forecasting Libraries ----
library(forecast)    

#Machine Learning Libraries ----
library(glmnet)       
library(earth)        
library(kknn)         
library(randomForest) 
library(ranger)       
library(xgboost)     

#Machine Learning: Ensembles
library(modeltime.ensemble)

#Time Series ML ----
library(tidymodels)   
library(rules)        
library(modeltime)    

#Core Libraries ----
library(tidyverse)    
library(lubridate)    
library(timetk)       

# Timing & Parallel Processing
library(tictoc)
library(future)
library(doFuture)


#Extras

library(fs)
library(dplyr)
library(tidyquant)
library(readxl)




financials <- read_excel('/home/analytics/R/Projects/Time Series 2/00_data/FinancialsV3.xlsx',
                         sheet = 1, col_names = TRUE)

financials$Category...1 <- NULL

financials %>% glimpse()

fin_data <- financials %>%
  gather(key = "Month", value = "Value", c(Jan, Feb, Mar, Apr, May, 
                                           Jun, Jul, Aug, Sep, Oct, 
                                           Nov, Dec)) %>%
  group_by(LOB, Year, Category...3, Metric, Month) %>%
  summarise(Value = sum(Value)) %>%
  mutate(Month = case_when(
    Month == "Jan" ~ "01",
    Month == "Feb" ~ "02",
    Month == "Mar" ~ "03",
    Month == "Apr" ~ "04",
    Month == "May" ~ "05",
    Month == "Jun" ~ "06",
    Month == "Jul" ~ "07",
    Month == "Aug" ~ "08",
    Month == "Sep" ~ "09",
    Month == "Oct" ~ "10",
    Month == "Nov" ~ "11",
    Month == "Dec" ~ "12",
    TRUE ~ Month
  )) %>%  mutate(Date = paste0(Year,"-",Month,"-","01") %>% as_date()) %>% 
  ungroup() %>%
  rename(Category = Category...3) %>%
  mutate(Group = case_when(
    LOB == 'LOB 1' ~ 'G1',
    LOB == 'LOB 2' ~ 'G1',
    LOB == 'LOB 3' ~ 'G2',
    LOB == 'LOB 4' ~ 'G2',
    LOB == 'LOB 5' ~ 'G3',
    LOB == 'LOB 6' ~ 'G3',
    LOB == 'LOB 7' ~ 'G4',
    TRUE ~ LOB
  )) %>% filter_by_time(.start_date = "2017-01-01", 
                        .end_date = "2021-09-01") %>% 
  group_by(LOB, Year, Category, Metric, Month, Group) %>%
  
    summarise_by_time(
    .by        = "month",
    Value      = last(Value),
    .type      = "ceiling"
  ) %>% mutate(Date = Date %-time% "1 day")  %>% ungroup()


fin_data <- fin_data %>% filter(!(Date < "2021-07-31" &
                        Category == 'Forecast')  ) 

fin_data$Year <- NULL

fin_data$Month <- NULL

fin_data_ac_rev <- fin_data %>% filter(Category == 'Actuals' & 
                                         Metric == 'Revenue' ) 


# All Businesses in one

fin_data_ac_rev_bus <- fin_data %>% filter(Category == 'Actuals' & 
                                             Metric == 'Revenue') %>%
  select(Date, Value) %>%
  group_by(Date) %>% summarize(Value = sum(Value)) 


fin_data_ac_rev_bus %>%
  plot_time_series(Date, Value, .smooth = FALSE, 
                   .facet_scales = "free",
                   .title = "Commercial Business Revenue (Actuals)" )  


# All Businesses in one (including Forecast)

fin_data_rev_bus <- fin_data %>% filter(Category %in% 
                                          c('Actuals', 'Forecast') & 
                                          Metric == 'Revenue') %>%
  select(Category, Date, Value) %>%
  group_by(Category, Date) %>%
  summarize(Value = sum(Value)) 


fin_data_rev_bus %>% ungroup() %>%
  plot_time_series(Date, Value, .color_var = Category, .smooth = FALSE,
                   .title = "Commercial Business Revenue (Actuals vs Forecast)")


## UNDERSTAND SEASONALITY

fin_data_ac_rev_bus %>% plot_acf_diagnostics(Value, 
                                      log_interval_vec(Value, offset = 1000000),
                                      .lags = 100)  #1 year


## SEASONAL DESCOMPOSITION

fin_data_ac_rev_bus %>%
  plot_stl_diagnostics(Date, log_interval_vec(Value, offset = 1000000), 
                       .facet_scales = "free_y",
                       .feature_set = c( "observed", "season", "trend", "remainder"))


(208041 - 194056) / 208041


############################## FORECASTING MODELS ##############################

#Parallel Processing ----

registerDoFuture()
n_cores <- parallel::detectCores()
plan(
  strategy = cluster,
  workers  = parallel::makeCluster(n_cores)
)

#Full Data ----

full_data_tbl <- fin_data_ac_rev_bus %>%
  
  mutate(value_trans = Value %>% log_interval_vec(offset = 1000000) )  %>%
  mutate(value_trans = standardize_vec(value_trans)) %>%
  
  #  log_interval_vec(): 
  #  Using limit_lower: 0
  #Using limit_upper: 17009662.34704
  #Using offset: 1e+06
  #Standardization Parameters
  #mean: -1.28392047958657
  #standard deviation: 1.04797755829809
  
  future_frame(Date, .length_out = 3, .bind_data = TRUE) %>% 
  
  tk_augment_fourier(Date, .periods = c(30, 365), .K = 2) %>%
  
  rowid_to_column(var = "rowid") 

full_data_tbl

full_data_tbl %>% tail(3)

table(is.na(full_data_tbl$Value))


View(full_data_tbl)

data_prepared_tbl <- full_data_tbl %>%
  filter(!is.na(Value)) #%>% drop_na() (WE ARE NOT DROPPING NAs FROM LAGS)

#Data Preparation for - Future Data ----
future_tbl <- full_data_tbl %>%
  filter(is.na(Value))

# 2.0 TIME SPLIT ----

splits <- time_series_split(data_prepared_tbl, assess = 12, skip = 12, 
                            cumulative = TRUE)

splits %>%
  tk_time_series_cv_plan() %>%
  plot_time_series_cv_plan(Date, value_trans)

# 3.0 RECIPE ----

train_cleaned <- training(splits) 
train_cleaned$event_date <- NULL

View(train_cleaned)
View(data_prepared_tbl)

################################ RECIPES ###################################


colnames(train_cleaned)
train_cleaned$Value <- NULL

#including:fourier, lags, splines, order 2 (normalized)
recipe_spec <- recipe(value_trans ~ ., data = train_cleaned) %>%  
  update_role(rowid, new_role = "indicator") %>%
  step_timeseries_signature(Date) %>% 
  step_rm(matches("(.xts$)|(.iso$)|(hour)|(minute)|(second)|(am.pm)|(week)|(mday)|(day)|(half)|(quarter)")) %>% 
  step_rm(ends_with('Date_month')) %>% 
  step_normalize(Date_index.num, Date_year) %>% 
  step_ns(ends_with("index.num"), deg_free = 3) %>%
  #step_other(LOB) %>% #Collapse Some Categorical Levels
  step_dummy(all_nominal(), one_hot = TRUE)


#########################SEQUENTIAL MODELS ###############################


recipe_arima <- recipe_spec %>%
 
  step_rm(ends_with('rowid')) %>%
  step_rm(matches("(Date_month.lbl)")) %>%
 
  step_rm(contains('Date_sin')) %>%
  step_rm(contains('Date_year')) %>%
  step_rm(contains('Date_cos')) #%>%
#  step_fourier(Date, period = c(12), K = 2)

recipe_arima %>% prep() %>% juice %>% glimpse()

#### 1. ARIMA ---

model_spec_arima <- arima_reg(
  mode = "regression",
  seasonal_period = 12,
  non_seasonal_ar = 1,
  non_seasonal_differences = 1,
  non_seasonal_ma = 1,
  seasonal_ar = 2,
  seasonal_differences = 1,
  seasonal_ma = 0
)  %>%
  set_engine("arima")

workflow_fit_arima <- workflow() %>%
  add_model(model_spec_arima) %>%
  add_recipe(recipe_arima)  %>% #adding recipe
 
  fit(training(splits))

#### 2. ARIMA STL ----

model_spec_arima_STL <-  seasonal_reg(
  
  #seasonal_period_1 = "1 year" ,  
  seasonal_period_1 = 12   #monthly
  #seasonal_period_2 =  730,  # montly #730.001  #24*30
  #seasonal_period_3 =  24*365
  
) %>%
  set_engine("stlm_arima") 


workflow_fit_arima_STL <- workflow() %>%
  add_model(model_spec_arima_STL) %>%
  add_recipe(recipe_arima)  %>% #adding recipe
  fit(training(splits))

workflow_fit_arima_STL$fit$fit

### 3. ARIMA + XGBOOST

model_arima_boost <- arima_boost(
  
  seasonal_period = 12,  
  non_seasonal_ar = 1,
  non_seasonal_differences = 1,
  non_seasonal_ma = 2,
  seasonal_ar = 2,
  seasonal_differences = 1,
  seasonal_ma = 0,
  #  min_n = 10000

  
  mtry = 20,
  min_n = 1,
  tree_depth = 3,
  learn_rate = 0.25,
#  loss_reduction = 0.15,
  trees = 15

) %>%
  set_engine("arima_xgboost")

workflow_fit_arima_XG <- workflow() %>%
  add_model(model_arima_boost) %>%
  add_recipe(recipe_arima)  %>% #adding recipe
  fit(training(splits))


submodels_1_tbl <- modeltime_table(
  
  workflow_fit_arima,
  workflow_fit_arima_STL,
  workflow_fit_arima_XG
)

submodels_1_tbl %>%
  modeltime_accuracy(testing(splits)) %>%
  arrange(rmse)



submodels_1_tbl$.model

###################### NON SEQUENTIAL MODELS #################################

#1. NNET ----

wflw_fit_nnet <- workflow() %>%
  add_model(
    spec = mlp(
                 mode = "regression"
            
               ) %>% set_engine("nnet", verbose = 0, trace = 0)
  ) %>%
  add_recipe(recipe_spec %>% update_role(Date, new_role = "indicator")) %>%
  fit(train_cleaned)

summary(wflw_fit_nnet$fit$fit)
wflw_fit_nnet
wflw_fit_nnet$fit$fit


############################ JOINING DATA FLOWS #####################


submodels_1_tbl <- modeltime_table(
  
  workflow_fit_arima,
  workflow_fit_arima_STL,
  workflow_fit_arima_XG
)

submodels_1_tbl %>%
  modeltime_accuracy(testing(splits)) %>%
  arrange(rmse)

# HYPER PARAMETER TUNING ---- 

#RESAMPLES - K-FOLD ----- 

set.seed(123)
resamples_kfold <- train_cleaned %>% vfold_cv(v = 5)  

resamples_kfold %>%
  tk_time_series_cv_plan() %>%
  plot_time_series_cv_plan(Date, value_trans, .facet_ncol = 2)

cell_folds <- vfold_cv(train_cleaned)

################# NNET TUNE  #############################
library(dials)
tidymodels_prefer()
library(tidymodels)

model_spec_nnet_tune <- mlp(
  mode = "regression",
  hidden_units = 5,
  penalty = 0.3,
  epochs = tune(),
  dropout = 0,
  activation = 'linear'
  
) %>%
  set_engine("nnet", trace = 0)



wflw_spec_nnet_tune <- workflow() %>%
  add_model(model_spec_nnet_tune) %>%
  add_recipe(recipe_spec %>% update_role(Date, new_role = "indicator"))


#tune
tic()
set.seed(123)
tune_results_nnet <- wflw_spec_nnet_tune %>%
  tune_grid(
    resamples = resamples_kfold,
    param_info = parameters(wflw_spec_nnet_tune) %>%
      update(
        
        #epochs = epochs(c(10, 100)) , #,
        epochs = epochs(c(10, 1000))
        #hidden_units = hidden_units(c(1,2))
      ),
    grid = 10,
    control = control_grid(verbose = TRUE, allow_par = TRUE)
  )
toc()

tune_results_nnet %>% show_best("rmse", n = Inf)

# ** Finalize

wflw_fit_nnet_tuned <- wflw_spec_nnet_tune %>%
  finalize_workflow(select_best(tune_results_nnet, "rmse")) %>%
  fit(train_cleaned)

############################## JOINING MODELS pt 2 ##########################

submodels_2_tbl <- modeltime_table(
  wflw_fit_nnet_tuned
  
) %>%
  update_model_description(1, "NEURAL NETWORK 24-5-1") %>%  #update labels
  
  combine_modeltime_tables(submodels_1_tbl) %>%
  
  update_model_description(2, "ARIMA (1,1,1)(2,1,0)[12] + XREGS") %>%
  update_model_description(3, "ARIMA(0,0,1) + XREGS + STL") %>%
  update_model_description(4, "XGBOOST ARIMA(1,1,2)(2,1,0)[12] + XREGS") 


#Calibration ----
calibration_tbl <- submodels_2_tbl %>%
  modeltime_calibrate(testing(splits))

#Accuracy ----
calibration_tbl %>% 
  modeltime_accuracy() %>%
  arrange(rmse)

#Forecast Test ----

calibration_tbl %>%
  modeltime_forecast(
    new_data    = testing(splits),
    actual_data = data_prepared_tbl,
    keep_data   = TRUE ## 
  ) %>%
  
  plot_modeltime_forecast(
    .facet_ncol         = 4, 
    .conf_interval_show = FALSE,
    .interactive        = TRUE
  )


colnames(forecast_test_tbl)

forecast_test_tbl <- submodels_2_tbl %>%
  modeltime_forecast(
    new_data    = testing(splits),
    actual_data = data_prepared_tbl,
    keep_data   = TRUE 
  ) %>%
  
  mutate(
    value_trans= value_trans %>% standardize_inv_vec(mean = 1.29048740547333, 
                                                     sd = 0.676453813509239)  
  ) %>%
  mutate(
    .value = .value %>% standardize_inv_vec(mean = 1.29048740547333 , 
                                            sd = 0.676453813509239)  
  ) %>%
  
  mutate(
    value_trans= value_trans %>% log_interval_inv_vec(limit_lower = 0, 
                                                      limit_upper = 33511492.929216, 
                                                      offset = 1e+06)  
  ) %>%
  mutate(
    .value = .value %>% log_interval_inv_vec(limit_lower = 0, 
                                             limit_upper = 33511492.929216, 
                                             offset = 1e+06)  
  )


### PREDICTION

table_forecast_metrics <- forecast_test_tbl %>% 
  select(Date, .value, .model_desc, value_trans)

forecast_test_tbl %>%
  plot_modeltime_forecast(
    .facet_ncol = 4
  )

colnames(forecast_test_tbl)


fin_data_forecast <- fin_data %>% filter(Category %in% c('Forecast') & 
                                           Metric == 'Revenue') %>%
  arrange(Date, ascending = TRUE) %>% group_by(Category, Metric, Date) %>% 
  summarize(.value = sum(Value)) %>%
  ungroup() %>%
  mutate(.model_desc = "Forecast (BE)")


fin_data_forecast$Category <- NULL
fin_data_forecast$Metric <- NULL
fin_data_forecast$value_trans <- c(19927566, 19405639, 20419407)


table_forecast_metrics <- rbind(table_forecast_metrics, fin_data_forecast)


table_forecast_metrics_ggplot <- table_forecast_metrics %>%
  ggplot( aes(x=Date, y=.value, group=.model_desc, color=.model_desc)) +
  geom_line() +
  scale_color_viridis(discrete = TRUE) +
  ggtitle("Popularity of American names in the previous 30 years") +
  theme_ipsum() +
  ylab("Number of babies born")

ggplotly(table_forecast_metrics_ggplot)


table_forecast_metrics %>%
#  filter(.key == "prediction") %>%
  select(.model_desc, .value, value_trans) %>%
  group_by(.model_desc) %>%  
  summarize_accuracy_metrics(
    truth      = value_trans, 
    estimate   = .value,
    metric_set = metric_set(mae, rmse, rsq)
  )


############# UNSEEN DATA #################


model_refit_tbl <- submodels_2_tbl %>%
  modeltime_refit(data_prepared_tbl)

modeltime_forecast_tbl <- model_refit_tbl %>%
  modeltime_forecast(
    new_data    = future_tbl,
    actual_data = data_prepared_tbl,
    keep_data   = TRUE 
  ) %>%
  mutate(
    value_trans= value_trans %>% standardize_inv_vec(mean = 1.29048740547333 , 
                                                     sd = 0.676453813509239)  
  ) %>%
  mutate(
    .value = .value %>% standardize_inv_vec(mean = 1.29048740547333 , 
                                            sd = 0.676453813509239)  
  ) %>%
  
  mutate(
    value_trans= value_trans %>% log_interval_inv_vec(limit_lower = 0, 
                                                      limit_upper = 33511492.929216, 
                                                      offset = 1e+06)  
  ) %>%
  mutate(
    .value = .value %>% log_interval_inv_vec(limit_lower = 0, 
                                             limit_upper = 33511492.929216, 
                                             offset = 1e+06)  
  )


modeltime_forecast_tbl %>% 
  plot_modeltime_forecast(
    .facet_ncol   = 4,
    .y_intercept  = 0
  )


# ENSEMBLE PANEL MODELS -----

#Average Ensemble ----

"select models which we want to include on the ENSAMBLE"

submodels_2_ids_to_keep <- c(1, 4)

ensemble_fit <- submodels_2_tbl %>%
  filter(.model_id %in% submodels_2_ids_to_keep) %>%

ensemble_average(type = "median")


model_ensemble_tbl <- modeltime_table(
  ensemble_fit
)

model_ensemble_tbl %>%
  modeltime_accuracy(testing(splits))


colnames(forecast_ensemble_test_tbl)


forecast_ensemble_test_tbl <- model_ensemble_tbl %>%
  modeltime_forecast(
    new_data    = testing(splits),
    actual_data = data_prepared_tbl,
    keep_data   = TRUE  
  ) %>%
  
  mutate(
    value_trans= value_trans %>% standardize_inv_vec(mean = 1.29048740547333 , 
                                                     sd = 0.676453813509239)  
  ) %>%
  mutate(
    .value = .value %>% standardize_inv_vec(mean = 1.29048740547333 , 
                                            sd = 0.676453813509239)  
  ) %>%
  
  mutate(
    value_trans= value_trans %>% log_interval_inv_vec(limit_lower = 0, 
                                                      limit_upper = 33511492.929216, 
                                                      offset = 1e+06)  
  ) %>%
  mutate(
    .value = .value %>% log_interval_inv_vec(limit_lower = 0, 
                                             limit_upper = 33511492.929216, 
                                             offset = 1e+06)  
  )


### ENSEMBLE PREDICTION

forecast_ensemble_test_tbl %>%
  plot_modeltime_forecast(
    .facet_ncol = 4
  )

"Calculate Accuracy"

forecast_ensemble_test_tbl %>%
  filter(.key == "prediction") %>%
  select(.value, value_trans) %>%
  summarize_accuracy_metrics(
    truth      = value_trans, 
    estimate   = .value,
    metric_set = metric_set(mae, rmse, rsq)
  )


"NOW AGAINST REAL FORECAST"

fin_data_forecast

table_forecast_ensemble_metrics <- forecast_ensemble_test_tbl %>% 
  select(Date, .value, .model_desc, .key, value_trans) 

fin_data_forecast$.key = 'ACTUAL'

table_forecast_ensemble_metrics <- rbind(table_forecast_ensemble_metrics, 
                                         fin_data_forecast)
table(table_forecast_ensemble_metrics$.model_desc)

table_forecast_ensemble_metrics_ggplot <- table_forecast_ensemble_metrics %>%
  ggplot(aes(x=Date, y=.value, group=.model_desc, color=.model_desc)) +
  geom_line() +
  scale_color_viridis(discrete = TRUE) +
  ggtitle("Title") +
  theme_ipsum() +
  ylab("ylab")

ggplotly(table_forecast_ensemble_metrics_ggplot)


#Turn OFF Parallel Backend
plan(sequential)

