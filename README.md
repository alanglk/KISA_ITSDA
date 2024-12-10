# KISA_ITSDA
Repository for "Introduction to Time Series Data Analysis" projects.

## Time Series Forecasting (TSF)

The objective is to train tree different models for a forecasting task on the selected time series dataset:
1. A classical model for time series data such as ARIMA
2. Perform data transformation and train an MLP, linear regressor, random forest model for regression...
3. Train a Deep-Learning model to perform the forecasting task

Finally a comparison between all the models will be performed.

### M5 Forecasting - Accuracy
The selected dataset is the [M5 Forecasting - Accuracy](https://www.kaggle.com/c/m5-forecasting-accuracy). 

**Files:**
- `calendar.csv` - Contains information about the dates on which the products are sold.
- `sales_train_validation.csv` - Contains the historical daily unit sales data per product and store [d_1 - d_1913]
- `sample_submission.csv` - The correct format for submissions. Reference the Evaluation tab for more info.
- `sell_prices.csv` - Contains information about the price of the products sold per store and date.
- `sales_train_evaluation.csv` - Includes sales [d_1 - d_1941] (labels used for the Public leaderboard)


## Implementation
Data structure
```
TSF/
├── classical_tsm.py                # classical TS data models (ARIMA)
├── common.py                       # Interfaces and shared functions
├── dl_tsm.py                       # Deep-Learning models for TS data
├── ml_transform.py                 # transform TS data and train tradition ML models 
├── models                          # models saved checkpoints
└── time_series_forecasting.ipynb
```

[Transformers](https://huggingface.co/blog/time-series-transformers)