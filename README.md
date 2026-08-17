# Sales Forecasting using Machine Learning

A machine learning project that analyzes historical sales data, performs time-series feature engineering, and forecasts future weekly sales using Random Forest Regression.

## Project Overview

This project uses historical sales transaction data to identify sales patterns and build a machine learning model for forecasting future sales.

The project includes data preprocessing, weekly sales aggregation, lag feature creation, rolling averages, model training, evaluation, and an interactive Streamlit dashboard.

## Dataset

The dataset contains 1,000 sales transactions with the following columns:

- Product_ID
- Sale_Date
- Sales_Rep
- Region
- Sales_Amount
- Quantity_Sold
- Product_Category
- Unit_Cost
- Unit_Price
- Customer_Type

## Machine Learning Approach

The transaction-level sales data is first aggregated into weekly sales.

Time-series features are then created from historical sales:

- Previous 1-week sales
- Previous 2-week sales
- Previous 4-week sales
- Previous 8-week sales
- Previous 12-week sales
- 4-week rolling average
- 8-week rolling average
- 12-week rolling average
- Month
- Week
- Year

These features are used to train a Random Forest Regression model.

## Model

The project uses:

**Random Forest Regression**

The dataset is divided chronologically:

- 80% historical data → Training
- 20% later data → Testing

A time-based split is used instead of a random split because the project focuses on forecasting.

## Model Evaluation

The model is evaluated using:

- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- R² Score

The project also compares the machine learning model with a simple average-sales baseline.

## Streamlit Dashboard

The interactive dashboard provides:

- Dataset overview
- Total sales
- Average sales
- Historical weekly sales
- Actual vs predicted sales
- Model performance metrics
- Future weekly sales forecast
- Feature importance
- Forecast horizon selection

## Project Structure

```text
Sales-Forecasting/
│
├── app.py
├── sales_data-selected-columns.csv
├── requirements.txt
└── README.md
