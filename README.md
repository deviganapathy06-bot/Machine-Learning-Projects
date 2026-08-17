# 📈 Sales Forecasting using Machine Learning

A machine learning and data analysis project developed during my **ReTech internship**, progressing from sales data analysis to a Machine Learning-based **Sales Forecasting System** with an interactive **Streamlit dashboard**.

The project analyzes historical sales transactions using **Python, Pandas, NumPy, Scikit-learn, Machine Learning, time-series feature engineering, and data visualization** to identify historical sales patterns and forecast future weekly sales.

> ⚠️ **Educational Project:** This system is developed for learning and analytical purposes. Forecasted values are based on historical data and should not be considered guaranteed future sales.

---

## 🚀 Project Overview

The project uses historical sales transaction data containing information about:

- Products
- Sales representatives
- Regions
- Sales amount
- Quantity sold
- Product categories
- Unit cost
- Unit price
- Customer type
- Sale date

The system converts transaction-level sales data into **weekly sales data** and uses historical sales patterns to create features for Machine Learning.

The Machine Learning model then predicts future weekly sales.

---

## 📊 Dataset

The dataset contains **1,000 sales transactions** with the following columns:

| Column | Description |
|---|---|
| `Product_ID` | Unique product identifier |
| `Sale_Date` | Date of the sale |
| `Sales_Rep` | Sales representative responsible for the sale |
| `Region` | Sales region |
| `Sales_Amount` | Total sales amount |
| `Quantity_Sold` | Quantity of products sold |
| `Product_Category` | Category of the product |
| `Unit_Cost` | Cost of one unit |
| `Unit_Price` | Selling price of one unit |
| `Customer_Type` | Type of customer |

---

## 🧠 Sales Forecasting Workflow

```text
Sales Transaction Data
        ↓
Data Loading
        ↓
Data Cleaning
        ↓
Date Processing
        ↓
Weekly Sales Aggregation
        ↓
Time-Series Feature Engineering
        ↓
Lag Features
        ↓
Rolling Averages
        ↓
Random Forest Regression
        ↓
Model Evaluation
        ↓
Future Sales Forecast
        ↓
Streamlit Dashboard
