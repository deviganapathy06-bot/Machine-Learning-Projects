import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


st.set_page_config(
    page_title="Sales Forecasting",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Sales Forecasting")

st.write(
    "A machine learning system that analyzes historical sales, "
    "creates time-series features, and forecasts future sales."
)


@st.cache_data
def load_data():

    df = pd.read_csv("sales_data-selected-columns.csv")

    df["Sale_Date"] = pd.to_datetime(
        df["Sale_Date"]
    )

    df = df.sort_values(
        "Sale_Date"
    ).reset_index(drop=True)

    return df


df = load_data()


st.header("📊 Dataset Overview")

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Transactions",
    f"{len(df):,}"
)

col2.metric(
    "Total Sales",
    f"${df['Sales_Amount'].sum():,.2f}"
)

col3.metric(
    "Average Sale",
    f"${df['Sales_Amount'].mean():,.2f}"
)

col4.metric(
    "Products",
    f"{df['Product_ID'].nunique():,}"
)


with st.expander("View Dataset"):

    st.dataframe(
        df,
        width="stretch"
    )


weekly_sales = (
    df.set_index("Sale_Date")
    ["Sales_Amount"]
    .resample("W")
    .sum()
    .reset_index()
)

weekly_sales.columns = [
    "Date",
    "Sales"
]


st.header("📈 Historical Weekly Sales")

fig, ax = plt.subplots(
    figsize=(12, 5)
)

ax.plot(
    weekly_sales["Date"],
    weekly_sales["Sales"],
    marker="o",
    label="Weekly Sales"
)

ax.set_title(
    "Weekly Sales Trend"
)

ax.set_xlabel(
    "Date"
)

ax.set_ylabel(
    "Sales Amount"
)

ax.grid(True)

ax.legend()

plt.xticks(
    rotation=45
)

plt.tight_layout()

st.pyplot(fig)


data = weekly_sales.copy()

data["Lag_1"] = (
    data["Sales"].shift(1)
)

data["Lag_2"] = (
    data["Sales"].shift(2)
)

data["Lag_4"] = (
    data["Sales"].shift(4)
)

data["Lag_8"] = (
    data["Sales"].shift(8)
)

data["Lag_12"] = (
    data["Sales"].shift(12)
)

data["Rolling_4_Week"] = (
    data["Sales"]
    .shift(1)
    .rolling(4)
    .mean()
)

data["Rolling_8_Week"] = (
    data["Sales"]
    .shift(1)
    .rolling(8)
    .mean()
)

data["Rolling_12_Week"] = (
    data["Sales"]
    .shift(1)
    .rolling(12)
    .mean()
)

data["Month"] = (
    data["Date"].dt.month
)

data["Week"] = (
    data["Date"]
    .dt.isocalendar()
    .week
    .astype(int)
)

data["Year"] = (
    data["Date"].dt.year
)

data = data.dropna().reset_index(
    drop=True
)


features = [

    "Lag_1",
    "Lag_2",
    "Lag_4",
    "Lag_8",
    "Lag_12",

    "Rolling_4_Week",
    "Rolling_8_Week",
    "Rolling_12_Week",

    "Month",
    "Week",
    "Year"
]


X = data[features]

y = data["Sales"]


split_index = int(
    len(data) * 0.80
)

X_train = X.iloc[
    :split_index
]

X_test = X.iloc[
    split_index:
]

y_train = y.iloc[
    :split_index
]

y_test = y.iloc[
    split_index:
]


model = RandomForestRegressor(

    n_estimators=300,

    max_depth=10,

    min_samples_leaf=2,

    random_state=42
)

model.fit(
    X_train,
    y_train
)


predictions = model.predict(
    X_test
)


mae = mean_absolute_error(
    y_test,
    predictions
)

rmse = np.sqrt(
    mean_squared_error(
        y_test,
        predictions
    )
)

r2 = r2_score(
    y_test,
    predictions
)


baseline_prediction = np.repeat(
    y_train.mean(),
    len(y_test)
)

baseline_mae = mean_absolute_error(
    y_test,
    baseline_prediction
)


st.header("🤖 Model Performance")

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "MAE",
    f"${mae:,.2f}"
)

col2.metric(
    "RMSE",
    f"${rmse:,.2f}"
)

col3.metric(
    "R² Score",
    f"{r2:.3f}"
)

col4.metric(
    "Baseline MAE",
    f"${baseline_mae:,.2f}"
)


if mae < baseline_mae:

    st.success(
        "The Random Forest model performs better "
        "than the average-sales baseline."
    )

else:

    st.warning(
        "The model does not outperform the baseline. "
        "The available historical data is limited, so "
        "the forecast should be treated as an educational model."
    )


st.header(
    "📊 Actual vs Predicted Sales"
)


results = pd.DataFrame({

    "Date":
        data.iloc[
            split_index:
        ]["Date"].values,

    "Actual Sales":
        y_test.values,

    "Predicted Sales":
        predictions

})


st.dataframe(
    results,
    width="stretch"
)


fig2, ax2 = plt.subplots(
    figsize=(12, 5)
)

ax2.plot(
    results["Date"],
    results["Actual Sales"],
    marker="o",
    label="Actual Sales"
)

ax2.plot(
    results["Date"],
    results["Predicted Sales"],
    marker="o",
    linestyle="--",
    label="Predicted Sales"
)

ax2.set_title(
    "Actual vs Predicted Weekly Sales"
)

ax2.set_xlabel(
    "Date"
)

ax2.set_ylabel(
    "Sales Amount"
)

ax2.grid(True)

ax2.legend()

plt.xticks(
    rotation=45
)

plt.tight_layout()

st.pyplot(fig2)


st.header(
    "🔮 Future Sales Forecast"
)


forecast_weeks = st.slider(

    "Number of weeks to forecast",

    min_value=1,

    max_value=12,

    value=6
)


forecast_data = weekly_sales.copy()

future_predictions = []


for i in range(
    forecast_weeks
):

    next_date = (

        forecast_data["Date"].iloc[-1]

        + pd.Timedelta(
            weeks=1
        )

    )

    if len(forecast_data) < 12:

        st.error(
            "Not enough historical data to create "
            "12-week lag features."
        )

        st.stop()


    lag_1 = (
        forecast_data["Sales"].iloc[-1]
    )

    lag_2 = (
        forecast_data["Sales"].iloc[-2]
    )

    lag_4 = (
        forecast_data["Sales"].iloc[-4]
    )

    lag_8 = (
        forecast_data["Sales"].iloc[-8]
    )

    lag_12 = (
        forecast_data["Sales"].iloc[-12]
    )

    rolling_4 = (
        forecast_data["Sales"]
        .tail(4)
        .mean()
    )

    rolling_8 = (
        forecast_data["Sales"]
        .tail(8)
        .mean()
    )

    rolling_12 = (
        forecast_data["Sales"]
        .tail(12)
        .mean()
    )


    future_features = pd.DataFrame({

        "Lag_1": [lag_1],

        "Lag_2": [lag_2],

        "Lag_4": [lag_4],

        "Lag_8": [lag_8],

        "Lag_12": [lag_12],

        "Rolling_4_Week": [rolling_4],

        "Rolling_8_Week": [rolling_8],

        "Rolling_12_Week": [rolling_12],

        "Month": [
            next_date.month
        ],

        "Week": [
            int(next_date.isocalendar().week)
        ],

        "Year": [
            next_date.year
        ]

    })


    prediction = model.predict(
        future_features
    )[0]


    future_predictions.append({

        "Date":
            next_date,

        "Forecasted Sales":
            prediction

    })


    new_row = pd.DataFrame({

        "Date": [
            next_date
        ],

        "Sales": [
            prediction
        ]

    })


    forecast_data = pd.concat(

        [
            forecast_data,
            new_row
        ],

        ignore_index=True

    )


forecast_df = pd.DataFrame(
    future_predictions
)


st.subheader(
    "Future Weekly Sales"
)


st.dataframe(
    forecast_df,
    width="stretch"
)


fig3, ax3 = plt.subplots(
    figsize=(12, 5)
)

ax3.plot(

    weekly_sales["Date"],

    weekly_sales["Sales"],

    label="Historical Sales"

)

ax3.plot(

    forecast_df["Date"],

    forecast_df["Forecasted Sales"],

    marker="o",

    linestyle="--",

    label="Forecast"

)

ax3.set_title(
    "Historical Sales and Future Forecast"
)

ax3.set_xlabel(
    "Date"
)

ax3.set_ylabel(
    "Sales Amount"
)

ax3.grid(True)

ax3.legend()

plt.xticks(
    rotation=45
)

plt.tight_layout()

st.pyplot(fig3)


st.header(
    "🔍 Feature Importance"
)


importance = pd.DataFrame({

    "Feature":
        features,

    "Importance":
        model.feature_importances_

})


importance = importance.sort_values(

    "Importance",

    ascending=False

)


st.dataframe(
    importance,
    width="stretch"
)


fig4, ax4 = plt.subplots(
    figsize=(10, 5)
)

ax4.barh(

    importance["Feature"],

    importance["Importance"]

)

ax4.set_title(
    "Random Forest Feature Importance"
)

ax4.set_xlabel(
    "Importance"
)

plt.tight_layout()

st.pyplot(fig4)


st.header(
    "📋 Project Summary"
)


st.write(

    f"""
    **Dataset:** Sales transaction dataset

    **Forecasting Frequency:** Weekly

    **Training Records:** {len(X_train)}

    **Testing Records:** {len(X_test)}

    **Machine Learning Model:** Random Forest Regression

    **Features:** Historical sales lag values, rolling averages,
    month, week and year.

    **MAE:** ${mae:,.2f}

    **RMSE:** ${rmse:,.2f}

    **R² Score:** {r2:.3f}

    **Forecast Horizon:** {forecast_weeks} week(s)
    """

)


st.success(
    "Sales forecasting analysis completed successfully!"
)