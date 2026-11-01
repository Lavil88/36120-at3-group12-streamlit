"""Student B: Market Risk Index (MRI), LightGBM regression.

Owner: Student B. Replace the starter visualisation and add your own content.
Required outputs (brief): Risk Score, Feature Importance.
"""

from __future__ import annotations

import streamlit as st

import pandas as pd
import altair as alt

from common.api_client import get_model_metadata, get_prediction, show_api_error
from common.config import STUDENTS, TOKENS, TOKEN_NAMES, selected_token
from common.data import load_history

STUDENT = "B"
TITLE = STUDENTS[STUDENT]["target"]



def home_visual(token: str) -> None:
    """Display Student B's historical MRI on the Home page."""
    st.markdown(f"**{TITLE}**")

    history = load_history(token)

    if history is None or len(history) < 21:
        st.info("Insufficient market data to display historical MRI.")
        return

    historical_mri = calculate_historical_mri(history).dropna()

    if historical_mri.empty:
        st.info("Historical MRI is unavailable.")
        return

    st.line_chart(
        historical_mri.tail(90),
        height=220,
        y_label="MRI (0-100)",
        x_label="Date",
    )

    st.caption(
        "Historical Market Risk Index calculated from daily market data. "
        "Higher scores indicate greater market risk."
    )




def calculate_historical_mri(history: pd.DataFrame) -> pd.Series:
    """Calculate historical MRI using the assignment formula."""
    close = history["close"]
    high = history["high"]
    open_price = history["open"]

    return20 = close.pct_change(20)
    volatility20 = close.pct_change().rolling(20).std()
    max_price20 = high.rolling(20).max()

    drawdown_risk = (max_price20 - close) / max_price20
    downside_risk = return20.clip(upper=0).abs()

    # Assumption: GapRisk uses a 20-day rolling average.
    daily_gap = (open_price - close.shift(1)).abs() / close.shift(1)
    gap_risk = daily_gap.rolling(window=20).mean()

    vrn = (volatility20 / 0.10).clip(0, 1)
    drn = (drawdown_risk / 0.40).clip(0, 1)
    rrn = (downside_risk / 0.30).clip(0, 1)
    grn = (gap_risk / 0.10).clip(0, 1)

    return (
        100 * (0.40 * vrn + 0.30 * drn + 0.20 * rrn + 0.10 * grn)
    ).rename("Historical MRI")


def calculate_risk_components(history: pd.DataFrame) -> pd.Series:
    """Calculate each component's contribution to the latest historical MRI."""
    close = history["close"]
    high = history["high"]
    open_price = history["open"]

    return20 = close.pct_change(20)
    volatility20 = close.pct_change().rolling(20).std()
    max_price20 = high.rolling(20).max()

    drawdown_risk = (max_price20 - close) / max_price20
    downside_risk = return20.clip(upper=0).abs()

    # Assumed 20-day rolling mean, matching the historical MRI chart.
    daily_gap = (open_price - close.shift(1)).abs() / close.shift(1)
    gap_risk = daily_gap.rolling(20).mean()

    vrn = (volatility20 / 0.10).clip(0, 1)
    drn = (drawdown_risk / 0.40).clip(0, 1)
    rrn = (downside_risk / 0.30).clip(0, 1)
    grn = (gap_risk / 0.10).clip(0, 1)

    contributions = pd.DataFrame({
        "Volatility Risk": 100 * 0.40 * vrn,
        "Drawdown Risk": 100 * 0.30 * drn,
        "Downside Return Risk": 100 * 0.20 * rrn,
        "Gap Risk": 100 * 0.10 * grn,
    })

    valid_contributions = contributions.dropna()

    if valid_contributions.empty:
        return None

    return valid_contributions.iloc[-1]



@st.cache_data(ttl=600, show_spinner=False)
def _all_tokens() -> pd.DataFrame:
    """Retrieve 7-day MRI predictions for all eight cryptocurrencies."""
    rows = []

    for token in TOKENS:
        result = get_prediction(STUDENT, token)

        if result.ok and result.data:
            prediction = result.data.get("prediction")
            prediction_date = result.data.get("prediction_date", "")

            rows.append({
                "Token": f"{token} ({TOKEN_NAMES[token]})",
                "Predicted MRI (0–100)": (
                    round(float(prediction), 1)
                    if isinstance(prediction, (int, float))
                    else None
                ),
                "Prediction date": prediction_date,
            })
        else:
            rows.append({
                "Token": f"{token} ({TOKEN_NAMES[token]})",
                "Predicted MRI (0–100)": None,
                "Prediction date": "",
            })

    return pd.DataFrame(rows)


def render() -> None:
    token = selected_token()
    st.header(f"{TITLE}: {token}")
    st.caption("LightGBM regression, 7 days ahead")
    
    
    # Custom Student B MRI prediction panel
    with st.spinner("Loading MRI prediction..."):
        result = get_prediction(STUDENT, token)

    feature_sources = {}
    if result.ok and result.data:
        data = result.data
        prediction = data.get("prediction")
        prediction_date = data.get("prediction_date", "")
        feature_sources = data.get("feature_sources", {})

        if isinstance(prediction, (int, float)):
            st.markdown("### Predicted Market Risk Index")

            col1, col2 = st.columns(2)

            col1.metric("Risk Score (0-100)", f"{prediction:.1f}")
            col2.metric("Prediction for", str(prediction_date))

            # Progress bar showing the MRI score on its defined scale
            st.progress(
                min(max(float(prediction) / 100, 0.0), 1.0)
            )

            st.caption(
                "Higher MRI scores indicate greater market risk. "
                "The score is an index, not a probability of loss."
            )

            if not 0 <= prediction <= 100:
                st.warning(
                    "The regression prediction is outside the MRI target "
                    "range of 0–100. The progress bar is limited to that "
                    "range for display."
                )

            missing = data.get("features_missing") or []
            if missing:
                st.info(
                    "Some inputs were unavailable and treated as missing: "
                    + ", ".join(missing)
                )
        else:
            st.warning("The API did not return a valid numeric MRI prediction.")
    else:
        show_api_error(result, "Market Risk Index")



    # Show feature sources in a compact, collapsible section.
    if feature_sources:
        with st.expander("Inputs used for this prediction (13 features)", expanded=False):

            source_labels = {
                "student_a": "Student A API",
                "student_b": "Student B API",
                "student_c": "Student C API",
                "student_d": "Student D API",
                "local_fallback": "Local fallback",
                "missing": "Missing",
            }

            inputs_df = pd.DataFrame(
                [
                    {
                        "Feature": feature,
                        "Source": source_labels.get(source, source),
                    }
                    for feature, source in feature_sources.items()
                ]
            )

            st.dataframe(
                inputs_df,
                hide_index=True,
                use_container_width=True,
                height=250,
            )
    
    # Historical MRI calculated 
    st.subheader("Historical Market Risk Index")

    history = load_history(token)

    if history is not None and not history.empty:
        historical_mri = calculate_historical_mri(history).dropna()

        if not historical_mri.empty:
            
            # Prepare historical MRI data for an interactive chart
            mri_chart_data = historical_mri.tail(180).reset_index()
            mri_chart_data.columns = ["Date", "MRI"]

            # Soft blue area underneath the historical MRI line
            area = (
                alt.Chart(mri_chart_data)
                .mark_area(
                    color="#3B82F6",
                    opacity=0.12,
                )
                .encode(
                    x=alt.X(
                        "Date:T",
                        title="Date",
                    ),
                    y=alt.Y(
                    "MRI:Q",
                    title="Market Risk Index (0-100)",
                    scale=alt.Scale(zero=True),
                    ),
                )
            )

            # Main historical MRI line with interactive tooltips
            line = (
                alt.Chart(mri_chart_data)
                .mark_line(
                    color="#2563EB",
                    strokeWidth=2.5,
                )
                .encode(
                    x=alt.X("Date:T"),
                    y=alt.Y("MRI:Q"),
                    tooltip=[
                        alt.Tooltip("Date:T", format="%d %b %Y"),
                        alt.Tooltip("MRI:Q", format=".2f"),
                    ],
                )
            )

            historical_chart = (
                (area + line)
                .properties(height=350)
                .interactive()
            )

            st.altair_chart(
                historical_chart,
                width="stretch",
            )


            st.caption(
                "Historical MRI calculated from daily cryptocurrency prices "
                "using the assignment formula. GapRisk uses an assumed "
                "20-day rolling average. This chart shows historical "
                "calculated risk, not past model predictions."
            )
        else:
            st.info("Insufficient historical data to calculate MRI.")
    else:
        st.warning("Historical market data is unavailable.")


    
    # Breakdown of the latest calculated historical MRI
    st.subheader("Current Market Risk Components")

    if history is not None and len(history) >= 21:
        components = calculate_risk_components(history)
        historical_values = calculate_historical_mri(history).dropna()

        if components is None or historical_values.empty:
            st.info("Insufficient valid data to calculate current risk components.")
            return

        latest_mri = historical_values.iloc[-1]
        # Verify that the risk components reproduce the historical MRI
        component_total = components.sum()

        if not pd.isna(component_total) and abs(component_total - latest_mri) > 1e-6:
            st.warning(
                 "Risk component total does not match the latest historical MRI."
            )

        st.metric("Latest Historical MRI", f"{latest_mri:.1f} / 100")

        
        # Display the four risk components with full labels
        components_df = components.rename("Contribution").reset_index()
        components_df.columns = ["Risk Component", "Contribution"]

        risk_chart = (
            alt.Chart(components_df)
            .mark_bar(color="#2675C7")
            .encode(
                x=alt.X(
                    "Contribution:Q",
                    title="Contribution to MRI (index points)",
                ),
                y=alt.Y(
                    "Risk Component:N",
                    sort="-x",
                    title=None,
                    axis=alt.Axis(labelLimit=250),
                ),
                tooltip=[
                    alt.Tooltip("Risk Component:N"),
                    alt.Tooltip("Contribution:Q", format=".3f"),
                ],
                color=alt.Color(
                    "Risk Component:N",
                    scale=alt.Scale(
                        domain=[
                            "Volatility Risk",
                            "Drawdown Risk",
                            "Downside Return Risk",
                            "Gap Risk",
                        ],
                        range=[
                            "#F59E0B",  # Amber
                            "#EF4444",  # Red
                            "#8B5CF6",  # Purple
                            "#06B6D4",  # Cyan
                        ],
                    ),
                    legend=None,
                ),
            )
            .properties(height=240)
        )

        st.altair_chart(risk_chart, width="stretch")


        st.caption(
            "Each bar shows the contribution of a risk component to "
            "the latest calculated historical MRI. These values add up "
            "to the historical MRI score, not the 7-day model prediction. "
            "GapRisk uses an assumed 20-day rolling average."
        )
    else:
        st.info("Insufficient historical data for risk components.")


    
    # Student B engineered features
    st.subheader("Market Trend Indicators")

    history = load_history(token)

    if history is not None and len(history) >= 20:
        close = history["close"]

        ma5 = close.rolling(5).mean().iloc[-1]
        ma20 = close.rolling(20).mean().iloc[-1]
        trend_strength = (ma5 / ma20) - 1

        col1, col2, col3 = st.columns(3)

        col1.metric("5-day moving average", f"${ma5:,.2f}")
        col2.metric("20-day moving average", f"${ma20:,.2f}")
        col3.metric("Trend Strength", f"{trend_strength:+.2%}")

        st.caption(
            "MA5 and MA20 represent short-term and longer-term price trends. "
            "Trend Strength compares the two moving averages: "
            "positive values indicate MA5 is above MA20, "
            "while negative values indicate MA5 is below MA20."
        )
    else:
        st.info("Insufficient historical data to calculate trend indicators.")
    
    
    # Visualise short-term and longer-term moving averages
    st.markdown("**Price and Moving Average Trends**")

    if history is not None and len(history) >= 20:
        close = history["close"]

        moving_averages = close.to_frame(name="Closing Price")
        moving_averages["MA5"] = close.rolling(5).mean()
        moving_averages["MA20"] = close.rolling(20).mean()

        
        # Prepare price and moving-average data for Altair
        price_chart_data = (
            moving_averages.tail(90)
            .reset_index()
            .rename(columns={moving_averages.index.name or "index": "Date"})
        )

        price_chart_data = price_chart_data.melt(
            id_vars="Date",
            value_vars=["Closing Price", "MA5", "MA20"],
            var_name="Indicator",
            value_name="Price",
        ).dropna()

        # Interactive chart with consistent colours
        price_chart = (
            alt.Chart(price_chart_data)
            .mark_line(strokeWidth=2.5)
            .encode(
                x=alt.X(
                    "Date:T",
                    title="Date",
                    axis=alt.Axis(format="%b %d"),
                ),
                y=alt.Y(
                    "Price:Q",
                    title="Price (USD)",
                    scale=alt.Scale(zero=False),
                ),
                color=alt.Color(
                    "Indicator:N",
                    scale=alt.Scale(
                        domain=["Closing Price", "MA5", "MA20"],
                        range=["#2563EB", "#EF4444", "#38BDF8"],
                    ),
                    legend=alt.Legend(
                        title=None,
                        orient="bottom",
                    ),
                ),
                tooltip=[
                    alt.Tooltip("Date:T", format="%d %b %Y"),
                    alt.Tooltip("Indicator:N"),
                    alt.Tooltip("Price:Q", format=",.2f"),
                ],
            )
            .properties(height=320)
            .interactive()
        )

        st.altair_chart(price_chart, width="stretch")


        st.caption(
            "The chart compares the closing price with its 5-day and "
            "20-day moving averages. MA5 reacts more quickly to recent "
            "price changes, while MA20 shows the longer-term trend."
        )
    else:
        st.info(
            "Insufficient historical data to display moving-average trends."
        )


    st.subheader("Feature importance")
    
        # Retrieve feature importance from Student B's API
    importance_result = get_model_metadata(STUDENT)
    
    if importance_result.ok and importance_result.data:
        importance = importance_result.data.get("feature_importance") or {}
    
        if importance:
                importance_series = pd.Series(importance, name="Importance")
                importance_series = importance_series.sort_values(ascending=False)
    
                st.bar_chart(
                    importance_series,
                    horizontal=True,
                    height=450,
                    sort="-Importance",
                )
    
                st.caption(
                    "Feature importance indicates which inputs the LightGBM "
                    "model uses most when making predictions. Higher values "
                    "indicate greater overall importance."
                )
        else:
                st.info("Feature importance is unavailable.")
    else:
            st.warning("Unable to load model feature importance.")

    
    # Final model evaluation results
    st.subheader("Model Performance")

    col1, col2, col3 = st.columns(3)

    col1.metric("Test RMSE", "9.46")
    col2.metric("Test MAE", "7.56")
    col3.metric("Test R²", "0.540")

    st.caption(
        "Algorithm: Optuna-optimised LightGBM Regressor "
        "(50 optimisation trials, 13 input features). "
        "Performance evaluated on the held-out test dataset."
    )

    with st.expander("What do these evaluation metrics mean?"):
        st.markdown(
            """
            - **RMSE (9.46):** Measures prediction error in MRI index points,
            giving greater weight to larger errors. Lower is better.
            - **MAE (7.56):** The average absolute prediction error is
            approximately 7.56 MRI points. Lower is better.
            - **R² (0.540):** The model explains approximately 54% of
            the variation in MRI scores in the test dataset.
            """
        )

    
# Explain the MRI target and its components
    with st.expander("What is the Market Risk Index (MRI)?"):
        st.write(
            "The Market Risk Index (MRI) measures cryptocurrency market "
            "risk on a scale from 0 to 100. Higher scores indicate "
            "greater market risk."
        )

        st.latex(
            r"MRI = 100(0.40VRN + 0.30DRN + 0.20RRN + 0.10GRN)"
        )

        st.markdown("**MRI components and their weights**")

        st.table({
            "Risk component": [
                "Volatility Risk (VRN)",
                "Drawdown Risk (DRN)",
                "Downside Return Risk (RRN)",
                "Gap Risk (GRN)",
            ],
            "Weight": ["40%", "30%", "20%", "10%"],
            "Description": [
                "Measures 20-day price volatility.",
                "Measures the decline from the 20-day maximum price.",
                "Measures negative returns over 20 days.",
                "Measures gaps between consecutive daily prices.",
            ],
        })

        st.caption(
            "The MRI is calculated from normalised risk components. "
            "The LightGBM regression model predicts the MRI seven days ahead. "
            "The score is not a probability of financial loss."
        )
   

    
    # Compare predicted MRI scores across all eight cryptocurrencies
    if st.toggle("Compare all 8 cryptocurrencies"):
        with st.spinner("Requesting MRI predictions for all tokens..."):
            comparison_df = _all_tokens()

        st.dataframe(
            comparison_df,
            hide_index=True,
            width="stretch",
        )

        
        # Visual comparison of predicted MRI scores
        chart_data = comparison_df.dropna(
            subset=["Predicted MRI (0–100)"]
        ).copy()

        if not chart_data.empty:
            st.markdown("**Predicted MRI Comparison**")

            comparison_chart = (
                alt.Chart(chart_data)
                .mark_bar(cornerRadiusEnd=4)
                .encode(
                    x=alt.X(
                        "Predicted MRI (0–100):Q",
                        title="Predicted Market Risk Index (0-100)",
                        scale=alt.Scale(domain=[0, 100]),
                    ),
                    y=alt.Y(
                        "Token:N",
                        sort="-x",
                        title=None,
                    ),
                    tooltip=[
                        alt.Tooltip("Token:N"),
                        alt.Tooltip(
                            "Predicted MRI (0–100):Q",
                            title="Predicted MRI",
                            format=".1f",
                        ),
                        alt.Tooltip("Prediction date:N"),
                    ],
                )
                .properties(height=300)
            )

            st.altair_chart(comparison_chart, width="stretch")


        st.caption(
            "Predicted Market Risk Index (MRI) seven days ahead. "
            "Higher scores indicate greater market risk. "
            "Unavailable predictions are displayed as blank values."
        )

    
    
  
