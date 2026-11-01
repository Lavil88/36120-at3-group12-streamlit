"""Student A: Asset Performance Index (API), XGBoost regression.

Owner: Student A (Chakshu Karwal, 26038226).
Required outputs: Performance Score, Feature Importance.
Every prediction, explanation and model detail comes from the Student A FastAPI service; no model runs
here. The index history chart only applies the brief's formula to Kraken prices, for context.
"""

from __future__ import annotations

import altair as alt
import pandas as pd
import streamlit as st

from common.api_client import get_features, get_model_metadata, get_prediction, show_api_error
from common.config import DISCLAIMER, STUDENTS, TOKEN_NAMES, TOKENS, selected_token
from common.data import load_history

STUDENT = "A"
TITLE = STUDENTS[STUDENT]["target"]
N_FEATURES = 8

# Reading guide for the 0-100 score (for users; the brief defines the score, not these labels)
BANDS = pd.DataFrame({
    "label": ["Weak", "Moderate", "Strong", "Very strong"],
    "low": [0, 40, 60, 80],
    "high": [40, 60, 80, 100],
    "colour": ["#C62828", "#F9A825", "#43A047", "#1B5E20"],
    "range": ["0 to 40", "40 to 60", "60 to 80", "80 to 100"],
})
MEANING = {
    "Weak": "Returns, trend and trading activity are expected to be weak next week.",
    "Moderate": "Performance is expected to be around average next week.",
    "Strong": "Returns and trend are expected to be clearly positive next week.",
    "Very strong": ("Very strong performance is expected. In testing, very high scores usually came down "
                    "within a week, so treat this with some caution."),
}
FEATURE_TEXT = {
    "Return20": "20-day return",
    "Return5": "5-day return",
    "TrendStrength": "Short vs long moving average (MA5 / MA20 - 1)",
    "RelativeVolume": "Today's volume vs 20-day average",
    "Volatility20": "20-day volatility",
    "API_t": "Today's Asset Performance Index",
    "API_proj": "Projected index (if prices stay flat for 7 days)",
    "is_weekend": "Weekend effect (lower weekend volume)",
}
# Short names for the inputs table, which sits in a half-width column
SHORT_NAME = {
    "Return20": "20-day return", "Return5": "5-day return", "TrendStrength": "Trend strength",
    "RelativeVolume": "Relative volume", "Volatility20": "20-day volatility",
    "API_t": "Today's index", "API_proj": "Projected index", "is_weekend": "Weekend",
}
FEATURE_SOURCE = {
    "Return20": "This API", "Return5": "This API", "API_proj": "This API",
    "TrendStrength": "Student B", "Volatility20": "Student C", "RelativeVolume": "Student D",
    "API_t": "Calculated", "is_weekend": "Calendar",
}
SERVICE_NAMES = {"student_b": "Student B (TrendStrength)", "student_c": "Student C (Volatility20)",
                 "student_d": "Student D (RelativeVolume)"}
# Average error of the final model per coin on the held-out test period (3 Jul to 14 Oct 2026, notebook J.4)
COIN_ERROR = pd.DataFrame({
    "Coin": ["TRX", "BTC", "ETH", "XMR", "XRP", "SOL", "DOGE", "XLM"],   # calmest to most volatile
    "Model MAE": [4.34, 7.81, 8.46, 11.26, 11.10, 11.09, 12.21, 14.64],
    "Projection baseline MAE": [4.26, 7.95, 8.04, 10.51, 11.92, 11.18, 13.08, 15.35],
})


# Helpers (plain calculations for display; no model)
def band_of(score: float) -> pd.Series:
    """Row of BANDS that contains the score."""
    for _, row in BANDS.iterrows():
        if score < row["high"] or row["label"] == "Very strong":
            return row
    return BANDS.iloc[-1]


def index_history(history: pd.DataFrame) -> pd.Series:
    """The brief's Asset Performance Index for every past day, calculated from Kraken prices."""
    close, volume = history["close"], history["volume"]
    return20 = close.pct_change(20, fill_method=None)
    trend_strength = close.rolling(5).mean() / close.rolling(20).mean() - 1
    volatility20 = close.pct_change(fill_method=None).rolling(20).std()
    relative_volume = volume / volume.rolling(20).mean()
    rs = ((return20 + 0.20) / 0.40).clip(0, 1)
    ts = ((trend_strength + 0.10) / 0.20).clip(0, 1)
    rv = relative_volume.clip(upper=2) / 2
    ps = 1 - volatility20
    return (100 * (0.40 * rs + 0.30 * ts + 0.20 * rv + 0.10 * ps)).rename("index")


def waterfall_table(explanation: dict, min_points: float = 0.05) -> pd.DataFrame:
    """Steps from the model's average forecast to this forecast, largest contributions first.

    Contributions smaller than min_points are combined into "Other inputs" to keep the chart readable.
    """
    base = float(explanation["base_value"])
    contrib = pd.Series(explanation["contributions"], dtype=float)
    big = contrib[contrib.abs() >= min_points].sort_values(key=abs, ascending=False)
    small = contrib[contrib.abs() < min_points].sum()
    steps = [(FEATURE_TEXT.get(name, name), value) for name, value in big.items()]
    if abs(small) > 0:
        steps.append(("Other inputs", small))

    rows, running = [{"step": "Average forecast (starting point)", "start": None, "end": base,
                      "change": base, "kind": "total"}], base
    for label, value in steps:
        rows.append({"step": label, "start": running, "end": running + value, "change": value,
                     "kind": "raises the score" if value >= 0 else "lowers the score"})
        running += value
    rows.append({"step": "This forecast", "start": None, "end": running, "change": running, "kind": "total"})
    return pd.DataFrame(rows)


def fmt_pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value:+.2%}"


# Home page visual
def home_visual(token: str) -> None:
    """Asset Performance Index over the last 180 days, on top of the reading-guide bands."""
    st.markdown(f"**{TITLE}**")
    history = load_history(token)
    if history is None:
        st.warning("Market data unavailable.")
        return
    df = index_history(history).dropna().tail(180).reset_index()
    st.altair_chart(_history_chart(df, height=220), width="stretch")
    st.caption("Calculated daily from Kraken prices with the brief's formula (0-100).")


def _history_chart(df: pd.DataFrame, height: int, forecast: dict | None = None) -> alt.LayerChart:
    """Index line over coloured bands; optionally the 7-day forecast with its likely range.

    History and forecast go into one table (column "kind"), so both are drawn on the same date axis.
    """
    data = df.rename(columns={"index": "score"}).assign(kind="History")
    if forecast is not None:
        latest = data.iloc[[-1]].assign(kind="Forecast")      # dashed line starts at the latest day
        ahead = pd.DataFrame({"date": [pd.Timestamp(forecast["date"])], "score": [forecast["index"]],
                              "low": [forecast["low"]], "high": [forecast["high"]], "kind": ["Forecast"]})
        data = pd.concat([data, latest, ahead], ignore_index=True)

    y_scale = alt.Scale(domain=[0, 100])
    shade = alt.Chart(BANDS).mark_rect(opacity=0.12).encode(
        y=alt.Y("low:Q", title="Index (0-100)", scale=y_scale), y2="high:Q",
        color=alt.Color("label:N", title=None, legend=alt.Legend(orient="bottom"),
                        scale=alt.Scale(domain=list(BANDS["label"]), range=list(BANDS["colour"]))),
        tooltip=[alt.Tooltip("label:N", title="Reading"), alt.Tooltip("range:N", title="Score")],
    )
    base = alt.Chart(data).encode(x=alt.X("date:T", title=None),
                                  y=alt.Y("score:Q", title="Index (0-100)", scale=y_scale))
    line = base.transform_filter(alt.datum.kind == "History").mark_line(color="#2F80ED", strokeWidth=2).encode(
        tooltip=[alt.Tooltip("date:T", title="Date"), alt.Tooltip("score:Q", title="Index", format=".1f")])
    layers = [shade, line]

    if forecast is not None:
        ahead_only = base.transform_filter("isValid(datum.low)")       # only the forecast row has a range
        layers += [
            base.transform_filter(alt.datum.kind == "Forecast").mark_line(
                color="black", strokeDash=[4, 4], strokeWidth=1.5),
            ahead_only.mark_rule(color="black", strokeWidth=2).encode(y="low:Q", y2="high:Q"),
            ahead_only.mark_point(color="black", fill="black", size=110, shape="diamond", opacity=1).encode(
                tooltip=[alt.Tooltip("date:T", title="Forecast for"),
                         alt.Tooltip("score:Q", title="Forecast", format=".1f"),
                         alt.Tooltip("low:Q", title="Likely from", format=".1f"),
                         alt.Tooltip("high:Q", title="Likely to", format=".1f")]),
        ]
    return alt.layer(*layers).properties(height=height)


# Page sections
def _score_card(data: dict) -> None:
    score = float(data["prediction"])
    band = band_of(score)
    low, high = data.get("likely_range") or [None, None]
    range_text = f"Likely between <b>{low:.0f}</b> and <b>{high:.0f}</b>" if low is not None else ""
    st.markdown(
        f"""<div style="border-left: 8px solid {band['colour']}; padding: 0.8rem 1.2rem; border-radius: 6px;
        background: rgba(127,127,127,0.08); margin-bottom: 0.8rem;">
        <div style="font-size: 0.9rem; opacity: 0.8;">Performance Score for {data['prediction_date']}
        (based on data up to {data['as_of']})</div>
        <div style="font-size: 2.4rem; font-weight: 700; color: {band['colour']};">{score:.1f}
        <span style="font-size: 1.1rem; opacity: 0.8;">/ 100 · {band['label']}</span></div>
        <div style="font-size: 0.95rem;">{MEANING[band['label']]} {range_text}</div></div>""",
        unsafe_allow_html=True,
    )

    today = (data.get("features_used") or {}).get("API_t")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Performance Score (7 days ahead)", f"{score:.1f}")
    c2.metric("Today's index", "n/a" if today is None else f"{today:.1f}",
              None if today is None else f"{score - today:+.1f} expected change", delta_color="normal")
    c3.metric("Typical error", f"± {data['typical_error']:.1f}" if "typical_error" in data else "n/a")
    c4.metric("Inputs available", f"{N_FEATURES - len(data.get('features_missing', []))} of {N_FEATURES}")
    _scale_chart(score, low, high, today)


def _scale_chart(score: float, low: float | None, high: float | None, today: float | None) -> None:
    """0-100 scale with the bands, the likely range (grey), the forecast (black) and today's index (dotted).

    Every layer shares one hidden 0-1 vertical scale, so the lines get the full chart height.
    """
    x = alt.X("x0:Q", title=None, scale=alt.Scale(domain=[0, 100]), axis=alt.Axis(values=[0, 20, 40, 60, 80, 100]))
    y = alt.Y("y0:Q", scale=alt.Scale(domain=[0, 1]), axis=None)
    bands = BANDS.rename(columns={"low": "x0", "high": "x1"}).assign(y0=0.0, y1=1.0)
    layers = [alt.Chart(bands).mark_rect(opacity=0.35).encode(
        x=x, x2="x1:Q", y=y, y2="y1:Q",
        color=alt.Color("label:N", legend=None,
                        scale=alt.Scale(domain=list(BANDS["label"]), range=list(BANDS["colour"]))),
        tooltip=[alt.Tooltip("label:N", title="Reading"), alt.Tooltip("range:N", title="Score")],
    )]
    if low is not None:
        likely = pd.DataFrame({"x0": [max(low, 0)], "x1": [min(high, 100)], "y0": [0.3], "y1": [0.7]})
        layers.append(alt.Chart(likely).mark_rect(color="#555", opacity=0.45).encode(
            x=x, x2="x1:Q", y=y, y2="y1:Q"))
    # Solid line for the forecast, dotted line for today's index
    for value, dash, width, name in [(score, [1, 0], 3, "Forecast"), (today, [3, 3], 2, "Today's index")]:
        if value is not None:
            mark = pd.DataFrame({"x0": [value], "y0": [0.0], "y1": [1.0], "what": [name]})
            layers.append(alt.Chart(mark).mark_rule(color="black", strokeDash=dash, strokeWidth=width).encode(
                x=x, y=y, y2="y1:Q",
                tooltip=[alt.Tooltip("what:N", title=" "), alt.Tooltip("x0:Q", title="Score", format=".1f")]))
    st.altair_chart(alt.layer(*layers).properties(height=60), width="stretch")
    st.caption("Solid black line: forecast · grey bar: likely range (± typical error) · dotted line: today's index.")


def _explanation(data: dict) -> None:
    explanation = data.get("explanation")
    if not explanation:
        st.info("No explanation returned by the API.")
        return
    df = waterfall_table(explanation)
    order = list(df["step"])
    lo = min(df["end"].min(), df["start"].min(skipna=True)) - 3
    hi = max(df["end"].max(), df["start"].max(skipna=True)) + 6     # room for the number labels
    df["start"] = df["start"].fillna(lo)          # totals are drawn from the left edge
    df["label"] = [f"{c:.1f}" if k == "total" else f"{c:+.1f}" for c, k in zip(df["change"], df["kind"])]
    df["label_at"] = df[["start", "end"]].max(axis=1)

    base = alt.Chart(df).encode(y=alt.Y("step:N", sort=order, title=None, axis=alt.Axis(labelLimit=320)))
    bars = base.mark_bar().encode(
        x=alt.X("start:Q", title="Score (index points)", scale=alt.Scale(domain=[lo, hi], nice=False)),
        x2="end:Q",
        color=alt.Color("kind:N", legend=None,
                        scale=alt.Scale(domain=["raises the score", "lowers the score", "total"],
                                        range=["#43A047", "#C62828", "#4C72B0"])),
        tooltip=["step:N", alt.Tooltip("change:Q", title="points", format="+.2f")],
    )
    labels = base.mark_text(align="left", dx=4, fontSize=12).encode(x="label_at:Q", text="label:N")
    st.altair_chart(alt.layer(bars, labels).properties(height=34 * len(df) + 40), width="stretch")

    drivers = df[df["kind"] != "total"].head(2)
    parts = [f"**{row['step'].lower()}** {'raised' if row['change'] >= 0 else 'lowered'} it by "
             f"{abs(row['change']):.1f} points" for _, row in drivers.iterrows()]
    st.caption("Green raised the score, red lowered it. Starting from the model's average forecast of "
               f"{explanation['base_value']:.1f}, " + " and ".join(parts)
               + ". Contributions are XGBoost SHAP values for this single forecast.")


def fmt_input(name: str, value: float | None) -> str:
    """Readable value for the inputs table (percentages for returns, 'x' for volume ratio)."""
    if value is None:
        return "missing"
    if name in ("Return20", "Return5", "TrendStrength"):
        return f"{value:+.2%}"
    if name == "Volatility20":
        return f"{value:.2%}"
    if name == "RelativeVolume":
        return f"{value:.2f}x"
    if name == "is_weekend":
        return "Yes" if value else "No"
    return f"{value:.1f}"


def _inputs(data: dict) -> None:
    used = data.get("features_used", {})
    rows = [{"Input": SHORT_NAME.get(f, f), "Value": fmt_input(f, v), "From": FEATURE_SOURCE.get(f, "")}
            for f, v in used.items()]
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    missing = data.get("features_missing", [])
    if missing:
        st.info("Some inputs were unavailable and passed to the model as missing values: "
                + ", ".join(SHORT_NAME.get(m, m) for m in missing) + ". The model was trained with simulated "
                "outages, and its test error stayed between 10.1 and 10.3 points in every outage scenario.")
    status = data.get("sibling_status", {})
    if status:
        st.caption("Teammates' services: " + " · ".join(
            f"{SERVICE_NAMES.get(k, k)}: {'✅ ok' if v == 'ok' else '⚠️ ' + v}" for k, v in status.items()))


def _history_with_forecast(token: str, data: dict) -> None:
    history = load_history(token)
    if history is None:
        st.warning("Market data unavailable, so the history chart cannot be shown.")
        return
    df = index_history(history).dropna().tail(180).reset_index()
    low, high = data.get("likely_range") or [data["prediction"], data["prediction"]]
    forecast = {"date": pd.Timestamp(data["prediction_date"]), "index": float(data["prediction"]),
                "low": low, "high": high}
    st.altair_chart(_history_chart(df, height=300, forecast=forecast), width="stretch")
    st.caption("Blue: the index over the last 180 days, calculated from Kraken prices with the brief's formula. "
               "Black diamond: the model's forecast for 7 days after the latest completed day; the vertical "
               "line through it is the likely range and the dashed line links it to today.")


def _own_features(token: str) -> None:
    res = get_features(STUDENT, token)
    if not res.ok:
        show_api_error(res, "Student A features")
        return
    f = res.data.get("features", {})
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Daily return", fmt_pct(f.get("DailyReturn")))
    c2.metric("5-day return", fmt_pct(f.get("Return5")))
    c3.metric("20-day return", fmt_pct(f.get("Return20")))
    momentum = f.get("Momentum")
    c4.metric("20-day momentum (USD)", "n/a" if momentum is None else f"{momentum:+,.2f}")
    st.caption(f"As of {res.data.get('as_of')}. These features are shared with Students B, C and D through the "
               "/features endpoint and used in their models.")


def _model_details(token: str) -> None:
    res = get_model_metadata(STUDENT)
    if not res.ok:
        show_api_error(res, "Model metadata")
        return
    meta = res.data
    importance = meta.get("feature_importance", {})
    if importance:
        df = pd.DataFrame({"feature": list(importance), "importance": list(importance.values())})
        df["description"] = df["feature"].map(FEATURE_TEXT).fillna(df["feature"])
        chart = alt.Chart(df).mark_bar(color="#4C72B0").encode(
            x=alt.X("importance:Q", title="Share of total gain", axis=alt.Axis(format="%")),
            y=alt.Y("description:N", sort="-x", title=None, axis=alt.Axis(labelLimit=320)),
            tooltip=["feature:N", alt.Tooltip("importance:Q", format=".1%")],
        ).properties(height=34 * len(df) + 20)
        st.altair_chart(chart, width="stretch")
        st.caption("How much each input improved the model during training. The projected index carries most of "
                   "the signal; today's index is never used because the projected index already contains it.")

    metrics = meta.get("test_metrics") or {}
    final = next((m for name, m in metrics.items() if name.startswith("Final")), None)
    if final:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Test MAE", f"{final['MAE']:.2f} points")
        c2.metric("Within 10 points", f"{final['within_10_points']:.1%}")
        c3.metric("More than 10 points too high", f"{final['over_by_more_than_10']:.1%}")
        c4.metric("R²", f"{final['R2']:.3f}")
        table = pd.DataFrame([{"Model": name, "MAE": m["MAE"], "Within 10 points": m["within_10_points"],
                               "> 10 points too high": m["over_by_more_than_10"]} for name, m in metrics.items()])
        st.dataframe(table.style.format({"MAE": "{:.2f}", "Within 10 points": "{:.1%}",
                                         "> 10 points too high": "{:.1%}"}), hide_index=True, width="stretch")
    st.caption(f"Algorithm: {meta.get('algorithm')} · Training period: "
               f"{' to '.join(meta.get('training_period') or [])} · Test period: 3 Jul to 14 Oct 2026")

    coin_row = COIN_ERROR[COIN_ERROR["Coin"] == token]
    with st.expander("How reliable is the score for each coin?"):
        if not coin_row.empty:
            st.write(f"On the test period, forecasts for **{token}** were off by "
                     f"**{coin_row['Model MAE'].iloc[0]:.1f} points** on average. Calm coins are much easier to "
                     "forecast than volatile ones:")
        st.dataframe(COIN_ERROR.style.format({"Model MAE": "{:.1f}", "Projection baseline MAE": "{:.1f}"}),
                     hide_index=True, width="stretch")
    with st.expander("Hyperparameters (tuned with Optuna)"):
        st.json(meta.get("hyperparameters") or {})


@st.cache_data(ttl=600, show_spinner=False)
def _all_tokens() -> pd.DataFrame:
    rows = []
    for t in TOKENS:
        res = get_prediction(STUDENT, t)
        if res.ok:
            score = float(res.data["prediction"])
            low, high = res.data.get("likely_range") or [None, None]
            rows.append({"Token": f"{t} ({TOKEN_NAMES[t]})", "Score": round(score, 1),
                         "Reading": band_of(score)["label"],
                         "Likely range": "" if low is None else f"{low:.0f} to {high:.0f}",
                         "For date": res.data.get("prediction_date", "")})
        else:
            rows.append({"Token": f"{t} ({TOKEN_NAMES[t]})", "Score": None, "Reading": "unavailable",
                         "Likely range": "", "For date": ""})
    return pd.DataFrame(rows)


def _compare_all() -> None:
    df = _all_tokens()
    scored = df.dropna(subset=["Score"])
    if not scored.empty:
        chart = alt.Chart(scored).mark_bar().encode(
            x=alt.X("Score:Q", scale=alt.Scale(domain=[0, 100]), title="Performance Score (7 days ahead)"),
            y=alt.Y("Token:N", sort="-x", title=None),
            color=alt.Color("Reading:N", legend=alt.Legend(orient="top"), title=None,
                            scale=alt.Scale(domain=list(BANDS["label"]), range=list(BANDS["colour"]))),
            tooltip=["Token:N", alt.Tooltip("Score:Q", format=".1f"), "Likely range:N"],
        ).properties(height=34 * len(scored) + 40)
        st.altair_chart(chart, width="stretch")
    st.dataframe(df, hide_index=True, width="stretch")



# Page
def render() -> None:
    token = selected_token()
    st.header(f"{TITLE}: {TOKEN_NAMES[token]} ({token})")
    st.caption("XGBoost regression · Performance Score 7 days ahead · Student A")

    with st.spinner("Requesting the Performance Score from the Student A API (the first request can take up "
                    "to a minute)..."):
        result = get_prediction(STUDENT, token)

    if not result.ok:
        show_api_error(result, "The Performance Score")
        if st.button("Try again"):
            get_prediction.clear()
            st.rerun()
    else:
        data = result.data
        _score_card(data)
        left, right = st.columns(2)
        with left:
            st.subheader("Why this score?")
            _explanation(data)
        with right:
            st.subheader("Inputs used for this forecast")
            _inputs(data)
        st.subheader("Index history and this week's forecast")
        _history_with_forecast(token, data)

    st.subheader("Return features (this service's features)")
    _own_features(token)

    st.subheader("Model information and feature importance")
    _model_details(token)

    with st.expander("What is the Asset Performance Index?"):
        st.markdown(
            "A 0-100 score that combines four signals, each first turned into a 0-1 score:\n\n"
            "| Part | Based on | Weight |\n|---|---|---|\n"
            "| Return Strength | 20-day return, capped at ±20% | 40% |\n"
            "| Trend Strength | 5-day vs 20-day moving average, capped at ±10% | 30% |\n"
            "| Relative Volume | today's volume vs its 20-day average, capped at 2x | 20% |\n"
            "| Price Stability | 1 - 20-day volatility | 10% |\n\n"
            "`API = 100 x (0.40 x RS + 0.30 x TS + 0.20 x RV + 0.10 x PS)`\n\n"
            "The model forecasts this score 7 days ahead. The labels Weak (below 40), Moderate (40-60), "
            "Strong (60-80) and Very strong (80+) are a reading guide only. The score summarises recent "
            "performance; it is not a buy or sell signal."
        )

    if st.toggle("Compare all 8 cryptocurrencies"):
        with st.spinner("Requesting Performance Scores for all tokens..."):
            _compare_all()

    st.caption(DISCLAIMER)