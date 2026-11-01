"""Configuration shared by all pages.

API base URLs are read, in this order, from:
1. Streamlit secrets (.streamlit/secrets.toml locally, or the app's Secrets settings on Streamlit Cloud)
2. Environment variables (e.g. in Docker)
3. The defaults below (empty until each API is deployed)
"""

from __future__ import annotations

import os

import streamlit as st

TOKENS: list[str] = ["BTC", "ETH", "XRP", "SOL", "DOGE", "XMR", "XLM", "TRX"]
TOKEN_NAMES: dict[str, str] = {
    "BTC": "Bitcoin", "ETH": "Ethereum", "XRP": "Ripple", "SOL": "Solana",
    "DOGE": "Dogecoin", "XMR": "Monero", "XLM": "Stellar", "TRX": "Tron",
}
FORECAST_HORIZON = 7

STUDENTS: dict[str, dict[str, str]] = {
    "A": {"target": "Asset Performance Index (API)", "env": "STUDENT_A_API_URL"},
    "B": {"target": "Market Risk Index (MRI)", "env": "STUDENT_B_API_URL"},
    "C": {"target": "Trend Category (TC)", "env": "STUDENT_C_API_URL"},
    "D": {"target": "Trading Opportunity Rating (TOR)", "env": "STUDENT_D_API_URL"},
}

# Fill in each Render URL once deployed (no trailing slash), or set them as secrets.

DEFAULT_API_URLS: dict[str, str] = {
    "A": "",
    "B": "https://three6120-26sp-at3-group12-studentb-api.onrender.com",
    "C": "https://studentc-trend-category-api.onrender.com",
    "D": "https://tor-api-student.onrender.com",
}

# Render's free tier sleeps idle services; the first request can take 30 to 60 seconds.
REQUEST_TIMEOUT_SECONDS = 60
HEALTH_TIMEOUT_SECONDS = 10

DISCLAIMER = (
    "This application is a university project. Ratings and scores are general information produced "
    "by machine learning models, not financial advice. Cryptocurrencies are highly volatile; consider "
    "your own circumstances and seek independent advice before making investment decisions."
)


def api_url(student: str) -> str:
    """Base URL of a student's FastAPI service, or an empty string if not configured."""
    env_key = STUDENTS[student]["env"]
    try:
        value = st.secrets.get(env_key)
    except Exception:  # no secrets file available
        value = None
    return (value or os.getenv(env_key) or DEFAULT_API_URLS[student]).rstrip("/")


def selected_token() -> str:
    """Token chosen in the sidebar (shared by every page)."""
    return st.session_state.get("token", TOKENS[0])
