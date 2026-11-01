# Cryptocurrency Financial Intelligence Platform: Streamlit front-end

36120 Advanced Machine Learning Applications, Assessment 3, **Group 12**

| Student | Name | Student ID | Page | Model |
|---|---|---|---|---|
| A | Chakshu Karwal | 26038226 | Asset Performance Index (API) | XGBoost regression |
| B | Prachi Dhimate | 26060133 | Market Risk Index (MRI) | LightGBM regression |
| C | Lavil Saldanha |26036727 | Trend Category (TC) | XGBoost classification |
| D | Vedang Gosavi | 26059271 | Trading Opportunity Rating (TOR) | LightGBM classification |

Deployed app: _Streamlit Community Cloud URL to add_

## What it does

Users pick one of 8 cryptocurrencies (BTC, ETH, XRP, SOL, DOGE, XMR, XLM, TRX). The home page shows the
price history and four market visualisations (one per student). Four tabs show each student's 7-day
prediction, confidence (classification models) and feature importance.

All predictions come from four independent FastAPI services deployed on Render; **no model inference
happens in this app**. Historical prices for charts are retrieved live from the Kraken public API.
If any API is unavailable or sleeping, the affected section shows a warning and the rest of the app
keeps working.

## Project structure

app/main.py entry point: sidebar, home page, navigation
common/config.py tokens, API URLs (from secrets or environment), timeouts, disclaimer
common/api_client.py API calls with timeouts and error handling, shared display helpers
common/data.py Kraken price history for charts (cached for 1 hour)
students/student_a.py Student A page (home_visual + render)
students/student_b.py Student B page
students/student_c.py Student C page
students/student_d.py Student D page
Dockerfile container build
github.txt repository URL


## Run locally

Requires Python 3.12.

```bash
python -m venv .venv
# Windows: .venv\Scripts\Activate.ps1    macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
cp .streamlit/secrets.toml.example .streamlit/secrets.toml   # then fill in the four Render URLs
streamlit run app/main.py
```

The app opens at http://localhost:8501. API URLs can also be provided as environment variables
(`STUDENT_A_API_URL` ... `STUDENT_D_API_URL`).

## Run with Docker

```bash
docker build -t at3-streamlit .
docker run -p 8501:8501 -e STUDENT_A_API_URL=... -e STUDENT_B_API_URL=... \
  -e STUDENT_C_API_URL=... -e STUDENT_D_API_URL=... at3-streamlit
```

## Deploy on Streamlit Community Cloud

1. New app > select this repository, branch `main`, main file path `app/main.py`.
2. Advanced settings > Python version 3.12.
3. App settings > Secrets: paste the four `STUDENT_X_API_URL` lines from `.streamlit/secrets.toml.example`.

## API contract used by the app

| Endpoint | Method | Body | Used for |
|---|---|---|---|
| `/health` | GET | | service status on the home page |
| `/predict` | POST | `{"token": "BTC", "forecast_horizon": 7}` | prediction, confidence, prediction date |
| `/features` | POST | `{"token": "BTC"}` | engineered features |
| `/model-metadata` | GET | | feature importance, labels |

Render's free tier sleeps idle services, so the first request can take up to a minute. The app waits up
to 60 seconds per prediction and caches results for 10 minutes.

## Contributing (each student works on their own branch)

1. `git switch -c student-x` (your letter).
2. Edit only your page in `students/student_x.py` (plus shared code if agreed with the group).
3. Push and open a pull request into `main`; another student must review and approve it
   (branch protection blocks direct pushes to `main`).

## Disclaimer

University project. Outputs are general information from machine learning models, not financial advice.
