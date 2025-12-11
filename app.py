import os
from datetime import timedelta

import numpy as np
import pandas as pd
import yfinance as yf
from flask import Flask, jsonify, send_from_directory
from sklearn.ensemble import RandomForestClassifier


app = Flask(__name__, static_folder="static", static_url_path="")


FEATURE_COLS = [
    "Open",
    "High",
    "Low",
    "Close",
    "Volume",
    "MA20",
    "MA50",
    "Daily_Returns",
    "Price_Change",
]


def fetch_and_prepare_data():
    """
    Download Reliance stock data, create features, and return the full DataFrame.
    We keep the latest row (tomorrow target unknown) for inference.
    """
    today = pd.Timestamp.today().normalize()
    # Add one day to ensure today's candle is included if market has closed
    end_date = today + timedelta(days=1)

    df = yf.download("RELIANCE.NS", start="2010-01-01", end=end_date)
    if df.empty:
        raise RuntimeError("Unable to download data for RELIANCE.NS")

    # Ensure consistent columns
    df.columns = ["Open", "High", "Low", "Close", "Volume"]

    # Feature engineering
    df["MA20"] = df["Close"].rolling(20).mean()
    df["MA50"] = df["Close"].rolling(50).mean()
    df["Daily_Returns"] = df["Close"].pct_change()
    df["Price_Change"] = df["Close"] - df["Open"]

    # Keep rows where features exist
    df = df.dropna(subset=FEATURE_COLS).copy()

    # Targets
    df["Tomorrow_Close"] = df["Close"].shift(-1)
    df["Trend"] = np.where(
        df["Tomorrow_Close"].notna(),
        (df["Tomorrow_Close"] > df["Close"]).astype(int),
        np.nan,
    )

    return df


def train_model(df: pd.DataFrame):
    """Train RandomForest on all rows where tomorrow's trend is known."""
    train_df = df[df["Trend"].notna()].copy()
    X_train = train_df[FEATURE_COLS]
    y_train = train_df["Trend"]

    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)
    return model


def build_predictions(df: pd.DataFrame, model: RandomForestClassifier):
    """
    Build predictions for:
    - today: using the previous day's features (predicting the latest known day)
    - tomorrow: using the latest day's features (predicting next day)
    """
    latest_idx = df.index[-1]
    prev_idx = df.index[-2]

    latest_features = df.iloc[-1][FEATURE_COLS].values.reshape(1, -1)
    prev_features = df.iloc[-2][FEATURE_COLS].values.reshape(1, -1)

    tomorrow_pred = model.predict(latest_features)[0]
    tomorrow_proba = model.predict_proba(latest_features)[0]

    today_pred = model.predict(prev_features)[0]
    today_proba = model.predict_proba(prev_features)[0]

    return {
        "latest_date": latest_idx.strftime("%d-%m-%Y"),
        "prev_date": prev_idx.strftime("%d-%m-%Y"),
        "tomorrow_date": (latest_idx + timedelta(days=1)).strftime("%d-%m-%Y"),
        "latest_market": {
            "open": round(df.iloc[-1]["Open"], 2),
            "high": round(df.iloc[-1]["High"], 2),
            "low": round(df.iloc[-1]["Low"], 2),
            "close": round(df.iloc[-1]["Close"], 2),
            "volume": int(df.iloc[-1]["Volume"]),
        },
        "today_prediction": {
            "for_date": latest_idx.strftime("%d-%m-%Y"),
            "trend": "UP" if today_pred == 1 else "DOWN",
            "prob_down": round(float(today_proba[0]), 3),
            "prob_up": round(float(today_proba[1]), 3),
        },
        "tomorrow_prediction": {
            "for_date": (latest_idx + timedelta(days=1)).strftime("%d-%m-%Y"),
            "trend": "UP" if tomorrow_pred == 1 else "DOWN",
            "prob_down": round(float(tomorrow_proba[0]), 3),
            "prob_up": round(float(tomorrow_proba[1]), 3),
        },
    }


@app.route("/api/predict")
def api_predict():
    df = fetch_and_prepare_data()
    model = train_model(df)
    result = build_predictions(df, model)
    return jsonify(result)


@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)

