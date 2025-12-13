import streamlit as st
from datetime import timedelta
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier
from pandas.tseries.offsets import BDay

# Page config
st.set_page_config(
    page_title="Reliance Stock Predictor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS to match original Flask design exactly
st.markdown("""
<style>
    :root {
        --bg: #0f172a;
        --card: #111827;
        --accent: #22d3ee;
        --accent-2: #a855f7;
        --text: #e5e7eb;
        --muted: #94a3b8;
        --danger: #f43f5e;
        --success: #34d399;
    }
    
    html, body {
        background: radial-gradient(circle at 20% 20%, #13203d, #0b1226 40%),
                    radial-gradient(circle at 80% 0%, #1f2031, #0b1226 50%),
                    #0f172a !important;
    }
    
    .stApp {
        background: radial-gradient(circle at 20% 20%, #13203d, #0b1226 40%),
                    radial-gradient(circle at 80% 0%, #1f2031, #0b1226 50%),
                    #0f172a !important;
    }
    
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 900px;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    .glass-container {
        background: rgba(255, 255, 255, 0.04);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 18px;
        box-shadow: 0 20px 60px rgba(0, 0, 0, 0.45);
        backdrop-filter: blur(16px);
        padding: 28px;
        margin: 20px 0;
    }
    
    .pill {
        display: inline-block;
        padding: 10px 14px;
        border-radius: 999px;
        background: linear-gradient(120deg, rgba(34, 211, 238, 0.2), rgba(168, 85, 247, 0.2));
        color: #e5e7eb;
        font-weight: 600;
        border: 1px solid rgba(255, 255, 255, 0.08);
        margin-bottom: 10px;
    }
    
    h1 {
        margin: 0;
        font-size: 26px;
        letter-spacing: -0.02em;
        color: #FFFFFF;
    }
    
    .prediction-card {
        background: rgba(17, 24, 39, 0.8);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 14px;
        padding: 16px;
        margin: 10px 0;
    }
    
    .prediction-card h3 {
        margin: 0 0 6px;
        font-size: 16px;
        color: #94a3b8;
        letter-spacing: 0.02em;
        text-transform: uppercase;
    }
    
    .value {
        font-size: 28px;
        font-weight: 800;
        letter-spacing: -0.01em;
        color: #e5e7eb;
    }
    
    .muted-text {
        color: #94a3b8;
        font-size: 14px;
    }
    
    .tag {
        display: inline-block;
        padding: 6px 10px;
        border-radius: 10px;
        background: rgba(255, 255, 255, 0.06);
        font-weight: 700;
        letter-spacing: 0.01em;
        font-size: 28px;
    }
    
    .tag.up {
        color: #34d399;
        border: 1px solid rgba(52, 211, 153, 0.3);
    }
    
    .tag.down {
        color: #f43f5e;
        border: 1px solid rgba(244, 63, 94, 0.3);
    }
    
    .status {
        font-size: 14px;
        color: #e5e7eb;
        margin: 10px 0;
    }
    
    .dot {
        display: inline-block;
        width: 10px;
        height: 10px;
        border-radius: 50%;
        background: #22d3ee;
        box-shadow: 0 0 0 6px rgba(34, 211, 238, 0.12);
        margin-right: 8px;
    }
    
    /* Button styling */
    .stButton > button {
        cursor: pointer;
        border: none;
        padding: 12px 18px;
        border-radius: 12px;
        font-weight: 700;
        transition: transform 0.15s ease;
        width: 100%;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
    }
    
    div[data-testid="stButton"]:has(button[kind="primary"]) > button {
        background: linear-gradient(120deg, #22d3ee, #a855f7);
        box-shadow: 0 12px 24px rgba(34, 211, 238, 0.2);
        color: #0f172a;
    }
    
    div[data-testid="stButton"]:has(button:not([kind="primary"])) > button {
        background: #1f2937;
        color: #e5e7eb;
        border: 1px solid rgba(255, 255, 255, 0.08);
    }
    
    .footer-text {
        color: #94a3b8;
        font-size: 13px;
        margin-top: 16px;
    }
</style>
""", unsafe_allow_html=True)

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


@st.cache_data(ttl=300)  # Cache for 5 minutes
def fetch_and_prepare_data():
    """
    Download Reliance stock data, create features, and return the full DataFrame.
    We keep the latest row (tomorrow target unknown) for inference.
    """
    today = pd.Timestamp.today().normalize()
    # Add one day to ensure today's candle is included if market has closed
    end_date = today + timedelta(days=1)

    df = yf.download("RELIANCE.NS", start="2010-01-01", end=end_date, auto_adjust=True)
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


@st.cache_data(ttl=300)
def train_model(df: pd.DataFrame):
    """Train RandomForest on all rows where tomorrow's trend is known."""
    train_df = df[df["Trend"].notna()].copy()
    X_train = train_df[FEATURE_COLS]
    y_train = train_df["Trend"]

    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)
    return model


def build_predictions(df: pd.DataFrame, model: RandomForestClassifier):
    latest_idx = df.index[-1]      # last completed trading day
    prev_idx = df.index[-2]        # previous trading day

    next_trading_day = latest_idx + BDay(1)

    latest_features = pd.DataFrame(
        [df.iloc[-1][FEATURE_COLS]],
        columns=FEATURE_COLS
    )
    prev_features = pd.DataFrame(
        [df.iloc[-2][FEATURE_COLS]],
        columns=FEATURE_COLS
    )

    tomorrow_pred = model.predict(latest_features)[0]
    tomorrow_proba = model.predict_proba(latest_features)[0]

    today_pred = model.predict(prev_features)[0]
    today_proba = model.predict_proba(prev_features)[0]

    return {
        "latest_date": latest_idx.strftime("%d-%m-%Y"),
        "prev_date": prev_idx.strftime("%d-%m-%Y"),
        "tomorrow_date": next_trading_day.strftime("%d-%m-%Y"),

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
            "for_date": next_trading_day.strftime("%d-%m-%Y"),
            "trend": "UP" if tomorrow_pred == 1 else "DOWN",
            "prob_down": round(float(tomorrow_proba[0]), 3),
            "prob_up": round(float(tomorrow_proba[1]), 3),
        },
    }


# Initialize session state
if "predictions" not in st.session_state:
    st.session_state.predictions = None
if "status" not in st.session_state:
    st.session_state.status = "Ready"

# Main UI - wrapped in glass container
st.markdown('<div class="glass-container">', unsafe_allow_html=True)

# Header
st.markdown("""
<div style="display: flex; justify-content: space-between; align-items: flex-start; gap: 20px; flex-wrap: wrap; margin-bottom: 20px;">
    <div>
        <div class="pill">Reliance Stock Predictor</div>
        <h1>AI-assisted trend outlook for today & tomorrow</h1>
    </div>
    <div style="display: flex; gap: 10px; padding-top: 10px;">
""", unsafe_allow_html=True)

# Buttons
col1, col2 = st.columns(2)
with col1:
    refresh_btn = st.button("Refresh data", use_container_width=True)
with col2:
    predict_btn = st.button("Predict now", type="primary", use_container_width=True)

st.markdown("</div></div>", unsafe_allow_html=True)

# Prediction logic - only run when buttons are clicked
if predict_btn or refresh_btn:
    try:
        # Clear cache to force fresh data fetch
        fetch_and_prepare_data.clear()
        train_model.clear()
        
        with st.spinner("Fetching latest data & running model..."):
            df = fetch_and_prepare_data()
            model = train_model(df)
            result = build_predictions(df, model)
            st.session_state.predictions = result
            st.session_state.status = f"Updated just now • Based on data through {result['latest_date']}"
    except Exception as e:
        st.error(f"Error: {str(e)}")
        st.session_state.status = "Error: unable to fetch prediction"
        st.session_state.predictions = None

# Status indicator
status_color = "#22d3ee" if "Ready" in st.session_state.status or "Updated" in st.session_state.status else "#f59e0b"
st.markdown(f'<div class="status"><span class="dot" style="background: {status_color}"></span>{st.session_state.status}</div>', unsafe_allow_html=True)

# Display predictions
if st.session_state.predictions:
    data = st.session_state.predictions
    
    # Three columns for predictions
    col1, col2, col3 = st.columns(3)
    
    with col1:
        m = data["latest_market"]
        st.markdown(f'''
        <div class="prediction-card">
            <h3>Today's Market (latest)</h3>
            <div class="value">Market date: {data['latest_date']}</div>
            <p class="muted-text">O {m["open"]} • H {m["high"]} • L {m["low"]} • C {m["close"]} • Vol {m["volume"]:,}</p>
        </div>
        ''', unsafe_allow_html=True)
    
    with col2:
        trend_class = "tag up" if data["today_prediction"]["trend"] == "UP" else "tag down"
        emoji = "📈" if data["today_prediction"]["trend"] == "UP" else "📉"
        st.markdown(f'''
        <div class="prediction-card">
            <h3>Today Prediction</h3>
            <div class="value"><span class="{trend_class}">{emoji} {data["today_prediction"]["trend"]}</span></div>
            <p class="muted-text">P(Down): {data["today_prediction"]["prob_down"]} | P(Up): {data["today_prediction"]["prob_up"]}</p>
        </div>
        ''', unsafe_allow_html=True)
    
    with col3:
        trend_class = "tag up" if data["tomorrow_prediction"]["trend"] == "UP" else "tag down"
        emoji = "📈" if data["tomorrow_prediction"]["trend"] == "UP" else "📉"
        st.markdown(f'''
        <div class="prediction-card">
            <h3>Tomorrow Prediction</h3>
            <div class="value"><span class="{trend_class}">{emoji} {data["tomorrow_prediction"]["trend"]}</span></div>
            <p class="muted-text">P(Down): {data["tomorrow_prediction"]["prob_down"]} | P(Up): {data["tomorrow_prediction"]["prob_up"]} (for {data["tomorrow_prediction"]["for_date"]})</p>
        </div>
        ''', unsafe_allow_html=True)
else:
    # Placeholder when no data yet
    st.info("No data yet. Click 'Predict now' to fetch the latest market data and predictions.")

# Footer
st.markdown("""
<div class="footer-text" style="margin-top: 16px; display: flex; justify-content: space-between; gap: 10px; flex-wrap: wrap;">
    <div>Model: RandomForest (200 trees) trained on RELIANCE.NS daily data (2010 - today).</div>
    <div>Data powered by Yahoo Finance • Tap "Predict now" to update.</div>
</div>
""", unsafe_allow_html=True)

# Close glass container
st.markdown('</div>', unsafe_allow_html=True)

