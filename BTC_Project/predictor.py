import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import os
from logger import logging
from custom_exception import CustomException

ARTIFACT_MODEL = "artifacts/arimax_model.pkl"
ARTIFACT_SCALER = "artifacts/exog_scaler.pkl"
CLEANED_DATA = "data/processed/cleaned.csv"
RAW_DATA = "data/raw/btc_price.csv"

FORECAST_CSV = "artifacts/forecast.csv"
FORECAST_PLOT = "static/forecast.png"


# -------------------------------
# Feature Engineering (Same as Notebook)
# -------------------------------

def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def compute_features(df):
    df["RSI"] = compute_rsi(df["Close"])
    df["Rolling_Vol"] = df["LogReturn"].rolling(30).std()
    df["Trend_Dist"] = (df["Close"] - df["Close"].rolling(200).mean()) / df["Close"].rolling(200).mean()
    df = df.dropna()
    return df


# -------------------------------
# Generate Tomorrow’s Exogenous Feature Row
# -------------------------------
def generate_tomorrow_features(df):
    last_row = df.iloc[-1]

    tomorrow = pd.DataFrame({
        "RSI": [last_row["RSI"]],
        "Rolling_Vol": [last_row["Rolling_Vol"]],
        "Trend_Dist": [last_row["Trend_Dist"]]
    })

    tomorrow.index = [df.index[-1] + pd.Timedelta(days=1)]
    return tomorrow


# -------------------------------
# BUY / SELL / HOLD Recommendation
# -------------------------------
def make_recommendation(expected_return):
    if expected_return > 0.01:  
        return "BUY — Model expects strong upward movement."
    elif expected_return < -0.01:
        return "SELL — Model expects decline."
    else:
        return "HOLD — Low volatility expected."


# -------------------------------
# Predictor (Main Function)
# -------------------------------
def forecast_tomorrow():
    try:
        logging.info("Loading cleaned dataset (features)...")
        features_df = pd.read_csv(CLEANED_DATA)
        
        logging.info("Loading raw dataset (for Close prices)...")
        raw_df = pd.read_csv(RAW_DATA)
        # Skip the second header row with 'BTC-USD' values
        if raw_df.iloc[0]['Close'] == 'BTC-USD':
            raw_df = raw_df.iloc[1:].reset_index(drop=True)
        raw_df['Close'] = pd.to_numeric(raw_df['Close'])
        
        # Align the dataframes - they should have same length after cleaning
        # Use the last N rows from raw_df to match features_df length
        raw_df = raw_df.tail(len(features_df)).reset_index(drop=True)

        logging.info("Generating tomorrow feature row...")
        tomorrow_features_raw = generate_tomorrow_features(features_df)

        logging.info("Loading trained model & scaler...")
        model = joblib.load(ARTIFACT_MODEL)
        scaler = joblib.load(ARTIFACT_SCALER)

        logging.info("Scaling tomorrow exog features...")
        tomorrow_exog = pd.DataFrame(
            scaler.transform(tomorrow_features_raw),
            index=tomorrow_features_raw.index,
            columns=tomorrow_features_raw.columns
        )

        logging.info("Predicting TOMORROW log return...")
        pred_log_return = model.forecast(steps=1, exog=tomorrow_exog)[0]

        last_price = raw_df["Close"].iloc[-1]
        predicted_price = last_price * np.exp(pred_log_return)

        recommendation = make_recommendation(pred_log_return)

        forecast_df = pd.DataFrame({
            "Date": tomorrow_exog.index,
            "Predicted_LogReturn": [pred_log_return],
            "Last_Close": [last_price],
            "Predicted_Close": [predicted_price],
            "Recommendation": [recommendation]
        })

        # Save CSV
        forecast_df.to_csv(FORECAST_CSV, index=False)
        logging.info(f"Saved forecast CSV → {FORECAST_CSV}")

        # Save plot
        plt.figure(figsize=(8, 4))
        plt.title("BTC Price Forecast for Tomorrow")
        last_60_prices = raw_df["Close"].tail(60).values
        plt.plot(range(len(last_60_prices)), last_60_prices, label="Historical Close")
        plt.scatter(len(last_60_prices), predicted_price, color='red', s=100, label="Tomorrow Prediction", zorder=5)
        plt.xlabel("Days")
        plt.ylabel("Price (USD)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(FORECAST_PLOT)
        plt.close()
        logging.info(f"Saved forecast plot → {FORECAST_PLOT}")

        logging.info("Prediction completed successfully.")
        return forecast_df

    except Exception as e:
        raise CustomException(e)
