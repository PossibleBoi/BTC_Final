import pandas as pd
import numpy as np
from logger import logging
from custom_exception import CustomException
from utils import save_dataframe, create_directories


class DataPreparation:

    def __init__(self):
        try:
            self.raw_data_path = "data/raw/btc_price.csv"
            self.cleaned_path = "data/processed/cleaned.csv"
            self.train_path = "data/processed/train.csv"
            self.test_path = "data/processed/test.csv"

            self.train_ratio = 0.8  # same as notebook

            create_directories(["data/processed"])

        except Exception as e:
            raise CustomException(e)

    def clean_data(self, df: pd.DataFrame):
        try:
            logging.info("Cleaning dataset...")

            # --- FIX NUMERIC TYPES ---
            numeric_cols = ["Open", "High", "Low", "Close", "Volume"]
            for col in numeric_cols:
                df[col] = pd.to_numeric(df[col], errors="coerce")

            # --- DATE SORT & INDEX ---
            df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
            df = df.dropna(subset=["Date"])
            df = df.sort_values("Date").set_index("Date")

            # -------------------------------------
            # 1. LOG RETURN (exact notebook formula)
            # -------------------------------------
            df["LogReturn"] = np.log(df["Close"] / df["Close"].shift(1))
            df["LogReturn"] = df["LogReturn"].replace([np.inf, -np.inf], 0).fillna(0)

            # -------------------------------------
            # 2. RSI (14-period)
            # -------------------------------------
            delta = df["Close"].diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)

            roll_up = gain.rolling(14).mean()
            roll_down = loss.rolling(14).mean()

            RS = roll_up / roll_down
            df["RSI"] = 100 - (100 / (1 + RS))
            df["RSI"] = df["RSI"].fillna(50)

            # -------------------------------------
            # 3. Rolling Volatility (30-day)
            # -------------------------------------
            df["Rolling_Vol"] = df["LogReturn"].rolling(30).std().fillna(0)

            # -------------------------------------
            # 4. Trend Distance (Close - SMA30)
            # -------------------------------------
            df["Trend_Dist"] = df["Close"] - df["Close"].rolling(30).mean()
            df["Trend_Dist"] = df["Trend_Dist"].fillna(0)

            # --------------------------------------------------------
            # 5. SELECT ONLY NOTEBOOK COLUMNS (VERY IMPORTANT)
            # --------------------------------------------------------
            df = df[["LogReturn", "RSI", "Rolling_Vol", "Trend_Dist"]]

            logging.info(f"Final modeling dataframe shape: {df.shape}")
            return df

        except Exception as e:
            raise CustomException(e)

    def split_data(self, df: pd.DataFrame):
        try:
            train_size = int(len(df) * self.train_ratio)

            train = df.iloc[:train_size]
            test = df.iloc[train_size:]

            save_dataframe(train, self.train_path)
            save_dataframe(test, self.test_path)

            return train, test

        except Exception as e:
            raise CustomException(e)

    def run(self):
        try:
            logging.info("Loading raw BTC CSV...")
            df = pd.read_csv(self.raw_data_path)

            cleaned = self.clean_data(df)
            save_dataframe(cleaned, self.cleaned_path)

            train_df, test_df = self.split_data(cleaned)

            logging.info("Data preparation: DONE.")
            return train_df, test_df

        except Exception as e:
            raise CustomException(e)


if __name__ == "__main__":
    DataPreparation().run()
