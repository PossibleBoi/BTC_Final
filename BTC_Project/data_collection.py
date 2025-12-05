import os
import yfinance as yf
import pandas as pd
from datetime import datetime
import time
from custom_exception import CustomException
from logger import logging


class DataCollector:
    def __init__(self, raw_data_path="data/raw/btc_price.csv"):
        self.raw_data_path = raw_data_path

        # Ensure directory exists
        os.makedirs(os.path.dirname(self.raw_data_path), exist_ok=True)

    def download_btc_data(self):
        """Download BTC price data from Yahoo Finance"""
        try:
            logging.info("Downloading BTC-USD data from Yahoo Finance...")

            df = yf.download(
                "BTC-USD",
                start="2014-01-01",
                end=datetime.today().strftime("%Y-%m-%d"),
                auto_adjust=False,   
                progress=False
            )


            if df.empty:
                raise CustomException("Downloaded BTC dataset is empty!")

            df.reset_index(inplace=True)
            logging.info("BTC data downloaded successfully!")

            return df

        except Exception as e:
            raise CustomException(e)

    def save_raw_data(self, df: pd.DataFrame):
        """Save data to the raw folder"""
        try:
            df.to_csv(self.raw_data_path, index=False)
            logging.info(f"Raw BTC data saved to {self.raw_data_path}")
        except Exception as e:
            raise CustomException(e)

    def run(self):
        """Main pipeline"""
        try:
            # if file exists, load it
            if os.path.exists(self.raw_data_path):
                logging.info("Raw data already exists. Loading local file...")
                df = pd.read_csv(self.raw_data_path)
                return df

            # otherwise download
            logging.info("Raw data not found. Downloading new BTC data...")
            df = self.download_btc_data()

            # save
            self.save_raw_data(df)
            return df

        except Exception as e:
            raise CustomException(e)


if __name__ == "__main__":
    collector = DataCollector()
    df = collector.run()
    print(df.head())
