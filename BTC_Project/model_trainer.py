import pandas as pd
import joblib
from statsmodels.tsa.arima.model import ARIMA
from sklearn.preprocessing import StandardScaler
from logger import logging
from custom_exception import CustomException
from utils import create_directories


class ModelTrainer:
    def __init__(self):
        try:
            # Explicit hardcoded paths (as requested)
            self.train_path = "data/processed/train.csv"
            self.test_path = "data/processed/test.csv"

            self.model_path = "artifacts/arimax_model.pkl"
            self.scaler_path = "artifacts/exog_scaler.pkl"

            # Required to store model + scaler
            create_directories(["artifacts"])

            # Exact features used in notebook
            self.exog_features = ["RSI", "Rolling_Vol", "Trend_Dist"]

        except Exception as e:
            raise CustomException(e)

    def load_data(self):
        try:
            logging.info("Loading train and test CSV files...")
            train_df = pd.read_csv(self.train_path)
            test_df = pd.read_csv(self.test_path)

            return train_df, test_df

        except Exception as e:
            raise CustomException(e)

    def prepare_xy(self, train_df, test_df):
        try:
            logging.info("Preparing X (exogenous) and y (target)...")

            # Notebook target
            y_train = train_df["LogReturn"]
            y_test = test_df["LogReturn"]

            # Notebook exogenous variables
            X_train_raw = train_df[self.exog_features]
            X_test_raw = test_df[self.exog_features]

            # Scale X EXACTLY like notebook
            scaler = StandardScaler()
            X_train = pd.DataFrame(
                scaler.fit_transform(X_train_raw),
                columns=self.exog_features,
                index=X_train_raw.index
            )
            X_test = pd.DataFrame(
                scaler.transform(X_test_raw),
                columns=self.exog_features,
                index=X_test_raw.index
            )

            # Save scaler for inference later
            joblib.dump(scaler, self.scaler_path)
            logging.info("Saved StandardScaler for exogenous variables.")

            return y_train, y_test, X_train, X_test

        except Exception as e:
            raise CustomException(e)

    def train_model(self, y_train, X_train):
        try:
            logging.info("Training ARIMAX(1,0,1) model (same as notebook)...")

            model = ARIMA(
                endog=y_train,
                exog=X_train,
                order=(1, 0, 1)  # EXACT notebook values
            )
            model_fit = model.fit()

            # Save fitted ARIMAX model
            joblib.dump(model_fit, self.model_path)
            logging.info("ARIMAX model saved successfully.")

            print(model_fit.summary())

            return model_fit

        except Exception as e:
            raise CustomException(e)

    def run(self):
        try:
            train_df, test_df = self.load_data()
            y_train, y_test, X_train, X_test = self.prepare_xy(train_df, test_df)
            model = self.train_model(y_train, X_train)

            logging.info("Model training completed.")
            return model

        except Exception as e:
            raise CustomException(e)


if __name__ == "__main__":
    ModelTrainer().run()
