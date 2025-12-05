#!/usr/bin/env python3
"""
pipeline.py
Full BTC ARIMAX MLOps pipeline:

Steps:
  - prepare  → Clean and feature-engineer BTC price data
  - train    → Train ARIMAX model, save artifacts
  - predict  → Predict tomorrow using trained model and save outputs
  - all      → Run everything (prepare → train → predict)

Usage:
    python pipeline.py prepare
    python pipeline.py train
    python pipeline.py predict
    python pipeline.py all
"""

import argparse
import os
from logger import logging
from custom_exception import CustomException

# Import pipeline components
from data_preparation import DataPreparation
from model_trainer import ModelTrainer
from predictor import forecast_tomorrow  


def ensure_dirs():
    """Ensures required directory structure exists."""
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("artifacts", exist_ok=True)
    os.makedirs("static", exist_ok=True)


def run_prepare():
    logging.info("PIPELINE: Starting DATA PREPARATION...")
    prep = DataPreparation()
    prep.run()
    logging.info("PIPELINE: Data preparation COMPLETE.")


def run_train():
    logging.info("PIPELINE: Starting MODEL TRAINING...")
    trainer = ModelTrainer()
    trainer.run()
    logging.info("PIPELINE: Model training COMPLETE.")


def run_predict():
    logging.info("PIPELINE: Starting PREDICTION...")
    df = forecast_tomorrow()
    logging.info("PIPELINE: Prediction COMPLETE. Saved forecast.csv & forecast.png.")
    return df


def run_all():
    run_prepare()
    run_train()
    result = run_predict()
    logging.info("PIPELINE: ALL steps completed successfully.")
    return result


def main():
    parser = argparse.ArgumentParser(description="BTC ARIMAX ML Pipeline")
    parser.add_argument(
        "step",
        choices=["prepare", "train", "predict", "all"],
        help="Which pipeline step to run",
    )
    args = parser.parse_args()

    ensure_dirs()

    try:
        if args.step == "prepare":
            run_prepare()
        elif args.step == "train":
            run_train()
        elif args.step == "predict":
            run_predict()
        elif args.step == "all":
            run_all()

    except Exception as e:
        logging.exception("PIPELINE FAILED.")
        raise CustomException(e)


if __name__ == "__main__":
    main()
