from flask import Flask, render_template
import pandas as pd
import os

from predictor import forecast_tomorrow

app = Flask(__name__)

FORECAST_CSV = "artifacts/forecast.csv"
FORECAST_PLOT = "static/forecast.png"


@app.route("/")
def index():

    # Run prediction fresh every time page loads
    forecast_df = forecast_tomorrow()

    result = forecast_df.iloc[0].to_dict()

    # Example: {'Predicted_Close': 43000.23, 'Recommendation': 'BUY', ...}

    return render_template(
        "index.html",
        prediction=result,
        forecast_plot=FORECAST_PLOT
    )


if __name__ == "__main__":
    app.run(debug=True)
