import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
from datetime import datetime, timedelta
from statsmodels.tsa.arima.model import ARIMA
from arch import arch_model
import scipy.stats as stats
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Bitcoin Directional Forecast",
    page_icon="🪙",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3.5rem;
        font-weight: bold;
        color: #FFFFFF;
        text-align: center;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 0.5rem;
        text-align: center;
    }
    .stMetric {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    /* Metric labels and values in black */
    [data-testid="stMetricLabel"], [data-testid="stMetricValue"], [data-testid="stMetricDelta"] {
        color: #000000 !important;
    }
</style>
""", unsafe_allow_html=True)

# Helper Functions
@st.cache_data(ttl=3600)
def download_btc_data(start_date="2014-01-01"):
    """Download Bitcoin price data"""
    btc = yf.download("BTC-USD", start=start_date, progress=False)
    if isinstance(btc.columns, pd.MultiIndex):
        btc.columns = btc.columns.get_level_values(0)
    return btc

def calculate_rsi(series, period=14):
    """Calculate RSI indicator"""
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def prepare_features(btc):
    """Create all features for the model"""
    # Clean data
    btc.index = pd.to_datetime(btc.index)
    btc = btc[~btc.index.duplicated(keep='first')]
    btc.dropna(inplace=True)
    btc.sort_index(inplace=True)
    
    # Log returns
    btc['LogReturn'] = np.log(btc['Close'] / btc['Close'].shift(1))
    
    # Technical features
    btc['RSI'] = calculate_rsi(btc['Close'])
    btc['Rolling_Vol'] = btc['LogReturn'].rolling(window=30).std()
    btc['SMA_20'] = btc['Close'].rolling(window=20).mean()
    btc['Trend_Dist'] = (btc['Close'] / btc['SMA_20']) - 1
    
    # Momentum features
    btc['Momentum_3'] = btc['Close'].pct_change(3)
    btc['Momentum_7'] = btc['Close'].pct_change(7)
    
    # Lag features
    btc['Lag_1'] = btc['LogReturn'].shift(1)
    btc['Lag_2'] = btc['LogReturn'].shift(2)
    
    btc.dropna(inplace=True)
    return btc

@st.cache_resource
def train_model(btc):
    """Train ARIMAX + GARCH model"""
    # Prepare data
    y = btc['LogReturn']
    exog_features = ['RSI', 'Rolling_Vol', 'Trend_Dist', 'Momentum_3', 'Momentum_7', 'Lag_1', 'Lag_2']
    X = btc[exog_features].shift(1)
    
    btc_aligned = btc.copy()
    btc_aligned[exog_features] = X
    btc_aligned.dropna(inplace=True)
    
    y = btc_aligned['LogReturn']
    X = btc_aligned[exog_features]
    
    # Train-test split
    train_size = int(len(btc_aligned) * 0.8)
    y_train = y.iloc[:train_size]
    X_train_raw = X.iloc[:train_size]
    
    # Scale features
    scaler = StandardScaler()
    X_train = pd.DataFrame(scaler.fit_transform(X_train_raw), 
                          index=X_train_raw.index, columns=X_train_raw.columns)
    
    # Train ARIMAX
    model = ARIMA(y_train, exog=X_train, order=(1,0,1))
    model_fit = model.fit()
    
    # Train GARCH
    residuals = model_fit.resid
    garch_model = arch_model(residuals * 100, vol='Garch', p=1, q=1)
    garch_fit = garch_model.fit(disp='off')
    train_vol = garch_fit.conditional_volatility / 100
    
    return model_fit, scaler, train_vol.mean(), exog_features

def make_prediction(btc, model_fit, scaler, avg_vol, exog_features):
    """Generate tomorrow's prediction"""
    # Extract latest features
    latest_features = {
        'RSI': btc['RSI'].iloc[-1],
        'Rolling_Vol': btc['Rolling_Vol'].iloc[-1],
        'Trend_Dist': btc['Trend_Dist'].iloc[-1],
        'Momentum_3': btc['Momentum_3'].iloc[-1],
        'Momentum_7': btc['Momentum_7'].iloc[-1],
        'Lag_1': btc['Lag_1'].iloc[-1],
        'Lag_2': btc['Lag_2'].iloc[-1]
    }
    
    # Scale features
    features_df = pd.DataFrame([latest_features])
    features_scaled = pd.DataFrame(scaler.transform(features_df), columns=exog_features)
    
    # Generate prediction
    forecast_mean = model_fit.forecast(steps=1, exog=features_scaled).values[0]
    forecast_vol = avg_vol
    
    # Calculate probabilities
    prob_up = 1 - stats.norm.cdf((0 - forecast_mean) / forecast_vol)
    prob_down = 1 - prob_up
    
    # Price predictions
    last_price = btc['Close'].iloc[-1]
    expected_price = last_price * np.exp(forecast_mean)
    price_lower = last_price * np.exp(forecast_mean - forecast_vol)
    price_upper = last_price * np.exp(forecast_mean + forecast_vol)
    
    return {
        'current_price': last_price,
        'expected_price': expected_price,
        'price_lower': price_lower,
        'price_upper': price_upper,
        'prob_up': prob_up,
        'prob_down': prob_down,
        'direction': 'UP' if prob_up > 0.5 else 'DOWN',
        'confidence': max(prob_up, prob_down),
        'features': latest_features
    }

# Main Dashboard
def main():
    # Header
    st.markdown('<p class="main-header">Bitcoin Directional Forecast Dashboard</p>', unsafe_allow_html=True)
    st.markdown("### ARIMAX + GARCH Probabilistic Prediction Model")
    
    # Sidebar
    st.sidebar.header("Settings")
    refresh = st.sidebar.button("Refresh Data", use_container_width=True)
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### About")
    st.sidebar.info("""
    This dashboard uses:
    - **ARIMAX(1,0,1)** for directional forecasting
    - **GARCH(1,1)** for volatility modeling
    - **7 Features**: RSI, Volatility, Trend Distance, 3-day & 7-day Momentum, Lag-1 & Lag-2
    """)
    
    # Load data and train model
    with st.spinner("Loading Bitcoin data..."):
        btc = download_btc_data()
        btc = prepare_features(btc)
    
    with st.spinner("Training models..."):
        model_fit, scaler, avg_vol, exog_features = train_model(btc)
    
    with st.spinner("Generating prediction..."):
        prediction = make_prediction(btc, model_fit, scaler, avg_vol, exog_features)
    
    # Main metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Current Price",
            value=f"${prediction['current_price']:,.2f}",
            delta=None
        )
    
    with col2:
        st.metric(
            label=" Expected Price",
            value=f"${prediction['expected_price']:,.2f}",
            delta=f"{((prediction['expected_price'] - prediction['current_price']) / prediction['current_price'] * 100):.2f}%"
        )
    
    with col3:
        direction_emoji = "📈" if prediction['direction'] == 'UP' else "📉"
        st.metric(
            label=f"{direction_emoji} Direction",
            value=prediction['direction'],
            delta=f"{prediction['confidence']:.1%} confidence"
        )
    
    with col4:
        st.metric(
            label="Probability UP",
            value=f"{prediction['prob_up']:.1%}",
            delta=f"{prediction['prob_down']:.1%} DOWN" if prediction['prob_down'] > prediction['prob_up'] else None
        )
    
    st.markdown("---")
    
    # Charts row
    col_chart, col_gauge = st.columns([2, 1])
    
    with col_chart:
        st.subheader("Price History & Prediction (6 Months)")
        
        # Filter last 6 months
        last_6m = datetime.now() - timedelta(days=180)
        plot_data = btc[btc.index >= last_6m].copy()
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot historical price
        ax.plot(plot_data.index, plot_data['Close'], label='Historical Price', 
                color='steelblue', linewidth=2)
        ax.plot(plot_data.index, plot_data['SMA_20'], label='SMA-20', 
                color='orange', linewidth=1.5, linestyle='--', alpha=0.7)
        
        # Plot prediction
        tomorrow = datetime.now() + timedelta(days=1)
        ax.scatter([tomorrow], [prediction['expected_price']], 
                  color='red', s=300, marker='*', zorder=5, 
                  label='Tomorrow Prediction', edgecolor='darkred', linewidths=2)
        
        # Error bars for confidence interval
        ax.errorbar([tomorrow], [prediction['expected_price']], 
                   yerr=[[prediction['expected_price']-prediction['price_lower']], 
                         [prediction['price_upper']-prediction['expected_price']]],
                   fmt='none', ecolor='red', elinewidth=2, capsize=8, alpha=0.6)
        
        # Formatting
        ax.set_ylabel('Price (USD)', fontsize=11)
        ax.set_xlabel('Date', fontsize=11)
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
        
        st.pyplot(fig)
    
    with col_gauge:
        st.subheader("Probability Gauge")
        
        # Create gauge chart
        fig, ax = plt.subplots(figsize=(6, 4), subplot_kw=dict(projection='polar'))
        
        # Probability angle (0 to 180 degrees)
        prob_angle = prediction['prob_up'] * np.pi
        
        # Draw gauge
        theta = np.linspace(0, np.pi, 100)
        radius = np.ones(100)
        
        # Color gradient
        colors = plt.cm.RdYlGn(np.linspace(0, 1, 100))
        for i in range(len(theta)-1):
            ax.plot([theta[i], theta[i+1]], [radius[i], radius[i+1]], 
                   color=colors[i], linewidth=8)
        
        # Needle
        ax.plot([0, prob_angle], [0, 1], color='black', linewidth=3, marker='o', markersize=10)
        
        # Remove labels
        ax.set_ylim(0, 1)
        ax.set_yticks([])
        ax.set_xticks([0, np.pi/2, np.pi])
        ax.set_xticklabels(['DOWN\n0%', '50%', 'UP\n100%'])
        ax.spines['polar'].set_visible(False)
        
        st.pyplot(fig)
        
        # Probability breakdown
        st.markdown(f"""
        <div style='text-align: center; padding: 1rem; background-color: #f0f2f6; border-radius: 0.5rem;'>
            <h3 style='color: {"green" if prediction["direction"] == "UP" else "red"};'>
                {prediction['direction']} Direction
            </h3>
            <p style='font-size: 1.2rem; margin: 0; color: #000000;'>
                <strong style='color: #000000;'>Confidence:</strong> <strong>{prediction['confidence']:.1%}</strong>
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Features and Price Range
    col_features, col_range = st.columns(2)
    
    with col_features:
        st.subheader("Feature Values")
        
        features_df = pd.DataFrame({
            'Feature': ['RSI', 'Rolling Volatility', 'Trend Distance', 
                       '3-Day Momentum', '7-Day Momentum', 'Lag-1', 'Lag-2'],
            'Value': [
                f"{prediction['features']['RSI']:.2f}",
                f"{prediction['features']['Rolling_Vol']:.6f}",
                f"{prediction['features']['Trend_Dist']*100:.2f}%",
                f"{prediction['features']['Momentum_3']*100:.2f}%",
                f"{prediction['features']['Momentum_7']*100:.2f}%",
                f"{prediction['features']['Lag_1']:.6f}",
                f"{prediction['features']['Lag_2']:.6f}"
            ]
        })
        
        st.dataframe(features_df, use_container_width=True, hide_index=True)
    
    with col_range:
        st.subheader("Price Prediction Range")
        
        st.markdown(f"""
        <div style='background-color: #f0f2f6; padding: 1.5rem; border-radius: 0.5rem;'>
            <p style='font-size: 1.1rem; margin: 0.5rem 0; color: #000000;'>
                <strong style='color: #000000;'>Expected Price:</strong> 
                <span style='font-size: 1.3rem; color: #1f77b4;'>${prediction['expected_price']:,.2f}</span>
            </p>
            <p style='font-size: 1rem; margin: 0.5rem 0; color: #000000;'>
                <strong style='color: #000000;'>95% Confidence Range:</strong>
            </p>
            <p style='font-size: 1.1rem; margin: 0.5rem 0; padding-left: 1rem; color: #000000;'>
                <strong style='color: #000000;'>Lower:</strong> <span style='color: red;'>${prediction['price_lower']:,.2f}</span><br>
                <strong style='color: #000000;'>Upper:</strong> <span style='color: green;'>${prediction['price_upper']:,.2f}</span>
            </p>
            <p style='font-size: 0.9rem; margin-top: 1rem; color: #000000;'>
                <em>Based on ARIMAX(1,0,1) + GARCH(1,1) model</em>
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    # Footer
    st.markdown("---")
    st.markdown(f"""
    <div style='text-align: center; color: #666; padding: 1rem;'>
        <p>Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        <p style='font-size: 0.85rem;'>
            This is for educational purposes only. Not financial advice.
        </p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
