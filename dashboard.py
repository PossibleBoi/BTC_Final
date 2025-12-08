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
import requests

warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Bitcoin Directional Forecast",
    page_icon="₿",
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
@st.cache_data(ttl=1800)  # Cache for 30 minutes
def fetch_crypto_news():
    """Fetch top crypto news from CryptoPanic API"""
    try:
        # CryptoPanic free API (no key required for basic access)
        url = "https://cryptopanic.com/api/v1/posts/?auth_token=free&public=true&kind=news&filter=rising&currencies=BTC"
        response = requests.get(url, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            news_items = []
            
            for item in data.get('results', [])[:5]:  # Get top 5 news
                news_items.append({
                    'title': item.get('title', 'No title'),
                    'url': item.get('url', '#'),
                    'source': item.get('source', {}).get('title', 'Unknown'),
                    'published': item.get('published_at', ''),
                    'votes': item.get('votes', {}).get('positive', 0)
                })
            
            return news_items
        else:
            # Fallback to sample news if API fails
            return get_fallback_news()
    except Exception as e:
        return get_fallback_news()

def get_fallback_news():
    """Fallback news when API is unavailable"""
    return [
        {
            'title': 'Bitcoin Market Analysis - News feed temporarily unavailable',
            'url': 'https://www.coindesk.com',
            'source': 'CoinDesk',
            'published': datetime.now().isoformat(),
            'votes': 0
        }
    ]

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
    
    # 1. MOMENTUM INDICATORS
    btc['RSI'] = calculate_rsi(btc['Close'])
    btc['RSI_Oversold'] = (btc['RSI'] < 30).astype(int)
    btc['RSI_Overbought'] = (btc['RSI'] > 70).astype(int)
    
    # 2. VOLATILITY FEATURES
    btc['Rolling_Vol_10'] = btc['LogReturn'].rolling(window=10).std()
    btc['Rolling_Vol_30'] = btc['LogReturn'].rolling(window=30).std()
    btc['Vol_Ratio'] = btc['Rolling_Vol_10'] / btc['Rolling_Vol_30'].replace(0, np.nan)
    
    # 3. TREND FEATURES
    btc['SMA_10'] = btc['Close'].rolling(window=10).mean()
    btc['SMA_20'] = btc['Close'].rolling(window=20).mean()
    btc['SMA_50'] = btc['Close'].rolling(window=50).mean()
    btc['Trend_Dist_10'] = (btc['Close'] / btc['SMA_10']) - 1
    btc['Trend_Dist_20'] = (btc['Close'] / btc['SMA_20']) - 1
    btc['SMA_Cross'] = (btc['SMA_10'] > btc['SMA_20']).astype(int)
    
    # 4. MOMENTUM (Price Rate of Change)
    btc['Momentum_1'] = btc['Close'].pct_change(1)
    btc['Momentum_3'] = btc['Close'].pct_change(3)
    btc['Momentum_7'] = btc['Close'].pct_change(7)
    btc['Momentum_14'] = btc['Close'].pct_change(14)
    
    # 5. LAG FEATURES
    btc['Lag_1'] = btc['LogReturn'].shift(1)
    btc['Lag_2'] = btc['LogReturn'].shift(2)
    btc['Lag_3'] = btc['LogReturn'].shift(3)
    btc['Lag_5'] = btc['LogReturn'].shift(5)
    
    # 6. VOLUME-BASED
    if 'Volume' in btc.columns:
        btc['Volume_MA'] = btc['Volume'].rolling(window=20).mean()
        btc['Volume_Ratio'] = btc['Volume'] / btc['Volume_MA'].replace(0, np.nan)
        btc['Volume_Trend'] = btc['Volume'].pct_change(5)
    
    # 7. PRICE RANGE FEATURES
    btc['High_Low_Ratio'] = btc['High'] / btc['Low'].replace(0, np.nan)
    btc['Close_Open_Diff'] = (btc['Close'] - btc['Open']) / btc['Open'].replace(0, np.nan)
    
    # 8. RETURN STATISTICS
    btc['Return_Mean_5'] = btc['LogReturn'].rolling(window=5).mean()
    btc['Return_Std_5'] = btc['LogReturn'].rolling(window=5).std()
    btc['Return_Skew_20'] = btc['LogReturn'].rolling(window=20).skew()
    
    btc.dropna(inplace=True)
    return btc

@st.cache_resource
def train_model(btc):
    """Train ARIMAX + GARCH model with automatic feature selection"""
    # All available features
    all_features = [
        'RSI', 'RSI_Oversold', 'RSI_Overbought',
        'Rolling_Vol_10', 'Rolling_Vol_30', 'Vol_Ratio',
        'Trend_Dist_10', 'Trend_Dist_20', 'SMA_Cross',
        'Momentum_1', 'Momentum_3', 'Momentum_7', 'Momentum_14',
        'Lag_1', 'Lag_2', 'Lag_3', 'Lag_5',
        'High_Low_Ratio', 'Close_Open_Diff',
        'Return_Mean_5', 'Return_Std_5', 'Return_Skew_20'
    ]
    
    # Add volume features if available
    if 'Volume_Ratio' in btc.columns:
        all_features.extend(['Volume_Ratio', 'Volume_Trend'])
    
    # Feature selection based on correlation
    y_full = btc['LogReturn'].copy()
    X_full = btc[all_features].shift(1).copy()
    
    valid_idx = ~X_full.isnull().any(axis=1)
    X_full = X_full[valid_idx]
    y_full = y_full[valid_idx]
    
    # Select features with |correlation| > 0.01
    feature_scores = {}
    for feat in all_features:
        corr = X_full[feat].corr(y_full)
        feature_scores[feat] = abs(corr)
    
    sorted_features = sorted(feature_scores.items(), key=lambda x: x[1], reverse=True)
    selected_features = [feat for feat, score in sorted_features if score > 0.01]
    
    # Remove multicollinearity
    if len(selected_features) > 1:
        corr_matrix = X_full[selected_features].corr()
        high_corr_pairs = []
        for i in range(len(selected_features)):
            for j in range(i+1, len(selected_features)):
                corr_val = corr_matrix.iloc[i, j]
                if abs(corr_val) > 0.8:
                    high_corr_pairs.append((selected_features[i], selected_features[j], corr_val))
        
        if high_corr_pairs:
            for feat1, feat2, corr_val in high_corr_pairs:
                if feature_scores[feat1] < feature_scores[feat2]:
                    if feat1 in selected_features:
                        selected_features.remove(feat1)
                else:
                    if feat2 in selected_features:
                        selected_features.remove(feat2)
    
    exog_features = selected_features
    
    # Prepare data with selected features
    y = btc['LogReturn']
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
    
    # Train GARCH (simplified approach like notebook)
    residuals = model_fit.resid
    garch_model = arch_model(residuals * 100, vol='Garch', p=1, q=1)
    garch_fit = garch_model.fit(disp='off')
    
    # Use last conditional volatility (simplified approach from notebook)
    forecast_vol = garch_fit.conditional_volatility.iloc[-1] / 100.0
    
    return model_fit, garch_fit, scaler, forecast_vol, exog_features

def make_prediction(btc, model_fit, garch_fit, scaler, forecast_vol, exog_features):
    """Generate tomorrow's prediction"""
    # Extract latest features dynamically
    latest_features = {}
    for feat in exog_features:
        latest_features[feat] = btc[feat].iloc[-1]
    
    # Scale features
    features_df = pd.DataFrame([latest_features])
    features_scaled = pd.DataFrame(scaler.transform(features_df), columns=exog_features)
    
    # Generate prediction
    forecast_mean = float(model_fit.forecast(steps=1, exog=features_scaled).iloc[0])
    forecast_sigma = forecast_vol
    
    # Calculate probabilities (matching notebook)
    z = forecast_mean / forecast_sigma
    prob_up = float(stats.norm.cdf(z))
    prob_down = 1.0 - prob_up
    
    # Price predictions
    last_price = btc['Close'].iloc[-1]
    expected_price = last_price * np.exp(forecast_mean)
    price_lower = last_price * np.exp(forecast_mean - 1.96 * forecast_sigma)
    price_upper = last_price * np.exp(forecast_mean + 1.96 * forecast_sigma)
    
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
    - **Automatic Feature Selection**: Correlation-based feature selection from 24+ technical indicators
    """)
    
    # Load data and train model
    with st.spinner("Loading Bitcoin data..."):
        btc = download_btc_data()
        btc = prepare_features(btc)
    
    with st.spinner("Training models..."):
        model_fit, garch_fit, scaler, forecast_vol, exog_features = train_model(btc)
    
    with st.spinner("Generating prediction..."):
        prediction = make_prediction(btc, model_fit, garch_fit, scaler, forecast_vol, exog_features)
    
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
        st.metric(
            label="Direction",
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
    
    # News and Price Range
    col_news, col_range = st.columns(2)
    
    with col_news:
        st.subheader("Top Bitcoin News")
        
        # Fetch crypto news
        news_items = fetch_crypto_news()
        
        # Display news in styled cards
        for i, news in enumerate(news_items):
            # Parse time
            try:
                pub_time = datetime.fromisoformat(news['published'].replace('Z', '+00:00'))
                time_ago = datetime.now(pub_time.tzinfo) - pub_time
                if time_ago.days > 0:
                    time_str = f"{time_ago.days}d ago"
                elif time_ago.seconds // 3600 > 0:
                    time_str = f"{time_ago.seconds // 3600}h ago"
                else:
                    time_str = f"{time_ago.seconds // 60}m ago"
            except:
                time_str = "Recently"
            
            # News card with better styling
            st.markdown(f"""
            <div style='background-color: #f8f9fa; padding: 1rem; border-radius: 0.5rem; margin-bottom: 0.8rem; border-left: 4px solid #1f77b4;'>
                <div style='margin-bottom: 0.5rem;'>
                    <a href="{news['url']}" target="_blank" style='color: #1f77b4; text-decoration: none; font-weight: 600; font-size: 0.95rem;'>
                        {news['title']}
                    </a>
                </div>
                <div style='display: flex; justify-content: space-between; align-items: center;'>
                    <span style='color: #666; font-size: 0.8rem;'>{news['source']}</span>
                    <span style='color: #999; font-size: 0.75rem;'>{time_str}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        # Add link to more news
        st.markdown("""
        <div style='text-align: center; margin-top: 1rem;'>
            <a href='https://cryptopanic.com' target='_blank' style='color: #666; font-size: 0.85rem; text-decoration: none;'>
                View more news on CryptoPanic →
            </a>
        </div>
        """, unsafe_allow_html=True)
    
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
                <em>Based on ARIMAX + GARCH model</em>
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
