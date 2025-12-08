# Additional Features for Bitcoin Prediction Model

## ✅ Currently Implemented Features

### Momentum & Technical Indicators
- **RSI** (14-period) + Binary signals (Oversold/Overbought)
- **Momentum** (1, 3, 7, 14-day price change rates)
- **Moving Average Crosses** (Golden/Death cross signals)

### Volatility Features
- **Rolling Volatility** (10-day and 30-day)
- **Volatility Ratio** (short-term vs long-term)
- **Return Statistics** (5-day mean, std, 20-day skew)

### Trend Features
- **Trend Distance** (Distance from 10/20-day MA)
- **SMA Cross** (10 vs 20-day crossover)

### Price Action
- **High/Low Ratio** (Daily range indicator)
- **Close/Open Difference** (Intraday momentum)

### Lag Features (Memory)
- **Past Returns** (Lag 1, 2, 3, 5)

### Volume Features
- **Volume Ratio** (Current vs 20-day average)
- **Volume Trend** (5-day volume change)

---

## 🚀 Additional Features You Should Add

### 1. **On-Chain Metrics** (Highly Effective for BTC)
These require external data sources but are VERY powerful for crypto:

```python
# Requires APIs like Glassnode, CryptoQuant, or Blockchain.com
btc['Active_Addresses'] = ...  # Daily active addresses (network activity)
btc['MVRV_Ratio'] = ...  # Market Value to Realized Value (overvalued/undervalued)
btc['Exchange_Netflow'] = ...  # BTC flowing into/out of exchanges (sell/buy pressure)
btc['SOPR'] = ...  # Spent Output Profit Ratio (profit-taking indicator)
btc['Mining_Difficulty'] = ...  # Mining difficulty changes
btc['Hash_Rate'] = ...  # Network hash rate (security/health)
btc['NUPL'] = ...  # Net Unrealized Profit/Loss (market sentiment)
```

**Why these work:** On-chain data shows actual blockchain activity and holder behavior, leading price movements.

---

### 2. **Market Sentiment Indicators**

```python
# Fear & Greed Index (CNN Crypto Fear & Greed Index API)
btc['Fear_Greed_Index'] = ...  # 0-100 scale

# Google Trends for "Bitcoin" (pytrends library)
btc['Search_Interest'] = ...  # Search volume for "bitcoin"

# Social Media Sentiment (Twitter API / Reddit API)
btc['Twitter_Sentiment'] = ...  # Positive/negative tweet ratio
btc['Reddit_Mentions'] = ...  # r/bitcoin activity

# Funding Rates (from exchanges like Binance)
btc['Funding_Rate'] = ...  # Perpetual futures funding rate (bullish/bearish)
```

**Why these work:** Sentiment often precedes price movements (FOMO/panic cycles).

---

### 3. **Cross-Asset Correlations**

```python
# Traditional Markets (yfinance)
spy = yf.download('SPY', start=start_date)  # S&P 500
dxy = yf.download('DX-Y.NYB', start=start_date)  # Dollar Index
gold = yf.download('GC=F', start=start_date)  # Gold futures
vix = yf.download('^VIX', start=start_date)  # Volatility Index

btc['SPY_Return'] = spy['Close'].pct_change()
btc['DXY_Return'] = dxy['Close'].pct_change()  # BTC often inverse to dollar
btc['Gold_Return'] = gold['Close'].pct_change()
btc['VIX_Level'] = vix['Close']  # Risk-off indicator

# Crypto Market (ETH as leading indicator)
eth = yf.download('ETH-USD', start=start_date)
btc['ETH_Return'] = eth['Close'].pct_change()
btc['BTC_ETH_Ratio'] = btc['Close'] / eth['Close']  # Dominance
```

**Why these work:** BTC doesn't trade in isolation - macro conditions drive crypto.

---

### 4. **Advanced Technical Indicators**

```python
# Bollinger Bands
btc['BB_Upper'] = btc['Close'].rolling(20).mean() + 2*btc['Close'].rolling(20).std()
btc['BB_Lower'] = btc['Close'].rolling(20).mean() - 2*btc['Close'].rolling(20).std()
btc['BB_Width'] = (btc['BB_Upper'] - btc['BB_Lower']) / btc['Close']
btc['BB_Position'] = (btc['Close'] - btc['BB_Lower']) / (btc['BB_Upper'] - btc['BB_Lower'])

# MACD (Moving Average Convergence Divergence)
ema12 = btc['Close'].ewm(span=12).mean()
ema26 = btc['Close'].ewm(span=26).mean()
btc['MACD'] = ema12 - ema26
btc['MACD_Signal'] = btc['MACD'].ewm(span=9).mean()
btc['MACD_Histogram'] = btc['MACD'] - btc['MACD_Signal']

# ATR (Average True Range) - volatility measure
high_low = btc['High'] - btc['Low']
high_close = abs(btc['High'] - btc['Close'].shift())
low_close = abs(btc['Low'] - btc['Close'].shift())
true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
btc['ATR'] = true_range.rolling(14).mean()

# Stochastic Oscillator
low14 = btc['Low'].rolling(14).min()
high14 = btc['High'].rolling(14).max()
btc['Stochastic_K'] = 100 * (btc['Close'] - low14) / (high14 - low14)
btc['Stochastic_D'] = btc['Stochastic_K'].rolling(3).mean()

# ADX (Average Directional Index) - trend strength
# (More complex - use TA-Lib library: ta.trend.ADXIndicator)
```

**Why these work:** Professional traders use these - capture different market dynamics.

---

### 5. **Time-Based Features (Seasonality)**

```python
# Day of Week Effect (crypto has weekend patterns)
btc['Day_of_Week'] = btc.index.dayofweek  # 0=Monday, 6=Sunday
btc['Is_Weekend'] = (btc['Day_of_Week'] >= 5).astype(int)

# Month Effect
btc['Month'] = btc.index.month

# Quarter Effect
btc['Quarter'] = btc.index.quarter

# Halving Cycles (BTC specific - every 4 years)
halving_dates = pd.to_datetime(['2012-11-28', '2016-07-09', '2020-05-11', '2024-04-15'])
btc['Days_Since_Halving'] = btc.index.to_series().apply(
    lambda x: (x - halving_dates[halving_dates < x].max()).days if any(halving_dates < x) else np.nan
)
btc['Days_Until_Halving'] = btc.index.to_series().apply(
    lambda x: (halving_dates[halving_dates > x].min() - x).days if any(halving_dates > x) else np.nan
)
```

**Why these work:** Crypto markets have calendar effects (weekend dumps, halving cycles).

---

### 6. **Order Book Features** (If Available)

```python
# Requires exchange API (Binance, Coinbase Pro)
btc['Bid_Ask_Spread'] = ...  # Liquidity measure
btc['Order_Book_Imbalance'] = ...  # Bid volume / Ask volume
btc['Depth_10_Levels'] = ...  # Sum of 10 best bid/ask levels
```

**Why these work:** Shows immediate supply/demand pressure.

---

### 7. **Interaction Features**

```python
# Combine existing features for non-linear relationships
btc['RSI_x_Momentum'] = btc['RSI'] * btc['Momentum_7']
btc['Vol_x_Trend'] = btc['Rolling_Vol_30'] * btc['Trend_Dist_20']
btc['RSI_Volatility_Ratio'] = btc['RSI'] / (btc['Rolling_Vol_30'] * 100 + 1)
```

**Why these work:** Capture complex market conditions (e.g., high momentum + oversold RSI = reversal).

---

## 📊 Priority Ranking (What to Add First)

### HIGH PRIORITY (Easy to Get, High Impact)
1. **Cross-Asset Correlations** (SPY, DXY, VIX, ETH) - Use yfinance
2. **Advanced Technical Indicators** (MACD, Bollinger Bands, ATR)
3. **Time-Based Features** (Day of week, halving cycles)
4. **Fear & Greed Index** - Free API available

### MEDIUM PRIORITY (Moderate Effort, Good Impact)
5. **Google Trends** - Free via pytrends library
6. **Funding Rates** - Available from Binance API (free)
7. **Interaction Features** - Easy to compute

### LOW PRIORITY (Hard to Get, But Powerful)
8. **On-Chain Metrics** - Requires paid API (Glassnode ~$300/month)
9. **Social Media Sentiment** - Requires API setup and NLP
10. **Order Book Data** - Requires real-time exchange connection

---

## 🛠️ Implementation Tips

### 1. **Start Small**
Add features in batches of 3-5, then rerun feature selection to see which help.

### 2. **Handle Missing Data**
External APIs may have gaps - use forward fill:
```python
btc['External_Feature'].fillna(method='ffill', inplace=True)
```

### 3. **Check Correlations**
After adding new features, rerun Section 6 to see which are useful.

### 4. **Avoid Overfitting**
More features ≠ better model. Feature selection will remove weak ones.

### 5. **Consider API Costs**
Free tier APIs:
- yfinance (stocks/indices)
- pytrends (Google Trends)
- CoinGecko API (basic crypto data)
- Fear & Greed Index API

Paid APIs (worth it for serious projects):
- Glassnode (on-chain data)
- CryptoQuant (exchange flows)
- Santiment (sentiment + on-chain)

---

## 📈 Expected Accuracy Improvements

With current features: **48-52%** directional accuracy  
Adding cross-asset correlations: **+2-4%** (to 50-56%)  
Adding sentiment indicators: **+1-3%** (to 51-59%)  
Adding on-chain metrics: **+3-7%** (to 54-63%)  

**Realistic best-case with all features: 55-63% accuracy**

Remember: Even 55% accuracy is excellent for crypto. Professional quant funds target 52-58%.

---

## 🎯 Quick Win: Add These 5 Features Now

```python
# 1. S&P 500 (market risk)
spy = yf.download('SPY', start=start_date, progress=False)
btc['SPY_Return'] = spy['Close'].pct_change().reindex(btc.index, method='ffill')

# 2. Dollar Index (BTC inverse correlation)
dxy = yf.download('DX-Y.NYB', start=start_date, progress=False)
btc['DXY_Return'] = dxy['Close'].pct_change().reindex(btc.index, method='ffill')

# 3. VIX (fear gauge)
vix = yf.download('^VIX', start=start_date, progress=False)
btc['VIX_Level'] = vix['Close'].reindex(btc.index, method='ffill')

# 4. Ethereum (crypto market leader)
eth = yf.download('ETH-USD', start=start_date, progress=False)
btc['ETH_Return'] = eth['Close'].pct_change().reindex(btc.index, method='ffill')

# 5. Day of Week
btc['Is_Weekend'] = (btc.index.dayofweek >= 5).astype(int)
```

Then rerun your model with these added to `all_features` list!
