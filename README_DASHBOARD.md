# Bitcoin Directional Forecast Dashboard

## Quick Start

### 1. Install Dependencies
```powershell
pip install -r requirements_dashboard.txt
```

### 2. Run Dashboard
```powershell
streamlit run dashboard.py
```

The dashboard will open in your browser at `http://localhost:8501`

## Features

### Current Implementation
✅ Real-time Bitcoin price data
✅ ARIMAX + GARCH prediction model
✅ Tomorrow's price prediction with confidence intervals
✅ Probability gauge (UP/DOWN)
✅ 6-month price chart with prediction
✅ Feature values display
✅ Responsive layout

### To Add (Days 3-5)
- [ ] Bitcoin news feed (CryptoPanic API)
- [ ] Historical accuracy tracker
- [ ] Model performance metrics
- [ ] FAQ section (alternative to chatbot)

## Dashboard Layout

```
┌─────────────────────────────────────────────────────┐
│  Current Price | Expected Price | Direction | Prob  │
├──────────────────────────┬──────────────────────────┤
│  Price Chart (6 months)  │  Probability Gauge       │
│  + Tomorrow's Prediction │  (Semicircle Meter)      │
├──────────────────────────┼──────────────────────────┤
│  Feature Values          │  Price Prediction Range  │
│  (RSI, Vol, Momentum)    │  (95% Confidence)        │
└──────────────────────────┴──────────────────────────┘
```

## Troubleshooting

**Issue: Dashboard won't start**
- Make sure all packages are installed: `pip install -r requirements_dashboard.txt`
- Check Python version: Requires Python 3.8+

**Issue: "Port already in use"**
- Stop other Streamlit instances or use: `streamlit run dashboard.py --server.port 8502`

**Issue: Slow loading**
- First load trains the model (~30 seconds)
- Subsequent loads are cached and instant

## Next Steps

1. **Day 3**: Add news feed
   - Register for CryptoPanic API (free)
   - Add news section below features

2. **Day 4**: Add FAQ section
   - Create expandable sections with Bitcoin info
   - Add "How it works" explanation

3. **Day 5**: Polish & test
   - Test on different browsers
   - Add error handling
   - Screenshot for report

## Notes
- Data refreshes every hour (cached)
- Model retrains on refresh
- Uses last ~11 years of BTC data
