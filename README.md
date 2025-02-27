# Technical Analysis Indicators

A Python package for calculating technical analysis indicators for financial data.

## Installation

```bash
pip install .
```

## Usage

```python
import numpy as np
from indicators import auto_register_indicators
from indicators.moving_averages import SimpleMovingAverage

# Register all available indicators
auto_register_indicators()

# Create an indicator
sma = SimpleMovingAverage(timeframe="1h", period=20, source="close")

# Use with numpy array data
data = np.array([...])  # OHLCV data
sma.update(data)
values = sma.values
```

## Available Indicators

- Moving Averages: SMA, EMA, etc.
- Oscillators: RSI, MACD, etc.
- Bands: Bollinger Bands, etc.
- Momentum: Stochastic, ROC, etc.

## License

This project is licensed under the MIT License - see the LICENSE file for details. 