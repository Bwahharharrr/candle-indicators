# Technical Analysis Indicators

A Python package for calculating technical analysis indicators for financial data.

## Installation

```bash
pip install .
```

## Usage

For usage examples, please check the `example.py` file in the repository. This provides comprehensive demonstrations of how to use the available indicators.

### Examples

To display all installed/auto-detected indicators (including custom ones) and their parameters with default values:
```bash
python example.py --show-indicators
```

A full working example:
```bash
python example.py --exchange BITFINEX --ticker tBTCUSD --timeframe 1h --start 2025-02-01 --end 2025-04-17
```

Note: The `example.py` file is intentionally kept as simple as possible to serve as a clear reference implementation.

## Custom Indicators

When creating custom or bespoke indicators, it is recommended to name your files with the pattern `my*.py` (e.g., `myindicator.py`, `my_custom_rsi.py`). These files are included in the `.gitignore` and won't be tracked by git.

This repository is intended to contain only standard indicators that match the behavior and calculations of TradingView indicators. Custom implementations should be kept separate using the naming convention above.

## Currently Available Indicators

- SMA (Simple Moving Average)
- EMA (Exponential Moving Average)
- CWEMA (Custom Weighted Exponential Moving Average)
- RSI (Relative Strength Index)
- MACD (Moving Average Convergence Divergence)
- BB (Bollinger Bands)

## License

This project is licensed under the MIT License - see the LICENSE file for details.