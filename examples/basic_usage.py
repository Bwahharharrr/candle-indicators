#!/usr/bin/env python3
"""
Basic usage example for the candle-indicators package.
This example demonstrates how to use various technical indicators
with sample price data.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# Add the parent directory to sys.path to import the local package
# This allows the example to use the local version of the package instead of an installed one
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Now import from the local package
from candle_indicators.moving_averages import SimpleMovingAverage, ExponentialMovingAverage
from candle_indicators.momentum import RSI, MACD
from candle_indicators.bands import BollingerBands

# Create sample OHLCV data
def generate_sample_data(length=200):
    """Generate sample price data for testing indicators."""
    # Start with a random walk for close prices
    np.random.seed(42)  # For reproducibility
    close = 1000 + np.cumsum(np.random.normal(0, 20, length))
    
    # Generate other price data based on close
    high = close + np.random.uniform(5, 15, length)
    low = close - np.random.uniform(5, 15, length)
    open_prices = close - np.random.normal(0, 10, length)
    volume = np.random.normal(1000000, 200000, length)
    
    # Create structured array with named fields
    dtype = np.dtype([
        ('open', 'f8'),
        ('high', 'f8'),
        ('low', 'f8'),
        ('close', 'f8'),
        ('volume', 'f8')
    ])
    
    data = np.zeros(length, dtype=dtype)
    data['open'] = open_prices
    data['high'] = high
    data['low'] = low
    data['close'] = close
    data['volume'] = volume
    
    return data

def plot_indicator(data, indicator, title, subplot=False, ax=None):
    """Plot an indicator with price data."""
    if not subplot:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={'height_ratios': [3, 1]})
    else:
        ax1, ax2 = ax
    
    # Plot price data
    ax1.plot(data['close'], label='Close Price')
    ax1.set_title('Price Data')
    ax1.set_ylabel('Price')
    ax1.grid(True)
    ax1.legend()
    
    # Plot indicator
    if indicator.values.dtype.names:
        # Handle structured arrays (like MACD, Bollinger Bands)
        if 'macd' in indicator.values.dtype.names:
            # MACD has special plotting
            ax2.plot(indicator.values['macd'], label='MACD Line')
            ax2.plot(indicator.values['signal'], label='Signal Line')
            ax2.bar(np.arange(len(indicator.values)), 
                   indicator.values['histogram'], 
                   alpha=0.3, label='Histogram')
        elif 'upper' in indicator.values.dtype.names:
            # Bollinger Bands
            ax1.plot(indicator.values['upper'], 'r--', label='Upper Band')
            ax1.plot(indicator.values['middle'], 'g--', label='Middle Band')
            ax1.plot(indicator.values['lower'], 'r--', label='Lower Band')
            ax1.fill_between(np.arange(len(indicator.values)), 
                            indicator.values['upper'], 
                            indicator.values['lower'], 
                            alpha=0.1)
            ax2.plot(indicator.values['percent_b'], label='%B')
    else:
        # Simple indicators like SMA, EMA, RSI
        ax2.plot(indicator.values, label=str(indicator))
        
    ax2.set_title(title)
    ax2.set_xlabel('Candle Index')
    ax2.grid(True)
    ax2.legend()
    
    if not subplot:
        plt.tight_layout()
        plt.show()

def calculate_rsi_directly(prices, period=14):
    """Calculate RSI directly without using the package implementation."""
    # Calculate price changes
    deltas = np.diff(prices)
    
    # Initialize arrays
    gains = np.zeros_like(deltas)
    losses = np.zeros_like(deltas)
    
    # Separate gains and losses
    gains[deltas > 0] = deltas[deltas > 0]
    losses[deltas < 0] = -deltas[deltas < 0]
    
    # Initialize result array
    rsi = np.zeros(len(prices))
    rsi[:] = np.nan
    
    # First average is simple average
    first_avg_gain = np.mean(gains[:period]) if len(gains) >= period else 0
    first_avg_loss = np.mean(losses[:period]) if len(losses) >= period else 0
    
    # Use Wilder's smoothing method
    avg_gain = np.zeros(len(deltas))
    avg_loss = np.zeros(len(deltas))
    
    if len(gains) >= period:
        avg_gain[period-1] = first_avg_gain
        avg_loss[period-1] = first_avg_loss
        
        # Calculate smoothed averages
        for i in range(period, len(deltas)):
            avg_gain[i] = (avg_gain[i-1] * (period-1) + gains[i]) / period
            avg_loss[i] = (avg_loss[i-1] * (period-1) + losses[i]) / period
        
        # Calculate RS and RSI
        rs = avg_gain / np.maximum(avg_loss, 1e-10)  # Avoid division by zero
        rsi_values = 100 - (100 / (1 + rs))
        
        # Assign RSI values to result array (offset by 1 due to diff)
        rsi[period:] = rsi_values[period-1:]
    
    return rsi

# Add this before the individual RSI plot section
# Create a simple wrapper object to mimic the Indicator interface
class DirectRSI:
    def __init__(self, values):
        self.values = values
        self.name = "DirectRSI(14)"
        self.timeframe = "1h"
    
    def __str__(self):
        return self.name

if __name__ == "__main__":
    print("Candle Indicators - Basic Usage Example")
    
    # Generate sample data
    data = generate_sample_data(200)
    print(f"Generated sample data with {len(data)} candles")
    
    # Example 1: Simple Moving Average (SMA)
    timeframe = "1h"  # Timeframe is just for naming in this example
    sma20 = SimpleMovingAverage(timeframe, 20)
    sma20.update(data)
    print(f"SMA(20) last 5 values: {sma20.values[-5:]}")
    
    # Example 2: Exponential Moving Average (EMA)
    ema20 = ExponentialMovingAverage(timeframe, 20)
    ema20.update(data)
    print(f"EMA(20) last 5 values: {ema20.values[-5:]}")
    
    # Example 3: RSI
    rsi14 = RSI(timeframe, 14)
    rsi14.update(data)
    
    # Debug RSI values from package
    print(f"Package RSI(14) shape: {rsi14.values.shape}")
    print(f"Package RSI(14) non-NaN values: {np.sum(~np.isnan(rsi14.values))}")

    # Calculate RSI directly
    direct_rsi = calculate_rsi_directly(data['close'], 14)
    print(f"Direct RSI(14) shape: {direct_rsi.shape}")
    print(f"Direct RSI(14) non-NaN values: {np.sum(~np.isnan(direct_rsi))}")
    print(f"Direct RSI(14) min: {np.nanmin(direct_rsi)}")
    print(f"Direct RSI(14) max: {np.nanmax(direct_rsi)}")
    print(f"Direct RSI(14) last 5 values: {direct_rsi[-5:]}")

    # Use the direct RSI for plotting
    # Plot 2: RSI
    plt.subplot(2, 2, 2)
    valid_indices = ~np.isnan(direct_rsi)
    plt.plot(np.arange(len(direct_rsi))[valid_indices], direct_rsi[valid_indices], 'b-')
    plt.axhline(y=70, color='r', linestyle='-', alpha=0.3)
    plt.axhline(y=30, color='g', linestyle='-', alpha=0.3)
    plt.title('RSI(14) - Direct Calculation')
    plt.ylim(0, 100)
    plt.grid(True)
    
    # Example 4: MACD
    macd = MACD(timeframe, 12, 26, 9)
    macd.update(data)
    print("MACD last value:")
    last_value = macd.get_value()
    for label, value in macd.format_value(last_value):
        print(f"  {label}: {value}")
    
    # Example 5: Bollinger Bands
    bb = BollingerBands(timeframe, 20, 2.0)
    bb.update(data)
    print("Bollinger Bands last value:")
    last_value = bb.get_value()
    for label, value in bb.format_value(last_value):
        print(f"  {label}: {value}")
    
    # Plot all indicators
    plt.figure(figsize=(15, 10))
    
    # Plot 1: Moving Averages
    plt.subplot(2, 2, 1)
    plt.plot(data['close'], label='Close')
    plt.plot(sma20.values, label='SMA(20)')
    plt.plot(ema20.values, label='EMA(20)')
    plt.title('Moving Averages')
    plt.grid(True)
    plt.legend()
    
    # Plot 3: MACD
    plt.subplot(2, 2, 3)
    plt.plot(macd.values['macd'], label='MACD Line')
    plt.plot(macd.values['signal'], label='Signal Line')
    plt.bar(np.arange(len(macd.values)), 
           macd.values['histogram'], 
           alpha=0.3, label='Histogram')
    plt.title('MACD(12,26,9)')
    plt.grid(True)
    plt.legend()
    
    # Plot 4: Bollinger Bands
    plt.subplot(2, 2, 4)
    plt.plot(data['close'], label='Close')
    plt.plot(bb.values['upper'], 'r--', label='Upper Band')
    plt.plot(bb.values['middle'], 'g--', label='Middle Band')
    plt.plot(bb.values['lower'], 'r--', label='Lower Band')
    plt.fill_between(np.arange(len(bb.values)), 
                    bb.values['upper'], 
                    bb.values['lower'], 
                    alpha=0.1)
    plt.title('Bollinger Bands(20,2)')
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    # Detailed individual plots
    print("\nGenerating detailed plots for each indicator...")
    
    # Individual plot for SMA
    plot_indicator(data, sma20, "Simple Moving Average (20)")
    
    # Individual plot for EMA
    plot_indicator(data, ema20, "Exponential Moving Average (20)")
    
    # Individual plot for RSI
    print("\nPlotting direct RSI calculation:")
    direct_rsi_obj = DirectRSI(direct_rsi)
    plot_indicator(data, direct_rsi_obj, "Relative Strength Index (14) - Direct Calculation")
    
    # Individual plot for MACD
    plot_indicator(data, macd, "MACD (12,26,9)")
    
    # Individual plot for Bollinger Bands
    plot_indicator(data, bb, "Bollinger Bands (20,2)")
    
    print("Example completed successfully!") 