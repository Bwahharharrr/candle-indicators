#!/usr/bin/env python3
"""
CSV Analysis Example for the candle-indicators package.
This example demonstrates how to load data from a CSV file,
apply technical indicators, and display the results.

# Simple format (default)
python examples/csv_analysis.py --csv data/sample_data.csv

# Pipe format (markdown-compatible)
python examples/csv_analysis.py --csv data/sample_data.csv --format pipe

# Show only the last 10 rows
python examples/csv_analysis.py --csv data/sample_data.csv --lookback 10

"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
from datetime import datetime
from tabulate import tabulate  # You may need to install this: pip install tabulate

# Add the parent directory to sys.path to import the local package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import indicators from the local package
from candle_indicators.moving_averages import SimpleMovingAverage, ExponentialMovingAverage
from candle_indicators.momentum import RSI, MACD
from candle_indicators.bands import BollingerBands

# Default CSV file path - change this to your file
DEFAULT_CSV_PATH = "data/sample_data.csv"

def load_csv_data(file_path):
    """Load OHLCV data from a CSV file."""
    # Read the CSV file
    df = pd.read_csv(file_path)
    
    # Ensure required columns exist
    required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"CSV file must contain column: {col}")
    
    # Convert to structured numpy array
    dtype = np.dtype([
        ('timestamp', 'i8'),
        ('open', 'f8'),
        ('high', 'f8'),
        ('low', 'f8'),
        ('close', 'f8'),
        ('volume', 'f8')
    ])
    
    data = np.zeros(len(df), dtype=dtype)
    data['timestamp'] = df['timestamp'].values
    data['open'] = df['open'].values
    data['high'] = df['high'].values
    data['low'] = df['low'].values
    data['close'] = df['close'].values
    data['volume'] = df['volume'].values
    
    return data

def format_timestamp(timestamp):
    """Format timestamp for display."""
    # Convert milliseconds to seconds if needed
    if timestamp > 1e10:  # Likely milliseconds
        timestamp = timestamp / 1000
    
    # Convert to datetime and format
    dt = datetime.fromtimestamp(timestamp)
    return dt.strftime('%Y-%m-%d %H:%M')

def apply_indicators(data, lookback=None):
    """Apply technical indicators to the data."""
    timeframe = "1h"  # Just for naming
    
    # Initialize indicators
    sma20 = SimpleMovingAverage(timeframe, 20)
    ema20 = ExponentialMovingAverage(timeframe, 20)
    rsi14 = RSI(timeframe, 14)
    macd = MACD(timeframe, 12, 26, 9)
    bb = BollingerBands(timeframe, 20, 2.0)
    
    # Update indicators with data
    sma20.update(data)
    ema20.update(data)
    rsi14.update(data)
    macd.update(data)
    bb.update(data)
    
    # Create results dataframe
    results = pd.DataFrame()
    
    # Add timestamp and price data
    results['Timestamp'] = [format_timestamp(ts) for ts in data['timestamp']]
    results['Open'] = data['open']
    results['High'] = data['high']
    results['Low'] = data['low']
    results['Close'] = data['close']
    results['Volume'] = data['volume']
    
    # Add indicator values
    results['SMA(20)'] = sma20.values
    results['EMA(20)'] = ema20.values
    results['RSI(14)'] = rsi14.values
    
    # Add MACD values
    if macd.values.dtype.names:
        results['MACD'] = macd.values['macd']
        results['Signal'] = macd.values['signal']
        results['Histogram'] = macd.values['histogram']
    
    # Add Bollinger Bands values
    if bb.values.dtype.names:
        results['BB Upper'] = bb.values['upper']
        results['BB Middle'] = bb.values['middle']
        results['BB Lower'] = bb.values['lower']
        results['%B'] = bb.values['percent_b']
    
    # Limit to lookback period if specified
    if lookback and lookback < len(results):
        return results.tail(lookback)
    
    return results

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Analyze CSV data with technical indicators')
    parser.add_argument('--csv', type=str, default=DEFAULT_CSV_PATH,
                        help=f'Path to CSV file (default: {DEFAULT_CSV_PATH})')
    parser.add_argument('--lookback', type=int, default=20,
                        help='Number of rows to display (default: 20)')
    parser.add_argument('--format', type=str, default='simple',
                        choices=['simple', 'grid', 'pipe', 'html', 'latex'],
                        help='Output format (default: simple)')
    args = parser.parse_args()
    
    try:
        # Load data
        print(f"Loading data from {args.csv}...")
        data = load_csv_data(args.csv)
        print(f"Loaded {len(data)} data points")
        
        # Apply indicators
        print("Applying technical indicators...")
        results = apply_indicators(data, args.lookback)
        
        # Display results
        print("\nResults:")
        print(tabulate(results, headers='keys', tablefmt=args.format, floatfmt='.4f'))
        
        print(f"\nAnalysis complete. Displayed {len(results)} rows.")
        
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 
