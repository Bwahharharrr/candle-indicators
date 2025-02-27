#!/usr/bin/env python3
"""
Historical Analysis Tool for Cryptocurrency Data

This script analyzes historical candle data across multiple timeframes,
calculating various technical indicators and displaying the results.

Examples:
    # Basic usage with default indicators
    python historical_analysis.py --exchange binance --ticker BTC/USDT --timeframe 1h

    # Analyze specific timeframes with SMA indicator
    python historical_analysis.py --exchange binance --ticker ETH/USDT --timeframe 5m \
        --analysis-tfs 15m 1h 4h --indicators SMA20:1h SMA50:4h

    # Analyze with date range and multiple indicators
    python historical_analysis.py --exchange kraken --ticker XRP/USD --timeframe 1m \
        --start 2023-01-01 --end 2023-02-01 \
        --analysis-tfs 5m 15m 1h 4h \
        --indicators SMA20 RSI14 MACD12,26,9 BB20,2

    # Show all available indicators
    python historical_analysis.py --exchange any --ticker any --timeframe any --indicators

    # Advanced timeframe filtering with indicators
    python historical_analysis.py --exchange binance --ticker SOL/USDT --timeframe 1m \
        --analysis-tfs ">=15m" "<=4h" \
        --indicators "RSI14:>=15m" "BB20,2:1h:close" "SMA50:4h"
"""
import argparse
import sys
import os
from candle_iterator import analyze_candle_data
import numpy as np
from typing import Dict, List, Set, Tuple, Optional, Callable, Any, Union
from dataclasses import dataclass, field
import time
import re
from abc import ABC, abstractmethod


# Add the parent directory to sys.path to import the local package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from candle_indicators.base import Indicator, IndicatorRegistry
from candle_indicators import auto_register_indicators

# Add colorama for cross-platform colored terminal output
try:
    from colorama import init, Fore, Style
    init()  # Initialize colorama
    HAS_COLORS = True
except ImportError:
    # Create dummy color classes if colorama is not available
    class DummyFore:
        def __getattr__(self, name):
            return ""
    class DummyStyle:
        def __getattr__(self, name):
            return ""
    Fore = DummyFore()
    Style = DummyStyle()
    HAS_COLORS = False

#------------------------------------------------------------------------------
# Utility Functions
#------------------------------------------------------------------------------

def parse_timeframe_minutes(tf: str) -> int:
    """Convert timeframe string to minutes.
    
    Handles formats like:
    - 5m (5 minutes)
    - 2h (2 hours)
    - 1D (1 day)
    - 1W (1 week)
    """
    # Use regex to extract number and unit
    match = re.match(r'(\d*)([mhDW])', tf)
    if not match:
        return 0
        
    num_str, unit = match.groups()
    num = int(num_str) if num_str else 1
    
    # Convert different time units to minutes
    if unit == 'm':
        return num
    elif unit == 'h':
        return num * 60
    elif unit == 'D':
        return num * 1440  # 24 * 60
    elif unit == 'W':
        return num * 10080  # 7 * 24 * 60
    return 0

def get_timeframe_sort_key(tf: str) -> int:
    """Get a sort key for timeframes to ensure they're displayed in ascending order."""
    return parse_timeframe_minutes(tf)

def matches_timeframe_filter(tf: str, filter_expr: str) -> bool:
    """Check if timeframe matches filter expression.
    
    Supports expressions like:
    - "1h" (exact match)
    - ">=1h" (greater than or equal to 1 hour)
    - "<=4h" (less than or equal to 4 hours)
    - ">15m" (greater than 15 minutes)
    - "<1D" (less than 1 day)
    - "=2h" (exactly 2 hours)
    """
    # Direct match
    if not filter_expr.startswith(('>=', '<=', '>', '<', '=')):
        return tf == filter_expr
        
    tf_minutes = parse_timeframe_minutes(tf)
    if tf_minutes == 0:
        return False
    
    # Handle equality first
    if filter_expr.startswith('='):
        compare_tf = filter_expr[1:]
        return tf_minutes == parse_timeframe_minutes(compare_tf)
        
    # Handle other comparisons
    compare_tf = filter_expr[2:] if filter_expr.startswith(('>=', '<=')) else filter_expr[1:]
    compare_minutes = parse_timeframe_minutes(compare_tf)
    
    if filter_expr.startswith('>='):
        return tf_minutes >= compare_minutes
    elif filter_expr.startswith('<='):
        return tf_minutes <= compare_minutes
    elif filter_expr.startswith('>'):
        return tf_minutes > compare_minutes
    elif filter_expr.startswith('<'):
        return tf_minutes < compare_minutes
    
    return False

def format_candle(i: int, candle: np.ndarray) -> str:
    """Format a candle for display with index, timestamp and OHLCV values."""
    return (f"[{i}] Timestamp: {candle['timestamp']}, "
            f"Open: {candle['open']:.2f}, "
            f"High: {candle['high']:.2f}, "
            f"Low: {candle['low']:.2f}, "
            f"Close: {candle['close']:.2f}, "
            f"Volume: {candle['volume']:.8f}")

def get_available_timeframes() -> List[str]:
    """Get all available timeframes that can be analyzed.
    
    Returns a comprehensive list of standard timeframes used in crypto trading,
    from 1-minute to 1-week intervals.
    """
    # Minutes
    minute_tfs = [f"{m}m" for m in [1, 3, 5, 15, 30]]
    # Hours
    hour_tfs = [f"{h}h" for h in [1, 2, 3, 4, 6, 8, 12]]
    # Days and weeks
    day_tfs = ["1D", "2D", "3D", "1W"]
    
    return minute_tfs + hour_tfs + day_tfs

def filter_timeframes(all_timeframes: List[str], filter_expressions: List[str]) -> List[str]:
    """Filter timeframes based on filter expressions.
    
    Takes a list of all possible timeframes and returns only those
    that match at least one of the provided filter expressions.
    
    Raises ValueError if no timeframes match the filters.
    """
    selected_timeframes = [
        tf for tf in all_timeframes 
        if any(matches_timeframe_filter(tf, expr) for expr in filter_expressions)
    ]
    
    if not selected_timeframes:
        raise ValueError(f"No valid timeframes selected from filter: {filter_expressions}")
        
    return selected_timeframes

#------------------------------------------------------------------------------
# Data Containers
#------------------------------------------------------------------------------

@dataclass
class TimeframeBuffers:
    """Container for multiple timeframe numpy arrays with circular buffer behavior.
    
    Manages a collection of fixed-size arrays for different timeframes,
    implementing a circular buffer pattern to efficiently store the most recent candles.
    """
    timeframes: list[str]
    capacity: int = 200  # Default buffer size for each timeframe

    def __post_init__(self):
        # Define the structured dtype for OHLCV data
        self.dtype = np.dtype([
            ('timestamp', 'i8'),    # int64 for millisecond timestamp
            ('open', 'f8'),         # float64 for price data
            ('high', 'f8'),
            ('low', 'f8'),
            ('close', 'f8'),
            ('volume', 'f8')
        ])
        
        # Initialize arrays and pointers for each timeframe
        self.arrays = {}
        self.pointers = {}
        self.is_filled = {}
        
        for tf in self.timeframes:
            self.arrays[tf] = np.zeros(self.capacity, dtype=self.dtype)
            self.pointers[tf] = 0
            self.is_filled[tf] = False
    
    def add_candle(self, timeframe: str, candle) -> None:
        """Add a new candle to the specified timeframe buffer.
        
        Implements circular buffer behavior - when the buffer is full,
        it wraps around and starts overwriting the oldest data.
        """
        if timeframe not in self.arrays:
            return
        
        # Add data at current pointer position
        self.arrays[timeframe][self.pointers[timeframe]] = (
            candle.timestamp,
            candle.open,
            candle.high,
            candle.low,
            candle.close,
            candle.volume
        )
        
        # Advance pointer
        self.pointers[timeframe] = (self.pointers[timeframe] + 1) % self.capacity
        if self.pointers[timeframe] == 0:
            self.is_filled[timeframe] = True
    
    def get_data(self, timeframe: str) -> np.ndarray:
        """Get the current data for a timeframe in chronological order.
        
        Handles the circular buffer logic to return data in the correct sequence,
        regardless of internal storage order.
        """
        if timeframe not in self.arrays:
            raise KeyError(f"Timeframe {timeframe} not found")
            
        if not self.is_filled[timeframe]:
            # If buffer hasn't wrapped yet, return only the filled portion
            return self.arrays[timeframe][:self.pointers[timeframe]]
        
        # If buffer has wrapped, reconstruct chronological order
        pointer = self.pointers[timeframe]
        return np.concatenate([
            self.arrays[timeframe][pointer:],
            self.arrays[timeframe][:pointer]
        ])

#------------------------------------------------------------------------------
# Indicator Management
#------------------------------------------------------------------------------

class IndicatorManager:
    """Manages indicators across multiple timeframes.
    
    Handles the creation, dependency resolution, and updating of technical indicators
    across different timeframes.
    """
    
    def __init__(self, buffers: TimeframeBuffers):
        self.buffers = buffers
        self.indicators = {}  # Dict[timeframe, List[Indicator]]
        
    def add_indicator(self, indicator: Indicator) -> None:
        """Add an indicator to the manager.
        
        Also recursively adds any dependencies that the indicator requires.
        """
        timeframe = indicator.timeframe
        if timeframe not in self.indicators:
            self.indicators[timeframe] = []
        self.indicators[timeframe].append(indicator)
        
        # Make sure any dependencies are also added
        for dep in indicator.dependencies:
            if dep.timeframe not in self.indicators or dep not in self.indicators[dep.timeframe]:
                self.add_indicator(dep)
    
    def update_indicators(self, timeframe: str, args=None) -> None:
        """Update all indicators for a specific timeframe.
        
        Fetches the latest data and updates each indicator's calculations.
        Handles errors gracefully with optional verbose logging.
        """
        if timeframe not in self.indicators:
            return
        
        try:
            # Get the latest data for this timeframe
            data = self.buffers.get_data(timeframe)
            
            if len(data) == 0:
                return
                
            # Update all indicators for this timeframe
            for indicator in self.indicators[timeframe]:
                try:
                    indicator.update(data)
                except Exception as e:
                    if args and args.verbose:
                        print(f"Error updating indicator {indicator}: {e}")
                        import traceback
                        print(traceback.format_exc())
        except Exception as e:
            if args and args.verbose:
                print(f"Error updating indicators for {timeframe}: {e}")
                import traceback
                print(traceback.format_exc())
            
    def get_indicator(self, name: str, timeframe: str) -> Optional[Indicator]:
        """Get an indicator by name and timeframe."""
        if timeframe not in self.indicators:
            return None
            
        for indicator in self.indicators[timeframe]:
            if indicator.name == name:
                return indicator
        return None
        
    def get_timeframe_indicators(self, timeframe: str) -> List[Indicator]:
        """Get all indicators for a specific timeframe."""
        return self.indicators.get(timeframe, [])
        
        
def setup_indicators(indicator_manager, indicator_specs: List[str], selected_timeframes: List[str], args) -> None:
    """Parse and create requested indicators.
    
    Handles complex indicator specifications including:
    - Basic indicators (SMA20)
    - Indicators with specific timeframes (SMA20:1h)
    - Indicators with timeframe filters (RSI14:>=15m)
    - Indicators with specific price sources (BB20,2:1h:close)
    
    Also handles error reporting and dependency resolution.
    """
    if not indicator_specs:
        return
        
    for ind_spec in indicator_specs:
        try:
            # Parse out the timeframe and source
            parts = ind_spec.split(':', 1)
            
            if len(parts) < 2:
                # No timeframe specified, create for all timeframes
                ind_def = parts[0]
                source = 'close'
                for tf in selected_timeframes:
                    try:
                        indicator = IndicatorRegistry.create_indicator(ind_def, tf, source, indicator_manager)
                    except Exception as e:
                        if args.verbose:
                            print(f"Error creating indicator {ind_def} for {tf}: {e}")
            else:
                ind_def = parts[0]
                tf_source = parts[1]
                
                # Handle source if present (after the last colon)
                if ':' in tf_source:
                    tf_expr, source = tf_source.rsplit(':', 1)
                else:
                    tf_expr = tf_source
                    source = 'close'
                
                # Handle timeframe expressions (>=1h, <=4h, etc.)
                if any(tf_expr.startswith(op) for op in ['>=', '<=', '>', '<', '=']):
                    # Find all matching timeframes
                    matching_timeframes = [
                        tf for tf in selected_timeframes 
                        if matches_timeframe_filter(tf, tf_expr)
                    ]
                    
                    if not matching_timeframes:
                        if args.verbose:
                            print(f"Warning: No timeframes match expression: {tf_expr}")
                        continue
                    
                    # Create indicators for all matching timeframes
                    for tf in matching_timeframes:
                        try:
                            indicator = IndicatorRegistry.create_indicator(ind_def, tf, source, indicator_manager)
                        except Exception as e:
                            if args.verbose:
                                print(f"Error creating indicator {ind_def} for {tf}: {e}")
                else:
                    # Direct timeframe specification
                    if tf_expr not in selected_timeframes:
                        if args.verbose:
                            print(f"Warning: Timeframe {tf_expr} not in selected timeframes")
                        continue
                    
                    try:
                        indicator = IndicatorRegistry.create_indicator(ind_def, tf_expr, source, indicator_manager)
                    except Exception as e:
                        if args.verbose:
                            print(f"Error creating indicator {ind_def} for {tf}: {e}")
                    
        except Exception as e:
            if args.verbose:
                print(f"Error processing indicator spec '{ind_spec}': {e}")
                import traceback
                print(traceback.format_exc())
    
    # Summary output in verbose mode only
    if args.verbose:
        print("Indicators setup complete:")
        for tf in selected_timeframes:
            indicators = indicator_manager.get_timeframe_indicators(tf)
            if indicators:
                print(f"  {tf}: {', '.join(str(ind) for ind in indicators)}")
            else:
                print(f"  {tf}: No indicators")

def display_results(buffers, indicator_manager, args=None):
    """Display the results of the analysis.
    
    Formats and prints:
    - Recent candles for each timeframe
    - Indicator values in either columnar or stacked format
    - Uses color formatting when available
    """
    # Sort timeframes for display
    sorted_timeframes = sorted(buffers.arrays.keys(), key=get_timeframe_sort_key)
    
    # Number of recent values to display
    num_recent = 3
    
    # Display results
    print(f"\n{Fore.CYAN}{Style.BRIGHT}Analysis Results:{Style.RESET_ALL}")
    print(f"{Fore.CYAN}================={Style.RESET_ALL}\n")
    
    for tf in sorted_timeframes:
        data = buffers.get_data(tf)
        if len(data) == 0:
            continue
            
        print(f"{Fore.GREEN}{Style.BRIGHT}Timeframe: {tf}{Style.RESET_ALL}")
        print(f"{Fore.GREEN}{'=' * (11 + len(tf))}{Style.RESET_ALL}")
        
        # Display last few candles
        print(f"\n{Fore.YELLOW}Last candles:{Style.RESET_ALL}")
        for i in range(max(0, len(data) - num_recent), len(data)):
            print(f"  {format_candle(i, data[i])}")
        
        # Display indicators with recent values
        indicators = indicator_manager.get_timeframe_indicators(tf)
        if indicators:
            print(f"\n{Fore.YELLOW}Indicators:{Style.RESET_ALL}")
            
            for ind in indicators:
                print(f"  {Fore.CYAN}{ind}{Style.RESET_ALL}:")
                
                # Get the last few values
                recent_indices = range(max(0, len(data) - num_recent), len(data))
                
                # Check if the indicator should be displayed in columnar mode
                if getattr(ind, 'display_mode', 'default') == 'columnar':
                    # First, get all the values and format them
                    all_formatted_values = []
                    for i in recent_indices:
                        try:
                            timestamp = data[i]['timestamp']
                            timestamp_str = time.strftime('%Y-%m-%d %H:%M', time.localtime(timestamp/1000))
                            source_value = data[i][ind.source]
                            ind_value = ind.get_value(i)
                            formatted_values = ind.format_value(ind_value, source_value, timestamp_str)
                            all_formatted_values.append((timestamp_str, source_value, formatted_values))
                        except Exception as e:
                            if args and args.verbose:
                                print(f"Error formatting value for index {i}: {e}")
                    
                    if all_formatted_values:
                        # Get the column headers from the first set of formatted values
                        headers = ['Timestamp', 'Source'] + [label for label, _ in all_formatted_values[0][2]]
                        
                        # Determine appropriate column widths
                        column_widths = []
                        # First two columns (timestamp and source)
                        column_widths.append(max(len(headers[0]), max(len(ts) for ts, _, _ in all_formatted_values)) + 2)
                        column_widths.append(max(len(headers[1]), 10) + 2)  # Source value width
                        
                        # Add widths for value columns
                        for i, header in enumerate(headers[2:], 2):
                            # Find the maximum length of values in this column
                            max_val_len = len(header)
                            for _, _, fvals in all_formatted_values:
                                if i-2 < len(fvals):  # Ensure index is valid
                                    _, val = fvals[i-2]
                                    max_val_len = max(max_val_len, len(val))
                            column_widths.append(max_val_len + 2)
                        
                        # Print the header
                        header_line = "    "
                        for i, header in enumerate(headers):
                            header_line += f"{header:<{column_widths[i]}}"
                        print(header_line)
                        
                        # Print the header separator
                        separator_line = "    "
                        for width in column_widths:
                            separator_line += f"{'-' * width}"
                        print(separator_line)
                        
                        # Print each row
                        for timestamp_str, source_value, formatted_values in all_formatted_values:
                            row_line = f"    {timestamp_str:<{column_widths[0]}}{source_value:<{column_widths[1]}.5f}"
                            for i, (_, value_str) in enumerate(formatted_values):
                                row_line += f"{value_str:<{column_widths[i+2]}}"
                            print(row_line)
                else:
                    # Default display mode (stacked)
                    # Display header with timestamps
                    print(f"    {'Timestamp':<20} {'Source':<10} {'Value':<15}")
                    print(f"    {'-'*20} {'-'*10} {'-'*15}")
                    
                    for i in recent_indices:
                        try:
                            # Get timestamp in readable format
                            timestamp = data[i]['timestamp']
                            timestamp_str = time.strftime('%Y-%m-%d %H:%M', time.localtime(timestamp/1000))
                            
                            # Get source value
                            source_value = data[i][ind.source]
                            
                            # Get indicator value for this index
                            try:
                                ind_value = ind.get_value(i)
                                
                                # Use the indicator's format_value method
                                formatted_values = ind.format_value(ind_value, source_value, timestamp_str)
                                
                                # If we have multiple formatted values (like for Bollinger Bands)
                                if len(formatted_values) > 1:
                                    # Print the timestamp and source only once
                                    print(f"    {timestamp_str:<20} {source_value:<10.5f}")
                                    # Then print each component with its label
                                    for label, value_str in formatted_values:
                                        print(f"      {label:<18} {'':<10} {value_str}")
                                else:
                                    # For simple indicators with a single value
                                    label, value_str = formatted_values[0]
                                    print(f"    {timestamp_str:<20} {source_value:<10.5f} {value_str}")
                                
                            except Exception as e:
                                if args and args.verbose:
                                    print(f"    {timestamp_str:<20} {source_value:<10.5f} ERROR: {e}")
                        except Exception as e:
                            if args and args.verbose:
                                print(f"    Error displaying row {i}: {e}")
                
                print()  # Add a blank line between indicators
        else:
            print(f"\n{Fore.YELLOW}No indicators found for this timeframe{Style.RESET_ALL}")

        print()  # Add a blank line between timeframes

def print_available_indicators():
    """Print documentation for all available indicators."""
    indicators = IndicatorRegistry.get_all_indicators()
    
    print(f"\n{Fore.CYAN}Available Indicators:{Style.RESET_ALL}\n")
    
    for prefix, ind_info in sorted(indicators.items()):
        print(f"{Fore.GREEN}{prefix}{Style.RESET_ALL} - {ind_info['description']}")
        print(f"  Format: {ind_info['format']}")
        
        if ind_info['examples']:
            print(f"  Examples:")
            for example in ind_info['examples']:
                print(f"    {example}")
            
        if ind_info['params']:
            print(f"  Parameters:")
            for param in ind_info['params']:
                print(f"    {param['name']}: {param['description']}")
                
        print()  # Add blank line between indicators

def main():
    start_time = time.time()
    base_candle_count = 0
    
    # Parse arguments
    parser = argparse.ArgumentParser(
        description="Analyze candle data across multiple timeframes"
    )
    parser.add_argument("--exchange", required=True)
    parser.add_argument("--ticker", required=True)
    parser.add_argument("--timeframe", required=True, help="Base timeframe (e.g. 1h)")
    parser.add_argument("--start", help="Start date (YYYY-MM-DD or YYYY-MM-DD HH:MM)")
    parser.add_argument("--end", help="End date (YYYY-MM-DD or YYYY-MM-DD HH:MM)")
    parser.add_argument("--analysis-tfs", nargs="+", default=["1m","5m","1h","1D"],
                    help="Timeframes or relational expressions (e.g. 1h 4h >=2h <=12h)")
    parser.add_argument("--data-dir", default="~/.corky",
                    help="Base directory for candle data")
    parser.add_argument("--buffer-size", type=int, default=200,
                    help="Size of the buffer for each timeframe")
    parser.add_argument("--verbose", "-v", action="store_true",
                    help="Print verbose output")
    parser.add_argument("--indicators", nargs="*", default=None,
                        help="Indicators to calculate (e.g. SMA20:4h BB20,2:1h)")
    parser.add_argument("--update-interval", type=int, default=1000,
                        help="Update indicators every N candles")
    args = parser.parse_args()
    
    # Register all indicators
    auto_register_indicators()
    
    # Show indicator help if --indicators is provided with no values
    if args.indicators is not None and len(args.indicators) == 0:
        print_available_indicators()
        sys.exit(0)
    
    try:
        # Get timeframes and filter them
        all_timeframes = get_available_timeframes()
        selected_timeframes = filter_timeframes(all_timeframes, args.analysis_tfs)
 
        if args.verbose:
            print(f"Selected timeframes: {', '.join(sorted(selected_timeframes, key=get_timeframe_sort_key))}")
    
        # Initialize buffer container
        buffers = TimeframeBuffers(timeframes=selected_timeframes, capacity=args.buffer_size)
        selected_timeframes_set = set(selected_timeframes)
        
        # Initialize indicator manager
        indicator_manager = IndicatorManager(buffers)
        
        # Setup indicators if requested
        if args.indicators:
            setup_indicators(indicator_manager, args.indicators, selected_timeframes, args)
        
        # Process candles in a single pass
        update_counter = 0
        for closure in analyze_candle_data(
            exchange=args.exchange,
            ticker=args.ticker,
            base_timeframe=args.timeframe,
            analysis_timeframes=selected_timeframes,
            start_date=args.start,
            end_date=args.end,
            data_dir=args.data_dir
        ):
            base_candle_count += 1
            update_counter += 1
            
            # Process only selected timeframes
            for tf in set(closure.timeframes) & selected_timeframes_set:
                buffers.add_candle(tf, closure.get_candle(tf))
            
            # Only update indicators periodically or at the end
            if update_counter >= args.update_interval:
                if args.verbose:
                    print(f"Updating indicators after {update_counter} candles...")
                for tf in selected_timeframes:
                    indicator_manager.update_indicators(tf, args)
                update_counter = 0
        
        # Final update of indicators
        if update_counter > 0:
            for tf in selected_timeframes:
                indicator_manager.update_indicators(tf, args)
                
        # Display the results
        display_results(buffers, indicator_manager, args)
        
        # Print execution summary
        print(f"\n{Fore.CYAN}{Style.BRIGHT}Execution Summary:{Style.RESET_ALL}")
        print(f"Base candles processed: {base_candle_count}")
        print(f"Total execution time: {time.time() - start_time:.2f} seconds")
            
    except ValueError as e:
        print(f"{Fore.RED}Error: {e}{Style.RESET_ALL}")
        sys.exit(1)
    except KeyboardInterrupt:
        print(f"\n{Fore.YELLOW}Operation cancelled by user{Style.RESET_ALL}")
        sys.exit(0)
    except Exception as e:
        print(f"{Fore.RED}Unexpected error: {e}{Style.RESET_ALL}")
        sys.exit(1)
    
if __name__ == "__main__":
    main()


