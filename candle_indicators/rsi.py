"""
Relative Strength Index implementation.
"""
import numpy as np
from typing import List, Optional, Union, Any, Dict, Tuple

from .base import Indicator, IndicatorRegistry


@IndicatorRegistry.register
class RSI(Indicator):
    """Relative Strength Index.
    
    RSI measures the magnitude of recent price changes to evaluate
    overbought or oversold conditions in the price of a stock or other asset.
    """
    
    # Class attribute for indicator registration
    PREFIX = "RSI"
    
    @classmethod
    def description(cls, pretty=False) -> str:
        """Returns a description of the RSI indicator.
        
        Args:
            pretty: If True, format the description with markdown for display
            
        Returns:
            A string describing the indicator
        """
        if pretty:
            from candle_indicators.utils.ui import COLOR_DESC, Style
            return f"{COLOR_DESC}Relative Strength Index{Style.RESET_ALL}"
        return "Relative Strength Index (RSI) measures the magnitude of recent price changes to evaluate overbought or oversold conditions."
    
    @classmethod
    def format_str(cls, pretty=False) -> str:
        """Returns a string describing the format of the RSI indicator command.
        
        Args:
            pretty: If True, format with markdown for display
            
        Returns:
            A string describing the command format
        """
        fmt =  "RSI<period>:<timeframe>[:<source>]"
        if pretty:
            from candle_indicators.utils.ui import COLOR_TYPE, Style
            return f"{COLOR_TYPE}{fmt}{Style.RESET_ALL}"
        return fmt
    
    @classmethod
    def examples(cls, pretty=False) -> List[str]:
        """Returns a list of example commands for the RSI indicator.
        
        Args:
            pretty: If True, format with markdown for display
            
        Returns:
            A list of example commands
        """
        if pretty:
            return [
                "RSI14:1h, RSI14:4h:high",
            ]
        else:
            return [
                "RSI14:1h, RSI14:4h:high",
            ]
    
    @classmethod
    def parameters(cls, pretty=False) -> List[Dict[str, Any]]:
        """Returns a list of parameters for the RSI indicator.
        
        Args:
            pretty: If True, format with markdown for display
            
        Returns:
            A list of parameter dictionaries
        """
        return [
            {
                'name': 'period',
                'description': 'Number of periods for RSI calculation',
                'required': True,
                'default': 14
            },
            {
                'name': 'source',
                'description': 'Price field to use',
                'required': False,
                'default': 'close',
                'options': ['open', 'high', 'low', 'close', 'volume']
            }
        ]
    
    @classmethod
    def params(cls, pretty=False) -> List[Dict[str, str]]:
        """Returns a list of parameter dictionaries for the RSI indicator.
        
        Args:
            pretty: If True, format with markdown for display
            
        Returns:
            A list of parameter dictionaries
        """
        params = [
            {"name": "period", "description": "Number of periods for RSI calculation (required)"},
            {"name": "source", "description": "Price field to use (optional, default: close)"}
        ]
        
        if pretty:
            from candle_indicators.utils.ui import COLOR_DESC, Style
            return [
                {"name": p["name"], "description": f"{COLOR_DESC}{p['description']}{Style.RESET_ALL}"}
                for p in params
            ]
        return params
    
    def __init__(self, timeframe: str, period: int = 14, source: str = 'close'):
        super().__init__(f"RSI{period}", timeframe, source)
        self.period = period
        
        # Initialize for incremental calculation
        # Preallocate fixed-size output buffer for RSI values
        buffer_capacity = max(200, period * 2)
        self.buffer = np.full(buffer_capacity, np.nan)
        self.position = 0
        self.is_filled = False
        self._buffer_count = 0
        self.values = np.array([])
        
        # For incremental calculation
        self._window_buffer = np.zeros(period + 1, dtype=float)  # +1 to calculate deltas
        self._window_pos = 0
        self._window_count = 0
        self._last_data_length = 0
        
        # Store the last gain and loss averages for incremental updates
        self._last_avg_gain = None
        self._last_avg_loss = None
        self._first_avg_calculated = False
    
    @classmethod
    def from_command(cls, ind_def: str, timeframe: str, source: str) -> 'RSI':
        """Create a new RSI from a command string."""
        
        # Extract period from command string
        period = int(ind_def.replace("RSI", ""))
        
        return cls(timeframe, period, source)
    
    def _calculate_full(self, data: np.ndarray) -> np.ndarray:
        """Calculate RSI values from scratch using all available data.
        
        Args:
            data: Price data array with length >= period+1
            
        Returns:
            Array of RSI values with the same length as data, with the first
            (period) values set to NaN
        """
        try:
            source_data = data[self.source]
        except (TypeError, IndexError, KeyError):
            # Handle case where data is just a raw array without field names
            source_data = data
        
        # Initialize result array with NaNs
        result = np.full(len(source_data), np.nan)
        
        # Need at least period+1 data points to calculate RSI
        if len(source_data) <= self.period:
            return result
            
        # Calculate price deltas (t - (t-1))
        deltas = np.zeros(len(source_data))
        deltas[1:] = np.diff(source_data)
        
        # Initialize arrays for gains and losses
        gains = np.zeros(len(deltas))
        losses = np.zeros(len(deltas))
        
        # Separate gains and losses
        gains[deltas >= 0] = deltas[deltas >= 0]
        losses[deltas < 0] = -deltas[deltas < 0]
        
        # Calculate average gains and losses
        avg_gain = np.zeros(len(deltas))
        avg_loss = np.zeros(len(deltas))
        
        # First average is a simple average
        if len(gains) >= self.period:
            avg_gain[self.period-1] = np.mean(gains[:self.period])
            avg_loss[self.period-1] = np.mean(losses[:self.period])
            
            # Use smoothed averages for subsequent values
            for i in range(self.period, len(deltas)):
                avg_gain[i] = (avg_gain[i-1] * (self.period-1) + gains[i]) / self.period
                avg_loss[i] = (avg_loss[i-1] * (self.period-1) + losses[i]) / self.period
                
            # Calculate RS and RSI
            rs = avg_gain / np.maximum(avg_loss, 1e-10)  # Add small value to prevent division by zero
            rsi = 100 - (100 / (1 + rs))
            
            # Copy RSI values to result array, adjusting for the shifted index due to diff
            result[self.period:] = rsi[self.period-1:]
                
        return result

    def _calculate_incremental(self, new_value: float) -> float:
        """Calculate RSI incrementally with a single new data point.
        
        This is a private method that efficiently updates the RSI using the ring buffer.
        
        Args:
            new_value: The new price value to add
            
        Returns:
            The calculated RSI value or NaN if not enough data
        """
        # Place the new value in the window buffer
        old_value = self._window_buffer[self._window_pos]
        self._window_buffer[self._window_pos] = new_value
        self._window_pos = (self._window_pos + 1) % len(self._window_buffer)
        
        # Track how many values we've seen
        if self._window_count < len(self._window_buffer):
            self._window_count += 1
            
        # Need at least 2 values to calculate delta
        if self._window_count < 2:
            return np.nan
            
        # Get the previous value (considering circular buffer)
        prev_pos = (self._window_pos - 2) % len(self._window_buffer)
        prev_value = self._window_buffer[prev_pos]
        
        # Calculate delta
        delta = new_value - prev_value
        
        # Calculate gain and loss
        gain = max(0, delta)
        loss = max(0, -delta)
        
        # If we don't have enough data for a full calculation, return NaN
        if self._window_count < self.period + 1:
            return np.nan
            
        # If this is our first full calculation, compute simple averages
        if not self._first_avg_calculated:
            # Collect all deltas, gains, and losses from the window buffer
            deltas = []
            gains = []
            losses = []
            
            # Iterate through buffer in chronological order, skipping the oldest value
            for i in range(1, self.period + 1):
                idx = (self._window_pos - i) % len(self._window_buffer)
                prev_idx = (self._window_pos - i - 1) % len(self._window_buffer)
                
                val = self._window_buffer[idx]
                prev_val = self._window_buffer[prev_idx]
                
                if not np.isnan(val) and not np.isnan(prev_val):
                    d = val - prev_val
                    deltas.append(d)
                    gains.append(max(0, d))
                    losses.append(max(0, -d))
            
            # Calculate simple averages
            self._last_avg_gain = np.mean(gains)
            self._last_avg_loss = np.mean(losses)
            self._first_avg_calculated = True
        else:
            # Calculate smoothed averages
            self._last_avg_gain = (self._last_avg_gain * (self.period - 1) + gain) / self.period
            self._last_avg_loss = (self._last_avg_loss * (self.period - 1) + loss) / self.period
        
        # Calculate RSI
        if self._last_avg_loss == 0:
            rsi = 100.0
        else:
            rs = self._last_avg_gain / self._last_avg_loss
            rsi = 100.0 - (100.0 / (1.0 + rs))
        
        # Write into the final output buffer
        self.buffer[self.position] = rsi
        self.position = (self.position + 1) % len(self.buffer)
        
        # Update the buffer count
        if self._buffer_count < len(self.buffer):
            self._buffer_count += 1
            if self._buffer_count == len(self.buffer):
                self.is_filled = True
        
        return rsi

    def update_with_data(self, data: np.ndarray) -> None:
        """Update indicator values with new data.
        
        This is the primary public method for updating the RSI with new data.
        
        Args:
            data: Price data array containing the source data field
        """
        # Extract the source data
        try:
            source_data = data[self.source]
        except (TypeError, IndexError, KeyError):
            # Handle case where data is just a raw array without field names
            source_data = data
        
        # Case 1: Single new data point - process incrementally
        if len(source_data) == 1:
            new_val = source_data[0]
            rsi_value = self._calculate_incremental(new_val)
            if self.values is None or len(self.values) == 0:
                self.values = np.array([rsi_value])
            else:
                self.values = np.append(self.values, rsi_value)
            self._last_data_length = self._last_data_length + 1
            return
            
        # Case 2: First calculation or full recalculation with multiple points
        if len(source_data) >= self.period + 1:  # Need period+1 for RSI
            if self._last_data_length == 0:
                # First calculation
                self.values = self._calculate_full(data)
                self._last_data_length = len(data)
            elif len(data) == self._last_data_length + 1:
                # One new data point in a structured array
                new_val = source_data[-1]
                rsi_value = self._calculate_incremental(new_val)
                if self.values is None or len(self.values) == 0:
                    self.values = np.array([rsi_value])
                else:
                    self.values = np.append(self.values, rsi_value)
                self._last_data_length += 1
            else:
                # Data has changed significantly - recalculate everything
                self.values = self._calculate_full(data)
                self._last_data_length = len(data)
                
                # Reset incremental calculation state
                self._first_avg_calculated = False
                self._last_avg_gain = None
                self._last_avg_loss = None
        else:
            # Not enough data yet, but still process each value incrementally
            for val in source_data:
                self._calculate_incremental(val)
            self.values = np.array([])
            self._last_data_length = len(source_data)
