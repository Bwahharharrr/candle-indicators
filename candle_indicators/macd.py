"""
Moving Average Convergence Divergence (MACD) implementation.
"""
import numpy as np
from typing import List, Optional, Union, Any, Dict, Tuple

from .base import Indicator, IndicatorRegistry
from .ema import ExponentialMovingAverage


@IndicatorRegistry.register
class MACD(Indicator):
    """Moving Average Convergence Divergence.
    
    MACD is calculated by subtracting the long-term EMA (26 periods) from
    the short-term EMA (12 periods). The signal line is typically a 9-period EMA
    of the MACD line.
    """
    
    # Class attribute for indicator registration
    PREFIX = "MACD"
    
    @classmethod
    def description(cls, pretty=False) -> str:
        """Returns a description of the MACD indicator.
        
        Args:
            pretty: If True, format the description with markdown for display
            
        Returns:
            A string describing the indicator
        """

        if pretty:
            from candle_indicators.utils.ui import COLOR_DESC, Style
            return f"{COLOR_DESC}Moving Average Convergence Divergence{Style.RESET_ALL}"
        return "MACD is a trend-following momentum indicator that shows the relationship between two moving averages of a security's price."
    
    @classmethod
    def format_str(cls, pretty=False) -> str:
        """Returns a string describing the format of the MACD indicator command.
        
        Args:
            pretty: If True, format with markdown for display
            
        Returns:
            A string describing the command format
        """
        fmt = "MACD<fast_period>,<slow_period>,<signal_period>:<timeframe>[:<source>]"
        if pretty:
            from candle_indicators.utils.ui import COLOR_TYPE, Style
            return f"{COLOR_TYPE}{fmt}{Style.RESET_ALL}"
        return fmt
    
    @classmethod
    def examples(cls, pretty=False) -> List[str]:
        """Returns a list of example commands for the MACD indicator.
        
        Args:
            pretty: If True, format with markdown for display
            
        Returns:
            A list of example commands
        """
        if pretty:
            return [
                "MACD12,26,9:1h, MACD12,26,9:1D:high"
            ]
        else:
            return [
                "MACD12,26,9:1h, MACD12,26,9:1D:high"
            ]
    
    @classmethod
    def parameters(cls, pretty=False) -> List[Dict[str, Any]]:
        """Returns a list of parameters for the MACD indicator.
        
        Args:
            pretty: If True, format with markdown for display
            
        Returns:
            A list of parameter dictionaries
        """
        return [
            {
                'name': 'fast_period',
                'description': 'Number of periods for fast EMA',
                'required': False,
                'default': 12
            },
            {
                'name': 'slow_period',
                'description': 'Number of periods for slow EMA',
                'required': False,
                'default': 26
            },
            {
                'name': 'signal_period',
                'description': 'Number of periods for signal line EMA',
                'required': False,
                'default': 9
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
        """Returns a list of parameter dictionaries for the MACD indicator.
        
        Args:
            pretty: If True, format with markdown for display
            
        Returns:
            A list of parameter dictionaries
        """
        params = [
            {"name": "fast_period", "description": "Number of periods for fast EMA (optional, default: 12)"},
            {"name": "slow_period", "description": "Number of periods for slow EMA (optional, default: 26)"},
            {"name": "signal_period", "description": "Number of periods for signal line EMA (optional, default: 9)"},
            {"name": "source", "description": "Price field to use (optional, default: close)"}
        ]
        
        if pretty:
            from candle_indicators.utils.ui import COLOR_DESC, Style
            return [
                {"name": p["name"], "description": f"{COLOR_DESC}{p['description']}{Style.RESET_ALL}"}
                for p in params
            ]
        return params
    
    def __init__(self, timeframe: str, fast_period: int = 12, slow_period: int = 26,
                 signal_period: int = 9, source: str = 'close',
                 fast_ma: Optional[Indicator] = None, 
                 slow_ma: Optional[Indicator] = None):
        # Format name
        name = f"MACD{fast_period},{slow_period},{signal_period}"
        if source != 'close':
            name = f"{name}[{source}]"
            
        super().__init__(name, timeframe, source)
        
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.signal_period = signal_period
        
        # Use provided MAs or create new ones
        if fast_ma is None:
            self.fast_ma = ExponentialMovingAverage(timeframe, fast_period, source)
        else:
            self.fast_ma = fast_ma
            # Register dependency to ensure it gets updated
            self.add_dependency(fast_ma)
            
        if slow_ma is None:
            self.slow_ma = ExponentialMovingAverage(timeframe, slow_period, source)
        else:
            self.slow_ma = slow_ma
            # Register dependency to ensure it gets updated
            self.add_dependency(slow_ma)
            
        # Initialize for incremental calculation
        # Preallocate fixed-size output buffer for MACD values
        buffer_capacity = max(200, slow_period * 2)
        self.buffer = np.full(buffer_capacity, np.nan, 
                             dtype=[('macd', 'f8'), ('signal', 'f8'), ('histogram', 'f8')])
        self.position = 0
        self.is_filled = False
        self._buffer_count = 0
        
        # Create structured array to hold results
        self.values = np.array([], 
                             dtype=[('macd', 'f8'), ('signal', 'f8'), ('histogram', 'f8')])
        
        # For signal line calculation
        self._signal_window = np.zeros(signal_period, dtype=float)
        self._signal_pos = 0
        self._signal_count = 0
        self._last_signal = np.nan
        
        # For full state tracking
        self._macd_history = np.zeros(signal_period, dtype=float)
        self._macd_pos = 0
        self._first_signal_calculated = False
        self._last_data_length = 0
    
    @classmethod
    def from_command(cls, ind_def: str, timeframe: str, source: str) -> 'MACD':
        """Create a new MACD from a command string."""
        
        # Format: MACD12,26,9
        # Extract parameters from command string
        params_part = ind_def.replace("MACD", "")
        
        # Handle case with or without parameters
        if params_part:
            # Split parameters
            params = params_part.split(",")
            
            # Extract periods
            fast_period = int(params[0])
            slow_period = int(params[1]) if len(params) > 1 else 26
            signal_period = int(params[2]) if len(params) > 2 else 9
            
            return cls(timeframe, fast_period, slow_period, signal_period, source)
        else:
            # Use default parameters
            return cls(timeframe, source=source)
    
    def _calculate_full(self, data: np.ndarray) -> np.ndarray:
        """Calculate MACD values from scratch using all available data.
        
        Args:
            data: Price data array
            
        Returns:
            Structured array with macd, signal, and histogram components
        """
        try:
            source_data = data[self.source]
        except (TypeError, IndexError, KeyError):
            # Handle case where data is just a raw array without field names
            source_data = data
        
        # Make sure the dependencies are updated
        self.fast_ma.update_with_data(data)
        self.slow_ma.update_with_data(data)
        
        # Create result array
        result = np.zeros(len(source_data), 
                         dtype=[('macd', 'f8'), ('signal', 'f8'), ('histogram', 'f8')])
        
        # Fill with NaN values initially
        result['macd'] = np.nan
        result['signal'] = np.nan
        result['histogram'] = np.nan
        
        # Calculate MACD = fast_ma - slow_ma
        # Get values from the dependencies
        fast_values = self.fast_ma.values
        slow_values = self.slow_ma.values
        
        # MACD is only valid where both fast and slow MA values are available
        valid_indices = np.where(
            (~np.isnan(fast_values)) & 
            (~np.isnan(slow_values))
        )[0]
        
        if len(valid_indices) > 0:
            # Calculate MACD line
            macd_line = fast_values[valid_indices] - slow_values[valid_indices]
            result['macd'][valid_indices] = macd_line
            
            # Calculate signal line (EMA of MACD line)
            if len(macd_line) >= self.signal_period:
                # Calculate EMA of MACD line as signal line
                signal_line = np.zeros_like(macd_line)
                
                # First value is SMA
                signal_line[:self.signal_period] = np.nan
                signal_line[self.signal_period-1] = np.mean(macd_line[:self.signal_period])
                
                # Calculate EMA for remaining values
                alpha = 2 / (self.signal_period + 1)
                for i in range(self.signal_period, len(macd_line)):
                    signal_line[i] = macd_line[i] * alpha + signal_line[i-1] * (1 - alpha)
                
                # Store signal line values
                result['signal'][valid_indices] = signal_line
                
                # Calculate histogram (MACD - signal)
                valid_signal_indices = np.where(~np.isnan(signal_line))[0]
                if len(valid_signal_indices) > 0:
                    hist_indices = valid_indices[valid_signal_indices]
                    result['histogram'][hist_indices] = (
                        result['macd'][hist_indices] - result['signal'][hist_indices]
                    )
                
        return result
    
    def _calculate_incremental(self, new_price: float) -> np.void:
        """Calculate MACD incrementally with a single new data point.
        
        This is a private method that efficiently updates the MACD using the ring buffer.
        
        Args:
            new_price: The new price value to add
            
        Returns:
            A structured array with macd, signal, and histogram fields
        """
        # The MAs should already be updated with this price point
        # Get the new values from each MA
        new_fast = self.fast_ma.get_value(0)
        new_slow = self.slow_ma.get_value(0)
        
        # If either MA doesn't have a value yet, return NaN result
        if new_fast is None or np.isnan(new_fast) or new_slow is None or np.isnan(new_slow):
            return np.array([(np.nan, np.nan, np.nan)], 
                           dtype=[('macd', 'f8'), ('signal', 'f8'), ('histogram', 'f8')])[0]
        
        # Calculate new MACD value
        new_macd = new_fast - new_slow
        
        # Store in history for signal line calculation
        self._macd_history[self._macd_pos] = new_macd
        self._macd_pos = (self._macd_pos + 1) % len(self._macd_history)
        
        # Calculate signal line
        signal_value = np.nan
        histogram_value = np.nan
        
        # If we don't have enough MACD values for a signal line yet
        if self._signal_count < self.signal_period:
            self._signal_count += 1
        
        # Once we have enough values, calculate the signal line
        if self._signal_count >= self.signal_period:
            if not self._first_signal_calculated:
                # Calculate first signal as SMA of MACD values
                signal_value = np.mean(self._macd_history)
                self._last_signal = signal_value
                self._first_signal_calculated = True
            else:
                # Use EMA formula for subsequent signal values
                multiplier = 2 / (self.signal_period + 1)
                signal_value = (new_macd - self._last_signal) * multiplier + self._last_signal
                self._last_signal = signal_value
                
            # Calculate histogram
            histogram_value = new_macd - signal_value
        
        # Create result structure
        macd_values = np.array([(new_macd, signal_value, histogram_value)], 
                              dtype=[('macd', 'f8'), ('signal', 'f8'), ('histogram', 'f8')])[0]
        
        # Write into the final output buffer
        self.buffer[self.position] = macd_values
        self.position = (self.position + 1) % len(self.buffer)
        
        # Update the buffer count
        if self._buffer_count < len(self.buffer):
            self._buffer_count += 1
            if self._buffer_count == len(self.buffer):
                self.is_filled = True
                
        return macd_values
    
    def update_with_data(self, data: np.ndarray) -> None:
        """Update indicator values with new data.
        
        This is the primary public method for updating the MACD with new data.
        It ensures all dependencies are updated first and then recalculates.
        
        Args:
            data: Price data array containing the source data field
        """
        # Extract the source data
        try:
            source_data = data[self.source]
        except (TypeError, IndexError, KeyError):
            # Handle case where data is just a raw array without field names
            source_data = data
            
        # Make sure dependencies are updated
        self.fast_ma.update_with_data(data)
        self.slow_ma.update_with_data(data)
        
        # Case 1: Single new data point - process incrementally
        if len(source_data) == 1:
            new_val = source_data[0]
            macd_values = self._calculate_incremental(new_val)
            if self.values is None or len(self.values) == 0:
                self.values = np.array([macd_values])
            else:
                self.values = np.append(self.values, [macd_values])
            self._last_data_length = self._last_data_length + 1
            return
            
        # Case 2: First calculation or full recalculation with multiple points
        if len(source_data) >= self.slow_period:
            if self._last_data_length == 0:
                # First calculation
                self.values = self._calculate_full(data)
                self._last_data_length = len(data)
            elif len(data) == self._last_data_length + 1:
                # One new data point in a structured array
                new_val = source_data[-1]
                macd_values = self._calculate_incremental(new_val)
                if self.values is None or len(self.values) == 0:
                    self.values = np.array([macd_values])
                else:
                    self.values = np.append(self.values, [macd_values])
                self._last_data_length += 1
            else:
                # Data has changed significantly - recalculate everything
                self.values = self._calculate_full(data)
                self._last_data_length = len(data)
                
                # Reset incremental calculation state
                self._first_signal_calculated = False
                self._signal_count = 0
                self._last_signal = np.nan
        else:
            # Not enough data yet, but still process each value incrementally
            for val in source_data:
                self._calculate_incremental(val)
            self.values = np.array([])
            self._last_data_length = len(source_data)
