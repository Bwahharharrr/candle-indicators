import numpy as np
from typing import List, Optional, Tuple, Union
from .base import Indicator, IndicatorRegistry, indicator_method
from .sma  import SimpleMovingAverage
import inspect

@IndicatorRegistry.register
class BollingerBands(Indicator):
    """Bollinger Bands indicator."""
    
    PREFIX = "BB"
    
    # Define the components this indicator provides
    # For structured arrays, we use None as the value
    COMPONENTS = {
        'upper': None,     # Structured array field
        'middle': None,    # Structured array field
        'lower': None,     # Structured array field
        'bandwidth': None, # Structured array field
        'percent_b': None  # Structured array field
    }
    
    def __init__(self, timeframe: str, period: int = 20, deviations: float = 2.0, 
                 source: str = 'close', base_ma: Optional[Indicator] = None):
        # Format the name to remove .0 suffix for integer values of deviations
        # This is the default naming that may be overridden by from_command
        if isinstance(deviations, int) or (isinstance(deviations, float) and deviations.is_integer()):
            name = f"BB{period},{int(deviations)}"
        else:
            name = f"BB{period},{deviations}"
            
        # Append source if it's not the default 'close'
        if source != 'close':
            name = f"{name}[{source}]"
            
        super().__init__(name, timeframe, source)
        self.period = period
        self.deviations = deviations  # Store exactly as provided, without type conversion
        self.display_mode = "columnar"  # Set display mode to columnar
        
        # Use provided MA or create a new SMA
        if base_ma is None:
            self.base_ma = SimpleMovingAverage(timeframe, period, source)
        else:
            self.base_ma = base_ma
            # Register dependency to ensure it gets updated
            self.add_dependency(base_ma)
            
        # Initialize for incremental calculation
        # Preallocate fixed-size output buffer for Bollinger Bands values
        buffer_capacity = max(200, period * 2)
        self.buffer = np.full(buffer_capacity, np.nan, 
                              dtype=[('upper', float), ('middle', float), ('lower', float), 
                                     ('bandwidth', float), ('percent_b', float)])
        self.position = 0
        self.is_filled = False
        self._buffer_count = 0
        self.values = np.array([], 
                              dtype=[('upper', float), ('middle', float), ('lower', float), 
                                     ('bandwidth', float), ('percent_b', float)])
                                     
        # Window buffer for standard deviation calculation
        self._window_buffer = np.zeros(period, dtype=float)
        self._window_pos = 0
        self._window_count = 0
        self._last_data_length = 0
   
    # Convenient component access methods
    @indicator_method(
        description="Get the upper Bollinger Band value(s)",
        returns="Upper band value (float) or series (list) depending on if length is provided"
    )
    def upper(self, offset: int = 0, length: int = None) -> Union[Optional[float], List[Optional[float]]]:
        """Get the upper band value(s).
        
        Args:
            offset: Offset from the most recent candle. 0 is current, -1 is previous, etc.
            length: If provided, returns a series of values starting at offset with the specified length
            
        Returns:
            Either a single value or a list of values depending on whether length is provided
        """
        if length is None:
            return self.get('upper', offset)
        else:
            result = []
            for i in range(offset, offset + length):
                result.append(self.get('upper', i))
            return result
    
    @indicator_method(
        description="Get the middle Bollinger Band value(s) (simple moving average)",
        returns="Middle band value (float) or series (list) depending on if length is provided"
    )
    def middle(self, offset: int = 0, length: int = None) -> Union[Optional[float], List[Optional[float]]]:
        """Get the middle band value(s).
        
        Args:
            offset: Offset from the most recent candle. 0 is current, -1 is previous, etc.
            length: If provided, returns a series of values starting at offset with the specified length
            
        Returns:
            Either a single value or a list of values depending on whether length is provided
        """
        if length is None:
            return self.get('middle', offset)
        else:
            result = []
            for i in range(offset, offset + length):
                result.append(self.get('middle', i))
            return result
    
    @indicator_method(
        description="Get the lower Bollinger Band value(s)",
        returns="Lower band value (float) or series (list) depending on if length is provided"
    )
    def lower(self, offset: int = 0, length: int = None) -> Union[Optional[float], List[Optional[float]]]:
        """Get the lower band value(s).
        
        Args:
            offset: Offset from the most recent candle. 0 is current, -1 is previous, etc.
            length: If provided, returns a series of values starting at offset with the specified length
            
        Returns:
            Either a single value or a list of values depending on whether length is provided
        """
        if length is None:
            return self.get('lower', offset)
        else:
            result = []
            for i in range(offset, offset + length):
                result.append(self.get('lower', i))
            return result
    
    @indicator_method(
        description="Get the bandwidth value(s)",
        returns="Bandwidth value (float) or series (list) depending on if length is provided"
    )
    def bandwidth(self, offset: int = 0, length: int = None) -> Union[Optional[float], List[Optional[float]]]:
        """Get the bandwidth value(s).
        
        Args:
            offset: Offset from the most recent candle. 0 is current, -1 is previous, etc.
            length: If provided, returns a series of values starting at offset with the specified length
            
        Returns:
            Either a single value or a list of values depending on whether length is provided
        """
        if length is None:
            return self.get('bandwidth', offset)
        else:
            result = []
            for i in range(offset, offset + length):
                result.append(self.get('bandwidth', i))
            return result
    
    @indicator_method(
        description="Get the percent B value(s)",
        returns="Percent B value (float) or series (list) depending on if length is provided"
    )
    def percent_b(self, offset: int = 0, length: int = None) -> Union[Optional[float], List[Optional[float]]]:
        """Get the percent B value(s).
        
        Args:
            offset: Offset from the most recent candle. 0 is current, -1 is previous, etc.
            length: If provided, returns a series of values starting at offset with the specified length
            
        Returns:
            Either a single value or a list of values depending on whether length is provided
        """
        if length is None:
            return self.get('percent_b', offset)
        else:
            result = []
            for i in range(offset, offset + length):
                result.append(self.get('percent_b', i))
            return result
    
    @classmethod
    def from_command(cls, ind_def: str, timeframe: str, source: str) -> 'BollingerBands':
        """Create a BollingerBands instance from command line definition."""
        prefix_len = len(cls.PREFIX)
        parts = ind_def.split(':')
        ind_part = parts[0]
        param_str = ind_part[prefix_len:]
        
        # Default values
        period = 20
        deviations = 2.0
        
        # Parse parameters
        if param_str:
            # Format: BB<period>[,<deviations>]
            if ',' in param_str:
                # Both period and deviations specified
                period_str, deviations_str = param_str.split(',', 1)
                period = int(period_str) if period_str else 20
                deviations = float(deviations_str) if deviations_str else 2.0
            else:
                # Only period specified
                period = int(param_str)
                
        return cls(timeframe, period, deviations, source)
    
    @classmethod
    def description(cls, pretty=False) -> str:
        desc = "Bollinger Bands volatility indicator"
        if pretty:
            from candle_indicators.utils.ui import COLOR_DESC, Style
            return f"{COLOR_DESC}{desc}{Style.RESET_ALL}"
        return desc
    
    @classmethod
    def format_str(cls, pretty=False) -> str:
        fmt = "BB<period>[,<deviations>]:<timeframe>[:<source>]"
        if pretty:
            from candle_indicators.utils.ui import COLOR_TYPE, Style
            return f"{COLOR_TYPE}{fmt}{Style.RESET_ALL}"
        return fmt
    
    @classmethod
    def examples(cls, pretty=False) -> List[str]:
        examples = [
            "BB20:1h, BB20,2.5:1h, BB20:1h,high",
        ]
        return examples
    
    @classmethod
    def params(cls, pretty=False) -> List[dict]:
        params = [
            {"name": "period", "description": "Number of candles for moving average and standard deviation (optional, default: 20)"},
            {"name": "deviations", "description": "Number of standard deviations (optional, default: 2.0)"},
            {"name": "source", "description": "Price field to use (optional, default: close)"}
        ]
        
        if pretty:
            from candle_indicators.utils.ui import COLOR_DESC, Style
            return [
                {"name": p["name"], "description": f"{COLOR_DESC}{p['description']}{Style.RESET_ALL}"}
                for p in params
            ]
        return params
    
    def _calculate_incremental(self, new_value: float) -> np.void:
        """Calculate Bollinger Bands incrementally with a single new data point.
        
        This is a private method that efficiently updates the indicator using the ring buffer.
        
        Args:
            new_value: The new price value to add
            
        Returns:
            A structured array with upper, middle, lower, bandwidth, and percent_b fields
        """
        # Get the new SMA value (middle band) from the base MA
        # The base_ma will have already been updated with the new value
        middle_val = self.base_ma.get_value(0)
        
        # If the base MA doesn't have enough data yet, we can't calculate BB
        if middle_val is None or np.isnan(middle_val):
            return np.array([np.nan, np.nan, np.nan, np.nan, np.nan], 
                          dtype=[('upper', float), ('middle', float), ('lower', float), 
                                 ('bandwidth', float), ('percent_b', float)])[0]
        
        # Update the window buffer for standard deviation calculation
        old_value = self._window_buffer[self._window_pos]
        self._window_buffer[self._window_pos] = new_value
        self._window_pos = (self._window_pos + 1) % self.period
        
        if self._window_count < self.period:
            self._window_count += 1
            
        # Only calculate BB if we have enough data
        if self._window_count == self.period:
            # Calculate standard deviation efficiently
            # We already know the mean (middle_val), so we can use this formula:
            # std = sqrt(sum((x - mean)²) / n)
            squared_diffs = (self._window_buffer - middle_val) ** 2
            std_dev = np.sqrt(np.mean(squared_diffs))
            
            # Calculate BB components
            upper_val = middle_val + (self.deviations * std_dev)
            lower_val = middle_val - (self.deviations * std_dev)
            
            # Calculate bandwidth
            bandwidth_val = (upper_val - lower_val) / middle_val if middle_val != 0 else np.nan
            
            # Calculate %B
            if upper_val != lower_val:
                percent_b_val = (new_value - lower_val) / (upper_val - lower_val)
            else:
                percent_b_val = np.nan
                
            # Create a structured array for the BB values
            bb_values = np.array([(upper_val, middle_val, lower_val, bandwidth_val, percent_b_val)], 
                               dtype=[('upper', float), ('middle', float), ('lower', float), 
                                      ('bandwidth', float), ('percent_b', float)])[0]
            
            # Write into the final output buffer
            self.buffer[self.position] = bb_values
            self.position = (self.position + 1) % len(self.buffer)
            
            # Update the buffer count
            if self._buffer_count < len(self.buffer):
                self._buffer_count += 1
                if self._buffer_count == len(self.buffer):
                    self.is_filled = True
                    
            return bb_values
        else:
            # Return NaN values until we have enough data
            return np.array([np.nan, np.nan, np.nan, np.nan, np.nan], 
                          dtype=[('upper', float), ('middle', float), ('lower', float), 
                                 ('bandwidth', float), ('percent_b', float)])[0]
    
    def _calculate_full(self, data: np.ndarray) -> np.ndarray:
        """Calculate Bollinger Bands over the entire dataset.
        
        This is a private method that performs a full calculation from scratch.
        Users should call update_with_data() instead which manages dependencies 
        and determines when to recalculate.
        
        Args:
            data: Price data array containing OHLC data
            
        Returns:
            Structured numpy array with upper, middle, lower, bandwidth, and percent_b fields
        """
        # First make sure the base MA is calculated
        self.base_ma.update_with_data(data)
        middle_band = self.base_ma.values
        
        # If there's not enough data, return empty result
        if len(data) < self.period or middle_band is None or len(middle_band) == 0:
            # Create a structured array for the result
            bb_dtype = np.dtype([
                ('upper', 'f8'),
                ('middle', 'f8'),
                ('lower', 'f8'),
                ('bandwidth', 'f8'),
                ('percent_b', 'f8')
            ])
            
            # Just need a 1-element empty result
            return np.array([], dtype=bb_dtype)
            
        # Create a structured array for the result
        result = np.zeros(len(data), dtype=np.dtype([
            ('upper', 'f8'),
            ('middle', 'f8'),
            ('lower', 'f8'),
            ('bandwidth', 'f8'),
            ('percent_b', 'f8')
        ]))
        
        # Initialize all fields with NaN values
        result['upper'][:] = np.nan
        result['middle'][:] = np.nan
        result['lower'][:] = np.nan
        result['bandwidth'][:] = np.nan
        result['percent_b'][:] = np.nan
        
        # Copy middle band to result
        result['middle'] = middle_band
        
        # Calculate standard deviation
        source_data = data[self.source]
        # Need a rolling window standard deviation
        stds = np.zeros(len(data))
        stds[:] = np.nan
        
        # Calculate rolling standard deviation
        for i in range(self.period-1, len(data)):
            window = source_data[i-(self.period-1):i+1]
            stds[i] = np.std(window, ddof=1)  # Use sample standard deviation
        
        # Calculate upper and lower bands
        valid_indices = ~np.isnan(middle_band) & ~np.isnan(stds)
        result['upper'][valid_indices] = middle_band[valid_indices] + self.deviations * stds[valid_indices]
        result['lower'][valid_indices] = middle_band[valid_indices] - self.deviations * stds[valid_indices]
        
        # Calculate bandwidth and %B
        bandwidth = (result['upper'] - result['lower']) / result['middle']
        result['bandwidth'][valid_indices] = bandwidth[valid_indices]
        
        # Calculate %B (position within the bands)
        # Formula: (price - lower) / (upper - lower)
        denom = result['upper'] - result['lower']
        percent_b = np.zeros(len(data))
        percent_b[:] = np.nan
        
        valid_denom = (denom > 0) & valid_indices
        percent_b[valid_denom] = (source_data[valid_denom] - result['lower'][valid_denom]) / denom[valid_denom]
        result['percent_b'] = percent_b
        
        return result
    
    def update_with_data(self, data: np.ndarray) -> None:
        """Update Bollinger Bands with new data.
        
        This is the primary public method for updating the indicator with new data.
        It ensures all dependencies are updated first and then recalculates the bands.
        
        Args:
            data: Price data array containing OHLC data
        """
        # Extract the source data
        try:
            source_data = data[self.source]
        except (TypeError, IndexError, KeyError):
            # Handle case where data is just a raw array without field names
            source_data = data
        
        # Make sure the base MA is updated
        self.base_ma.update_with_data(data)
        
        # Case 1: Single new data point - process incrementally
        if len(source_data) == 1:
            new_val = source_data[0]
            bb_values = self._calculate_incremental(new_val)
            if self.values is None or len(self.values) == 0:
                self.values = np.array([bb_values])
            else:
                self.values = np.append(self.values, [bb_values])
            self._last_data_length = self._last_data_length + 1
            return
            
        # Case 2: First calculation or full recalculation with multiple points
        if len(source_data) >= self.period:
            if self._last_data_length == 0:
                # First calculation
                self.values = self._calculate_full(data)
                self._last_data_length = len(data)
            elif len(data) == self._last_data_length + 1:
                # One new data point in a structured array
                new_val = source_data[-1]
                bb_values = self._calculate_incremental(new_val)
                if self.values is None or len(self.values) == 0:
                    self.values = np.array([bb_values])
                else:
                    self.values = np.append(self.values, [bb_values])
                self._last_data_length += 1
            else:
                # Data has changed significantly - recalculate everything
                self.values = self._calculate_full(data)
                self._last_data_length = len(data)
        else:
            # Not enough data yet, but still process each value incrementally
            for val in source_data:
                self._calculate_incremental(val)
            self.values = np.array([])
            self._last_data_length = len(source_data)
    
    def format_value(self, value, source_value=None, timestamp_str=None) -> List[Tuple[str, str]]:
        """Format Bollinger Bands value for display with descriptive labels.
        
        Args:
            value: The Bollinger Bands structured array or tuple
            source_value: The source value used for calculation
            timestamp_str: The timestamp string for this value
            
        Returns:
            List of (label, formatted_value) tuples with descriptive labels
        """
        if value is None:
            return [("", "N/A")]
            
        result = []
        
        # Handle numpy.void or structured array with dtype.names
        if hasattr(value, 'dtype') and hasattr(value.dtype, 'names') and value.dtype.names:
            # Extract values from structured array or numpy.void
            for field in value.dtype.names:
                field_value = value[field]
                if np.isnan(field_value):
                    value_str = "NaN"
                else:
                    value_str = f"{field_value:.6f}"
                    
                # Use descriptive labels
                if field == 'upper':
                    label = "Upper Band"
                elif field == 'middle':
                    label = "Middle Band"
                elif field == 'lower':
                    label = "Lower Band"
                elif field == 'bandwidth':
                    label = "Bandwidth"
                elif field == 'percent_b':
                    label = "%B"
                else:
                    label = field
                    
                result.append((label, value_str))
                
        # Handle tuple or simple array (convert to tuple first)
        elif isinstance(value, (tuple, np.ndarray)) and len(value) == 5:
            values = tuple(value)
            labels = ["Upper Band", "Middle Band", "Lower Band", "Bandwidth", "%B"]
            
            for label, val in zip(labels, values):
                if np.isnan(val):
                    value_str = "NaN"
                else:
                    value_str = f"{val:.6f}"
                result.append((label, value_str))
        else:
            # Fallback to default formatting
            result = super().format_value(value, source_value, timestamp_str)
            
        return result
