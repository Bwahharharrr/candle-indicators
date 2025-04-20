import numpy as np
from typing import List
from .base import Indicator, IndicatorRegistry


@IndicatorRegistry.register
class CustomWeightedEMA(Indicator):
    """Custom Weighted Exponential Moving Average indicator.
    Similar to the standard EMA but uses a custom weight factor in the multiplier."""
    
    PREFIX = "CWEMA"
    
    def __init__(self, timeframe: str, period: int, weight: float = 2.0, source: str = 'close'):
        super().__init__(f"CWEMA{period},{weight}", timeframe, source)
        self.period = period
        self.weight = weight
        self._multiplier = weight / (period + 1)
        
        # Preallocate a fixed-size output buffer
        buffer_capacity = max(200, period * 2)
        self.buffer = np.full(buffer_capacity, np.nan)
        self.position = 0
        self.is_filled = False
        self._buffer_count = 0
        
        # For incremental calculation
        self._last_ema = None
        self._sma_buffer = np.zeros(period, dtype=float)
        self._sma_pos = 0
        self._sma_count = 0
        self._sma_sum = 0.0
        
        # Initialize values and _last_data_length
        self.values = np.array([])
        self._last_data_length = 0
    
    @classmethod
    def from_command(cls, ind_def: str, timeframe: str, source: str) -> 'CustomWeightedEMA':
        """Create a CWEMA instance from command line definition."""
        prefix_len = len(cls.PREFIX)
        parts = ind_def.split(':')
        ind_part = parts[0]
        
        # Extract parameters after the prefix
        params_str = ind_part[prefix_len:]
        
        # Default values
        period = 20
        weight = 2.0
        
        if ',' in params_str:
            # Split by comma to get period and weight
            param_parts = params_str.split(',')
            period = int(param_parts[0])
            if len(param_parts) > 1:
                weight = float(param_parts[1])
        elif params_str:
            # Just period provided
            period = int(params_str)
            
        return cls(timeframe, period, weight, source)
    
    @classmethod
    def description(cls, pretty=False) -> str:
        desc = "Custom Weighted Exponential Moving Average"
        if pretty:
            from candle_indicators.utils.ui import COLOR_DESC, Style
            return f"{COLOR_DESC}{desc}{Style.RESET_ALL}"
        return desc
    
    @classmethod
    def format_str(cls, pretty=False) -> str:
        fmt = "CWEMA<period>[,<weight>]:<timeframe>[:<source>]"
        if pretty:
            from candle_indicators.utils.ui import COLOR_TYPE, Style
            return f"{COLOR_TYPE}{fmt}{Style.RESET_ALL}"
        return fmt
    
    @classmethod
    def examples(cls, pretty=False) -> List[str]:
        examples = [
            "CWEMA20:1h, CWEMA20:1h,high",
        ]
        return examples
    
    @classmethod
    def params(cls, pretty=False) -> List[dict]:
        params = [
            {"name": "period", "description": "Number of candles for calculation (required)"},
            {"name": "weight", "description": "Custom weight multiplier (optional, default: 2.0)"},
            {"name": "source", "description": "Price field to use (optional, default: close)"}
        ]
        
        if pretty:
            from candle_indicators.utils.ui import COLOR_DESC, Style
            return [
                {"name": p["name"], "description": f"{COLOR_DESC}{p['description']}{Style.RESET_ALL}"}
                for p in params
            ]
        return params
    
    def _calculate_incremental(self, new_value: float) -> float:
        """
        Calculate the latest custom-weighted EMA incrementally.
        This is a private method that uses the same approach as EMA but with a custom multiplier.
        """
        if self._last_ema is None:
            # Still in initial SMA mode
            old_val = self._sma_buffer[self._sma_pos]
            if self._sma_count < self.period:
                self._sma_sum += new_value
                self._sma_count += 1
            else:
                self._sma_sum += (new_value - old_val)

            self._sma_buffer[self._sma_pos] = new_value
            self._sma_pos = (self._sma_pos + 1) % self.period

            if self._sma_count == self.period:
                init_ema = self._sma_sum / self.period
                self._last_ema = init_ema

                self.buffer[self.position] = init_ema
                self.position = (self.position + 1) % len(self.buffer)

                if self._buffer_count < len(self.buffer):
                    self._buffer_count += 1
                    if self._buffer_count == len(self.buffer):
                        self.is_filled = True

                return init_ema
            else:
                return np.nan
        else:
            # Normal custom-weighted EMA update
            ema_val = (new_value - self._last_ema) * self._multiplier + self._last_ema
            self._last_ema = ema_val

            self.buffer[self.position] = ema_val
            self.position = (self.position + 1) % len(self.buffer)

            if self._buffer_count < len(self.buffer):
                self._buffer_count += 1
                if self._buffer_count == len(self.buffer):
                    self.is_filled = True

            return ema_val

    def _calculate_full(self, data: np.ndarray) -> np.ndarray:
        """Calculate the full custom-weighted EMA over the entire dataset.
        This is a private method used for batch calculations."""
        if len(data) < self.period:
            return np.array([])

        source_data = data[self.source]
        data_length = len(source_data)
        result = np.full(data_length, np.nan, dtype=float)

        # Seed the EMA with an SMA for the first `period` points
        init_sma = source_data[:self.period].mean()
        result[self.period - 1] = init_sma

        # Then apply the custom-weighted EMA for the rest
        for i in range(self.period, data_length):
            result[i] = (source_data[i] - result[i - 1]) * self._multiplier + result[i - 1]

        # Update incremental state
        self._last_ema = result[-1]
        self._sma_buffer[:] = source_data[-self.period:]
        self._sma_sum = self._sma_buffer.sum()
        self._sma_count = self.period
        self._sma_pos = 0

        # Store final EMA into the ring buffer
        self.buffer[self.position] = self._last_ema
        self.position = (self.position + 1) % len(self.buffer)

        if self._buffer_count < len(self.buffer):
            self._buffer_count += 1
            if self._buffer_count == len(self.buffer):
                self.is_filled = True

        return result

    def update_with_data(self, data: np.ndarray) -> None:
        """Perform an incremental update if there's exactly one new data point.
        
        This is the primary public method for updating the custom weighted EMA with new data.
        It will automatically determine whether to do a full calculation or use
        incremental updates based on the amount of new data.
        
        Args:
            data: Price data array containing the source data field
        """
        # Extract the source data
        try:
            source_data = data[self.source]
        except (TypeError, IndexError, KeyError):
            # Handle case where data is just a raw array without field names
            source_data = data
        
        # Case 1: Single new data point - extract it and process incrementally
        if len(source_data) == 1:
            new_val = source_data[0]
            new_ema = self._calculate_incremental(new_val)
            if self.values is None or len(self.values) == 0:
                self.values = np.array([new_ema])
            else:
                self.values = np.append(self.values, new_ema)
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
                new_ema = self._calculate_incremental(new_val)
                if self.values is None or len(self.values) == 0:
                    self.values = np.array([new_ema])
                else:
                    self.values = np.append(self.values, new_ema)
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
