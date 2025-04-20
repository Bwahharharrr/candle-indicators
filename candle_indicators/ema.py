import numpy as np
from typing import List
from .base import Indicator, IndicatorRegistry


@IndicatorRegistry.register
class ExponentialMovingAverage(Indicator):
    """Exponential Moving Average indicator with incremental and bulk calculations."""
    
    PREFIX = "EMA"
    
    def __init__(self, timeframe: str, period: int = 12, source: str = 'close'):
        super().__init__(f"EMA{period}", timeframe, source)
        self.period = period
        self._multiplier = 2.0 / (period + 1.0)  # Standard EMA multiplier
        
        # Preallocate a fixed-size output buffer for the final EMA values
        buffer_capacity = max(200, period * 2)
        self.buffer = np.full(buffer_capacity, np.nan)
        self.position = 0
        self.is_filled = False
        self._buffer_count = 0  # Tracks how many valid entries in the buffer
        self.values = np.array([])  # Initialize values array
        self._last_data_length = 0  # Initialize last data length
        
        # For incremental SMA-seeded calculation
        self._sma_buffer = np.zeros(period, dtype=float)
        self._sma_pos = 0
        self._sma_count = 0
        self._sma_sum = 0.0
        
        # Last calculated EMA value
        self._last_ema = np.nan
    
    @classmethod
    def from_command(cls, ind_def: str, timeframe: str, source: str) -> 'ExponentialMovingAverage':
        """Create an EMA instance from command line definition."""
        prefix_len = len(cls.PREFIX)
        ind_part = ind_def[:ind_def.find(':')] if ':' in ind_def else ind_def
        period_str = ind_part[prefix_len:]
        period = int(period_str) if period_str else 12  # Default to 12 if not specified
        return cls(timeframe, period, source)
    
    @classmethod
    def description(cls, pretty=False) -> str:
        desc = "Exponential Moving Average"
        if pretty:
            from candle_indicators.utils.ui import COLOR_DESC, Style
            return f"{COLOR_DESC}{desc}{Style.RESET_ALL}"
        return desc
    
    @classmethod
    def format_str(cls, pretty=False) -> str:
        fmt = "EMA<period>:<timeframe>[:<source>]"
        if pretty:
            from candle_indicators.utils.ui import COLOR_TYPE, Style
            return f"{COLOR_TYPE}{fmt}{Style.RESET_ALL}"
        return fmt
    
    @classmethod
    def examples(cls, pretty=False) -> List[str]:
        examples = [
            "EMA12:1h, EMA26:1D,high",
        ]
        return examples
    
    @classmethod
    def params(cls, pretty=False) -> List[dict]:
        params = [
            {"name": "period", "description": "Number of candles for averaging (optional, default: 12)"},
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
        Incrementally calculate the latest EMA value.
        This is a private method that handles calculation for each new data point.
        If we haven't reached the first full period, accumulate in the SMA buffer.
        Once we have that first average, switch to EMA updates.
        """
        if np.isnan(self._last_ema):
            # First pass: calculate SMA
            old_val = self._sma_buffer[self._sma_pos]
            
            if self._sma_count < self.period:
                # Still accumulating initial values
                self._sma_sum += new_value
                self._sma_count += 1
            else:
                # Have a full window, update sum by replacing oldest value
                self._sma_sum += (new_value - old_val)
            
            # Update the SMA buffer
            self._sma_buffer[self._sma_pos] = new_value
            self._sma_pos = (self._sma_pos + 1) % self.period
            
            # Check if we've accumulated enough values for the initial SMA
            if self._sma_count == self.period:
                # Initialize EMA with SMA
                init_ema = self._sma_sum / self.period
                self._last_ema = init_ema
                
                # Store in buffer
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
            # Regular EMA calculation: EMA_today = (Price_today - EMA_yesterday) * multiplier + EMA_yesterday
            ema_val = (new_value - self._last_ema) * self._multiplier + self._last_ema
            self._last_ema = ema_val
            
            # Store in buffer
            self.buffer[self.position] = ema_val
            self.position = (self.position + 1) % len(self.buffer)
            
            if self._buffer_count < len(self.buffer):
                self._buffer_count += 1
                if self._buffer_count == len(self.buffer):
                    self.is_filled = True
                    
            return ema_val
    
    def _calculate_full(self, data: np.ndarray) -> np.ndarray:
        """Calculate EMA for an entire dataset (batch mode).
        This is a private method used for full calculations over a dataset."""
        if len(data) < self.period:
            return np.array([])
            
        source_data = data[self.source]
        data_length = len(source_data)
        result = np.full(data_length, np.nan)
        
        # First value is SMA
        first_sma = np.mean(source_data[:self.period])
        result[self.period-1] = first_sma
        
        # Calculate the rest using the EMA formula
        for i in range(self.period, data_length):
            result[i] = (source_data[i] - result[i-1]) * self._multiplier + result[i-1]
        
        # Update state for incremental calculations
        self._last_ema = result[-1]
        
        # Also update the SMA buffer with the last period values
        # (this helps with the transition to incremental updates)
        self._sma_buffer[:] = source_data[-self.period:]
        self._sma_sum = np.sum(self._sma_buffer)
        self._sma_count = self.period
        self._sma_pos = 0
        
        # Update the circular buffer
        self.buffer[self.position] = self._last_ema
        self.position = (self.position + 1) % len(self.buffer)
        
        if self._buffer_count < len(self.buffer):
            self._buffer_count += 1
            if self._buffer_count == len(self.buffer):
                self.is_filled = True
        
        return result
    
    def update_with_data(self, data: np.ndarray) -> None:
        """Optimized update that handles a single new point incrementally.
        
        This is the primary public method for updating the EMA with new data.
        It automatically determines whether to use incremental calculation or 
        perform a full recalculation.
        
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
            if not np.isnan(new_ema):  # Only append valid values
                if self.values is None or len(self.values) == 0:
                    self.values = np.array([new_ema])
                else:
                    self.values = np.append(self.values, new_ema)
            self._last_data_length = self._last_data_length + 1
            return
            
        # Case 2: First calculation or full recalculation with multiple points
        if len(source_data) >= self.period:
            # Create a structured array for _calculate_full if needed
            if isinstance(data, np.ndarray) and not hasattr(data, 'dtype') or not hasattr(data.dtype, 'names') or self.source not in data.dtype.names:
                structured_data = np.zeros(len(source_data), dtype=[(self.source, float)])
                structured_data[self.source] = source_data
                calc_data = structured_data
            else:
                calc_data = data

            if self._last_data_length == 0:
                # First calculation
                self.values = self._calculate_full(calc_data)
                self._last_data_length = len(calc_data) 
            elif len(calc_data) == self._last_data_length + 1:
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
                self.values = self._calculate_full(calc_data)
                self._last_data_length = len(calc_data)
        else:
            # Not enough data yet, but still process each value incrementally
            for val in source_data:
                self._calculate_incremental(val)
            self.values = np.array([])
            self._last_data_length = len(source_data)