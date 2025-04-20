import numpy as np
from typing import List
from .base import Indicator, IndicatorRegistry


@IndicatorRegistry.register
class SimpleMovingAverage(Indicator):
    """Simple Moving Average indicator with ring buffer optimization."""

    PREFIX = "SMA"

    def __init__(self, timeframe: str, period: int, source: str = 'close'):
        super().__init__(f"SMA{period}", timeframe, source)
        self.period = period

        # Preallocate a fixed-size output buffer for final SMA values
        buffer_capacity = max(200, period * 2)
        self.buffer = np.full(buffer_capacity, np.nan)
        self.position = 0
        self.is_filled = False
        self._buffer_count = 0  # Tracks how many valid entries in self.buffer
        self.values = np.array([])  # Initialize values array

        # For incremental calculation:
        # - _window_buffer: holds the last `period` values in a ring buffer
        # - _window_pos: current write position in _window_buffer
        # - _window_count: how many values have been added (up to self.period)
        # - _sum: running total of the last `period` values
        self._window_buffer = np.zeros(period, dtype=float)
        self._window_pos = 0
        self._window_count = 0
        self._sum = 0.0
        self._last_data_length = 0  # Initialize last data length

    @classmethod
    def from_command(cls, ind_def: str, timeframe: str, source: str) -> 'SimpleMovingAverage':
        """Create an SMA instance from command line definition."""
        prefix_len = len(cls.PREFIX)
        parts = ind_def.split(':')
        ind_part = parts[0]
        period_str = ind_part[prefix_len:]
        period = int(period_str)
        return cls(timeframe, period, source)

    @classmethod
    def description(cls, pretty=False) -> str:
        desc = "Simple Moving Average"
        if pretty:
            from candle_indicators.utils.ui import COLOR_DESC, Style
            return f"{COLOR_DESC}{desc}{Style.RESET_ALL}"
        return desc

    @classmethod
    def format_str(cls, pretty=False) -> str:
        fmt = "SMA<period>:<timeframe>[:<source>]"
        if pretty:
            from candle_indicators.utils.ui import COLOR_TYPE, Style
            return f"{COLOR_TYPE}{fmt}{Style.RESET_ALL}"
        return fmt

    @classmethod
    def examples(cls, pretty=False) -> List[str]:
        examples = [
            "SMA20:1h, SMA20:1h:high",
        ]
        return examples

    @classmethod
    def params(cls, pretty=False) -> List[dict]:
        params = [
            {"name": "period", "description": "Number of candles to average (required)"},
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
        Calculate only the latest SMA value using the ring buffer mechanism.
        This is a private method that avoids repeated Python list operations 
        in favor of a fixed-size NumPy array.
        """
        old_value = self._window_buffer[self._window_pos]  # Value to overwrite

        if self._window_count < self.period:
            # Still filling the buffer for the first time
            self._sum += new_value
            self._window_count += 1
        else:
            # Buffer is full; subtract the oldest value before overwriting
            self._sum += (new_value - old_value)

        # Overwrite or place the new value
        self._window_buffer[self._window_pos] = new_value
        self._window_pos = (self._window_pos + 1) % self.period

        # Only produce an SMA if the buffer is fully initialized
        if self._window_count == self.period:
            sma_value = self._sum / self.period

            # Write into the final output buffer
            self.buffer[self.position] = sma_value
            self.position = (self.position + 1) % len(self.buffer)

            # Update the buffer count
            if self._buffer_count < len(self.buffer):
                self._buffer_count += 1
                if self._buffer_count == len(self.buffer):
                    self.is_filled = True

            return sma_value
        else:
            return np.nan

    def _calculate_full(self, data: np.ndarray) -> np.ndarray:
        """
        Calculate SMA for the entire dataset (required for charting).
        This is a private method that uses numpy.cumsum for batch calculation, 
        which is generally efficient.
        """
        if len(data) < self.period:
            return np.array([])

        source_data = data[self.source]
        source_len = len(source_data)

        # Preallocate output array with NaNs
        result = np.full(source_len, np.nan)

        # Compute the cumulative sum
        cumsum = np.cumsum(source_data)

        # Compute the SMA for periods >= `period`
        # SMA_i = (sum(data[i-period+1:i+1])) / period
        result[self.period-1:] = (cumsum[self.period-1:] - np.append(0, cumsum[:-self.period])) / self.period

        # Reset the incremental calculation state to match the last window of data
        if source_len >= self.period:
            last_window = source_data[-self.period:]
            self._window_buffer[:] = last_window
            self._window_pos = 0
            self._window_count = self.period
            self._sum = np.sum(last_window)

            # Update circular buffer with latest SMA
            latest_sma = result[-1]
            self.buffer[self.position] = latest_sma
            self.position = (self.position + 1) % len(self.buffer)

            if self._buffer_count < len(self.buffer):
                self._buffer_count += 1
                if self._buffer_count == len(self.buffer):
                    self.is_filled = True

        return result

    def update_with_data(self, data: np.ndarray) -> None:
        """
        Update indicator with new data.
        
        This is the primary public method that performs an incremental calculation 
        when only one new data point arrives, or a full calculation if the data size 
        has changed more significantly.
        
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
            new_sma = self._calculate_incremental(new_val)
            if self.values is None or len(self.values) == 0:
                self.values = np.array([new_sma])
            else:
                self.values = np.append(self.values, new_sma)
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
                new_sma = self._calculate_incremental(new_val)
                if self.values is None or len(self.values) == 0:
                    self.values = np.array([new_sma])
                else:
                    self.values = np.append(self.values, new_sma)
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
