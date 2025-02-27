import numpy as np
from typing import List, Optional
from .base import Indicator, IndicatorRegistry
from .moving_averages import ExponentialMovingAverage

@IndicatorRegistry.register
class RSI(Indicator):
    """Relative Strength Index indicator."""
    
    PREFIX = "RSI"
    
    def __init__(self, timeframe: str, period: int = 14, source: str = 'close'):
        super().__init__(f"RSI{period}", timeframe, source)
        self.period = period
    
    @classmethod
    def from_command(cls, ind_def: str, timeframe: str, source: str) -> 'RSI':
        """Create an RSI instance from command line definition."""
        if len(ind_def) > 3:
            period = int(ind_def[3:])
        else:
            period = 14  # Default
        return cls(timeframe, period, source)
    
    @classmethod
    @property
    def description(cls) -> str:
        return "Relative Strength Index - momentum oscillator"
    
    @classmethod
    @property
    def format_str(cls) -> str:
        return "RSI[period]:<timeframe>[:<source>]"
    
    @classmethod
    @property
    def examples(cls) -> List[str]:
        return [
            "RSI14:1h - 14-period RSI on 1h timeframe",
            "RSI:4h - 14-period (default) RSI on 4h timeframe",
            "RSI7:1D:close - 7-period RSI on 1D timeframe using close prices"
        ]
    
    @classmethod
    @property
    def params(cls) -> List[dict]:
        return [
            {"name": "period", "description": "Number of candles for calculation (optional, default: 14)"},
            {"name": "source", "description": "Price field to use (optional, default: close)"}
        ]
        
    def calculate(self, data: np.ndarray) -> np.ndarray:
        if len(data) < self.period + 1:  # Need at least period+1 data points to calculate one RSI value
            return np.array([])
            
        # Extract source data
        source_data = data[self.source]
        
        # Calculate price changes
        deltas = np.diff(source_data)
        
        # Initialize result array
        result = np.zeros(len(data))
        result[:] = np.nan
        
        # Initialize gain and loss arrays
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


@IndicatorRegistry.register
class MACD(Indicator):
    """Moving Average Convergence Divergence indicator."""
    
    PREFIX = "MACD"
    
    def __init__(self, timeframe: str, fast_period: int = 12, slow_period: int = 26,
                 signal_period: int = 9, source: str = 'close',
                 fast_ma: Optional[Indicator] = None, 
                 slow_ma: Optional[Indicator] = None):
        name = f"MACD{fast_period},{slow_period},{signal_period}"
        super().__init__(name, timeframe, source)
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.signal_period = signal_period
        
        # Set up dependent MAs or use provided ones
        if fast_ma is None:
            self.fast_ma = ExponentialMovingAverage(timeframe, fast_period, source)
        else:
            self.fast_ma = fast_ma
            self.add_dependency(fast_ma)
            
        if slow_ma is None:
            self.slow_ma = ExponentialMovingAverage(timeframe, slow_period, source)
        else:
            self.slow_ma = slow_ma
            self.add_dependency(slow_ma)
    
    @classmethod
    def from_command(cls, ind_def: str, timeframe: str, source: str) -> 'MACD':
        """Create a MACD instance from command line definition."""
        # Default values
        fast_period, slow_period, signal_period = 12, 26, 9
        
        # Parse parameters if provided
        if len(ind_def) > 4:  # More than just "MACD"
            params = ind_def[4:].split(',')
            if len(params) >= 1 and params[0]:
                fast_period = int(params[0])
            if len(params) >= 2 and params[1]:
                slow_period = int(params[1])
            if len(params) >= 3 and params[2]:
                signal_period = int(params[2])
                
        return cls(timeframe, fast_period, slow_period, signal_period, source)
    
    @classmethod
    @property
    def description(cls) -> str:
        return "Moving Average Convergence Divergence"
    
    @classmethod
    @property
    def format_str(cls) -> str:
        return "MACD[fast_period,slow_period,signal_period]:<timeframe>[:<source>]"
    
    @classmethod
    @property
    def examples(cls) -> List[str]:
        return [
            "MACD:1h - Default MACD(12,26,9) on 1h timeframe",
            "MACD12,26,9:4h - MACD with default parameters on 4h timeframe",
            "MACD6,13,4:1D:high - MACD with custom parameters on 1D timeframe using high prices"
        ]
    
    @classmethod
    @property
    def params(cls) -> List[dict]:
        return [
            {"name": "fast_period", "description": "Fast EMA period (optional, default: 12)"},
            {"name": "slow_period", "description": "Slow EMA period (optional, default: 26)"},
            {"name": "signal_period", "description": "Signal SMA period (optional, default: 9)"},
            {"name": "source", "description": "Price field to use (optional, default: close)"}
        ]
        
    def calculate(self, data: np.ndarray) -> np.ndarray:
        if len(data) <= self.slow_period:
            # Not enough data
            return np.array([])
            
        # Define the dtype for structured output
        macd_dtype = np.dtype([
            ('macd', 'f8'),
            ('signal', 'f8'),
            ('histogram', 'f8')
        ])
        
        # Calculate component MAs
        self.fast_ma.update(data)
        self.slow_ma.update(data)
        
        fast_ma_values = self.fast_ma.values
        slow_ma_values = self.slow_ma.values
        
        # Create result array
        result = np.zeros(len(data), dtype=macd_dtype)
        result['macd'][:] = np.nan
        result['signal'][:] = np.nan
        result['histogram'][:] = np.nan
        
        # Valid range starts after both MAs can be calculated
        valid_start = max(self.fast_period, self.slow_period) - 1
        if valid_start < len(data):
            # Calculate MACD line
            macd_line = fast_ma_values[valid_start:] - slow_ma_values[valid_start:]
            
            # Calculate signal line using EMA of MACD line
            if len(macd_line) >= self.signal_period:
                signal_line = np.zeros(len(macd_line))
                
                # First value is SMA
                signal_line[:self.signal_period-1] = np.nan
                signal_line[self.signal_period-1] = np.mean(macd_line[:self.signal_period])
                
                # Calculate EMA
                multiplier = 2 / (self.signal_period + 1)
                for i in range(self.signal_period, len(macd_line)):
                    signal_line[i] = (macd_line[i] - signal_line[i-1]) * multiplier + signal_line[i-1]
                
                # Calculate histogram
                histogram = macd_line - signal_line
                
                # Transfer values to result array
                result['macd'][valid_start:] = macd_line
                result['signal'][valid_start+self.signal_period-1:] = signal_line[self.signal_period-1:]
                result['histogram'][valid_start+self.signal_period-1:] = histogram[self.signal_period-1:]
                
        return result
