import numpy as np
from typing import List
from .base import Indicator, IndicatorRegistry

@IndicatorRegistry.register
class SimpleMovingAverage(Indicator):
    """Simple Moving Average indicator."""
    
    PREFIX = "SMA"
    
    def __init__(self, timeframe: str, period: int, source: str = 'close'):
        super().__init__(f"SMA{period}", timeframe, source)
        self.period = period
    
    @classmethod
    def from_command(cls, ind_def: str, timeframe: str, source: str) -> 'SimpleMovingAverage':
        """Create an SMA instance from command line definition."""
        period = int(ind_def[3:])
        return cls(timeframe, period, source)
    
    @classmethod
    @property
    def description(cls) -> str:
        return "Simple Moving Average"
    
    @classmethod
    @property
    def format_str(cls) -> str:
        return "SMA<period>:<timeframe>[:<source>]"
    
    @classmethod
    @property
    def examples(cls) -> List[str]:
        return [
            "SMA20:1h - 20-period SMA on 1h timeframe using close prices",
            "SMA20:1h:high - 20-period SMA on 1h timeframe using high prices"
        ]
    
    @classmethod
    @property
    def params(cls) -> List[dict]:
        return [
            {"name": "period", "description": "Number of candles to average (required)"},
            {"name": "source", "description": "Price field to use (optional, default: close)"}
        ]
        
    def calculate(self, data: np.ndarray) -> np.ndarray:
        if len(data) < self.period:
            return np.array([])
            
        # Extract source data
        source_data = data[self.source]
        
        # Use numpy's cumsum for faster calculation
        result = np.zeros(len(data))
        result[:self.period-1] = np.nan
        
        cumsum = np.cumsum(np.insert(source_data, 0, 0))
        result[self.period-1:] = (cumsum[self.period:] - cumsum[:-self.period]) / self.period
            
        return result


@IndicatorRegistry.register
class ExponentialMovingAverage(Indicator):
    """Exponential Moving Average indicator."""
    
    PREFIX = "EMA"
    
    def __init__(self, timeframe: str, period: int, source: str = 'close'):
        super().__init__(f"EMA{period}", timeframe, source)
        self.period = period
        
    @classmethod
    def from_command(cls, ind_def: str, timeframe: str, source: str) -> 'ExponentialMovingAverage':
        """Create an EMA instance from command line definition."""
        period = int(ind_def[3:])
        return cls(timeframe, period, source)
    
    @classmethod
    @property
    def description(cls) -> str:
        return "Exponential Moving Average"
    
    @classmethod
    @property
    def format_str(cls) -> str:
        return "EMA<period>:<timeframe>[:<source>]"
    
    @classmethod
    @property
    def examples(cls) -> List[str]:
        return [
            "EMA20:1h - 20-period EMA on 1h timeframe using close prices",
            "EMA50:4h:high - 50-period EMA on 4h timeframe using high prices"
        ]
    
    @classmethod
    @property
    def params(cls) -> List[dict]:
        return [
            {"name": "period", "description": "Number of candles for EMA calculation (required)"},
            {"name": "source", "description": "Price field to use (optional, default: close)"}
        ]
        
    def calculate(self, data: np.ndarray) -> np.ndarray:
        if len(data) < 2:  # Need at least 2 points for EMA
            return np.array([])
            
        # Extract source data
        source_data = data[self.source]
        
        # Calculate the multiplier
        multiplier = 2 / (self.period + 1)
        
        # Initialize result array
        result = np.zeros(len(data))
        result[:] = np.nan
        
        # Calculate SMA for the first point
        if len(data) >= self.period:
            result[self.period-1] = np.mean(source_data[:self.period])
            
            # Calculate EMA for remaining points
            for i in range(self.period, len(data)):
                result[i] = (source_data[i] - result[i-1]) * multiplier + result[i-1]
                
        return result
