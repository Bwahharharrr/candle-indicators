import numpy as np
from typing import List, Optional, Tuple
from .base import Indicator, IndicatorRegistry
from .moving_averages import SimpleMovingAverage

@IndicatorRegistry.register
class BollingerBands(Indicator):
    """Bollinger Bands indicator."""
    
    PREFIX = "BB"
    
    def __init__(self, timeframe: str, period: int = 20, deviations: float = 2.0, 
                 source: str = 'close', base_ma: Optional[Indicator] = None):
        name = f"BB{period},{deviations}"
        super().__init__(name, timeframe, source)
        self.period = period
        self.deviations = deviations
        self.display_mode = "columnar"  # Set display mode to columnar
        
        # Use provided MA or create a new SMA
        if base_ma is None:
            self.base_ma = SimpleMovingAverage(timeframe, period, source)
        else:
            self.base_ma = base_ma
            # Register dependency to ensure it gets updated
            self.add_dependency(base_ma)
    
    @classmethod
    def from_command(cls, ind_def: str, timeframe: str, source: str) -> 'BollingerBands':
        """Create a BollingerBands instance from command line definition."""
        period = 20  # Default
        deviations = 2.0  # Default
        
        if len(ind_def) > 2:  # More than just "BB"
            params = ind_def[2:].split(',')
            if len(params) >= 1 and params[0]:
                period = int(params[0])
            if len(params) >= 2 and params[1]:
                deviations = float(params[1])
        
        return cls(timeframe, period, deviations, source)
    
    @classmethod
    @property
    def description(cls) -> str:
        return "Bollinger Bands - volatility bands around moving average"
    
    @classmethod
    @property
    def format_str(cls) -> str:
        return "BB[period,deviations]:<timeframe>[:<source>]"
    
    @classmethod
    @property
    def examples(cls) -> List[str]:
        return [
            "BB:1h - Default Bollinger Bands(20,2.0) on 1h timeframe",
            "BB20,2:4h - Bollinger Bands with 20 period and 2 stdevs on 4h timeframe",
            "BB50,2.5:1D:high - Bollinger Bands with custom parameters on 1D timeframe using high prices"
        ]
    
    @classmethod
    @property
    def params(cls) -> List[dict]:
        return [
            {"name": "period", "description": "Number of candles for the moving average (optional, default: 20)"},
            {"name": "deviations", "description": "Number of standard deviations for bands (optional, default: 2.0)"},
            {"name": "source", "description": "Price field to use (optional, default: close)"}
        ]
        
    def calculate(self, data: np.ndarray) -> np.ndarray:
        if len(data) < self.period:
            return np.array([])
            
        # Define the dtype for structured output
        bb_dtype = np.dtype([
            ('upper', 'f8'),
            ('middle', 'f8'),
            ('lower', 'f8'),
            ('bandwidth', 'f8'),
            ('percent_b', 'f8')
        ])
        
        # Calculate the middle band (SMA)
        self.base_ma.update(data)
        middle_band = self.base_ma.values
        
        # Create result array
        result = np.zeros(len(data), dtype=bb_dtype)
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
