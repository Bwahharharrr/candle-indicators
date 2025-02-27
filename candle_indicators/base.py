import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Type, Optional, Tuple

class IndicatorRegistry:
    """Registry for indicator types."""
    
    _registry = {}
    
    @classmethod
    def register(cls, indicator_class):
        """Register an indicator class using its PREFIX."""
        if not hasattr(indicator_class, 'PREFIX') or indicator_class.PREFIX is None:
            raise ValueError(f"Indicator class {indicator_class.__name__} must have a PREFIX attribute")
        
        cls._registry[indicator_class.PREFIX] = indicator_class
        return indicator_class  # Return the class to allow use as a decorator
    
    @classmethod
    def create_indicator(cls, ind_def: str, timeframe: str, source: str, manager):
        """Create an indicator using the appropriate class."""
        for prefix, indicator_class in cls._registry.items():
            if ind_def.startswith(prefix):
                indicator = indicator_class.from_command(ind_def, timeframe, source)
                manager.add_indicator(indicator)
                return indicator
        return None
    
    @classmethod
    def get_all_indicators(cls) -> Dict[str, Dict[str, Any]]:
        """Get all registered indicators with their documentation."""
        return {
            prefix: {
                'class': indicator_class,
                'description': indicator_class.description,
                'format': indicator_class.format_str,
                'examples': indicator_class.examples,
                'params': indicator_class.params
            }
            for prefix, indicator_class in cls._registry.items()
        }

class Indicator(ABC):
    """Base class for all indicators."""
    
    # Class must define this attribute for auto-registration
    PREFIX = None
    
    def __init__(self, name: str, timeframe: str, source: str = 'close'):
        self.name = name
        self.timeframe = timeframe
        self.source = source
        self.values = np.array([])
        self.dependencies = []
        self._last_data_length = 0  # Track data length to avoid recalculating
        self.display_mode = "default"  # Can be "default", "columnar", etc.
    
    @classmethod
    @abstractmethod
    def from_command(cls, ind_def: str, timeframe: str, source: str) -> 'Indicator':
        """Create an indicator instance from command line definition."""
        pass
    
    @classmethod
    @property
    def description(cls) -> str:
        """Description of the indicator."""
        return "Base Indicator"
    
    @classmethod
    @property
    def format_str(cls) -> str:
        """Format string for command line usage."""
        return "<indicator>:<timeframe>[:<source>]"
    
    @classmethod
    @property
    def examples(cls) -> List[str]:
        """Examples of command line usage."""
        return []
    
    @classmethod
    @property
    def params(cls) -> List[dict]:
        """Parameters description."""
        return [
            {"name": "source", "description": "Price field to use (optional, default: close, options: open, high, low, close, volume)"}
        ]
        
    @abstractmethod
    def calculate(self, data: np.ndarray) -> np.ndarray:
        """Calculate indicator values from price data."""
        pass
    
    def update(self, data: np.ndarray) -> None:
        """Update indicator values with new data."""
        # Only recalculate if data length has changed
        if len(data) != self._last_data_length:
            try:
                self.values = self.calculate(data)
            except Exception as e:
                self.values = np.array([])
            self._last_data_length = len(data)
    
    def add_dependency(self, indicator: 'Indicator') -> None:
        """Add a dependency on another indicator."""
        if indicator.timeframe != self.timeframe:
            raise ValueError(f"Indicator timeframes must match: {self.timeframe} != {indicator.timeframe}")
        self.dependencies.append(indicator)
    
    def get_value(self, index: int = -1) -> Any:
        """Get indicator value at specific index (default: latest)."""
        try:
            if not hasattr(self, 'values') or self.values is None:
                return None
            
            if hasattr(self.values, '__len__') and len(self.values) == 0:
                return None
            
            if hasattr(self.values, '__len__') and index >= len(self.values):
                return None
            
            if hasattr(self.values, '__getitem__'):
                value = self.values[index]
                return value
            else:
                # Not an array-like object
                return self.values
        except Exception as e:
            return None
    
    def format_value(self, value, source_value=None, timestamp_str=None) -> List[Tuple[str, str]]:
        """Format indicator value for display.
        
        Args:
            value: The indicator value to format
            source_value: The source value used for calculation (e.g., close price)
            timestamp_str: The timestamp string for this value
            
        Returns:
            List of (label, formatted_value) tuples, where label can be empty for main value
            or descriptive for components of complex indicators
        """
        # Default implementation for simple indicators
        if value is None:
            return [("", "N/A")]
        elif isinstance(value, float) or isinstance(value, np.float64):
            if np.isnan(value):
                return [("", "NaN")]
            else:
                return [("", f"{value:.6f}")]
        elif isinstance(value, np.ndarray):
            # Default formatting for array values - join with commas
            value_str = ", ".join(f"{v:.6f}" for v in value)
            return [("", value_str)]
        else:
            # Default formatting for other types
            return [("", str(value))]
    
    def __str__(self) -> str:
        if self.source == 'close':  # Only show source if it's not the default
            return f"{self.name}({self.timeframe})"
        return f"{self.name}[{self.source}]({self.timeframe})"
