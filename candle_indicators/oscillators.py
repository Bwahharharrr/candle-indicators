import numpy as np
from typing import List, Dict, Callable, Any
from .base import Indicator, IndicatorRegistry

@IndicatorRegistry.register
class CompositeIndicator(Indicator):
    """Indicator that combines values from other indicators."""
    
    PREFIX = "COMP"
    
    def __init__(self, name: str, timeframe: str, calculation_func: Callable):
        super().__init__(name, timeframe, 'close')  # Source doesn't matter for composite
        self.calculation_func = calculation_func
    
    @classmethod
    def from_command(cls, ind_def: str, timeframe: str, source: str) -> 'CompositeIndicator':
        """Create a composite indicator from command line definition.
        Note: This is a placeholder - composite indicators are typically
        created programmatically, not from command line."""
        name = ind_def[4:] if len(ind_def) > 4 else "CompositeIndicator"
        return cls(name, timeframe, lambda data, indicators: np.array([]))
    
    @classmethod
    @property
    def description(cls) -> str:
        return "Composite indicator that combines values from other indicators"
    
    @classmethod
    @property
    def format_str(cls) -> str:
        return "COMP<name>:<timeframe>"
    
    @classmethod
    @property
    def examples(cls) -> List[str]:
        return [
            "Composite indicators are typically created programmatically, not from command line"
        ]
    
    @classmethod
    @property
    def params(cls) -> List[dict]:
        return [
            {"name": "name", "description": "Name of the composite indicator"}
        ]
        
    def calculate(self, data: np.ndarray) -> np.ndarray:
        # Make sure all dependencies are updated
        for dep in self.dependencies:
            dep.update(data)
            
        # Call the calculation function with data and dependencies
        return self.calculation_func(data, self.dependencies)
