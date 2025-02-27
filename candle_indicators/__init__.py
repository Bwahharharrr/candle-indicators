import os
import importlib
import inspect
from typing import Dict, Any

# Import base classes
from .base import Indicator, IndicatorRegistry

def auto_register_indicators():
    """Automatically register all indicator classes from the indicators package."""
    # Get the directory of the indicators package
    package_dir = os.path.dirname(__file__)
    
    # Find all Python files in the package
    module_files = [f[:-3] for f in os.listdir(package_dir) 
                   if f.endswith('.py') and f != '__init__.py']
    
    # Import each module
    for module_name in module_files:
        # Import the module
        importlib.import_module(f'.{module_name}', package='candle_indicators')

# Export key classes and functions
__all__ = ['Indicator', 'IndicatorRegistry', 'auto_register_indicators']
