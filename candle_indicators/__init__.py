import os
import importlib
import inspect
from typing import Dict, Any

# Import base classes
from .base import Indicator, IndicatorRegistry

def auto_register_indicators(verbose=False):
    """Automatically register all indicator classes from the indicators package.
    
    Args:
        verbose: If True, print debug information during registration
    """
    # Get the directory of the indicators package
    package_dir = os.path.dirname(__file__)
    
    # Find all Python files in the package
    module_files = [f[:-3] for f in os.listdir(package_dir) 
                   if f.endswith('.py') and f != '__init__.py']
    
    if verbose:
        print(f"Found indicator modules: {module_files}")
    
    # Import each module
    for module_name in module_files:
        try:
            # Import the module
            module = importlib.import_module(f'.{module_name}', package='candle_indicators')
            if verbose:
                print(f"Imported module: {module_name}")
            
            # Check for indicator classes in the module
            for name, obj in inspect.getmembers(module):
                if (inspect.isclass(obj) and 
                    issubclass(obj, Indicator) and 
                    hasattr(obj, 'PREFIX')):
                    if verbose:
                        print(f"Found indicator: {obj.PREFIX} ({obj.__name__})")
        except Exception as e:
            if verbose:
                print(f"Error importing module {module_name}: {e}")

# Export key classes and functions
__all__ = ['Indicator', 'IndicatorRegistry', 'auto_register_indicators']

# Auto-register all indicators when the package is imported
auto_register_indicators(verbose=False)
