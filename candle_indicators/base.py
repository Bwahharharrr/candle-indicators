"""Base classes and utilities for candle indicators."""
import inspect
import sys
import os
from typing import Dict, Any, List, Optional, Union, Tuple
import numpy as np
import pandas as pd
from datetime import datetime

# Import from internal utils package
from candle_indicators.utils.ui import (
    Fore, Style, format_header, format_subheader, format_method,
    format_description, format_error, format_component, format_value,
    format_example_header, INFO, WARNING, ERROR, SUCCESS, UPDATE,
    COLOR_DIR, COLOR_FILE, COLOR_TIMESTAMPS, COLOR_ROWS, COLOR_NEW,
    COLOR_VAR, COLOR_TYPE, COLOR_DESC, COLOR_REQ
)

import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Type, Optional, Tuple, Callable
from functools import wraps
import textwrap  # Add import for textwrap

def indicator_method(description=None, returns=None):
    """Decorator to mark and document indicator accessor methods.
    
    Args:
        description: Description of what the method returns
        returns: Description of the return type/format
        
    Example:
        @indicator_method("Get the upper Bollinger Band", "Upper band value or series")
        def upper(self, offset=0, length=None):
            ...
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
            
        # Attach metadata to the function
        wrapper._is_indicator_method = True
        wrapper._method_description = description
        wrapper._method_returns = returns
        return wrapper
    return decorator

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
    def create_indicator(cls, indicator_string: str):
        """
        Create an indicator using a single string parameter.
        
        Format: IndicatorName{settings}:timeframe:source
        Examples:
            - SMA20:1h:close
            - RSI14:1D:close
            - MACD12,26,9:4h:high
            
        The source part is optional and defaults to 'close' if not provided.
        """
        # Split the indicator string into parts based on colon
        parts = indicator_string.split(':')
        
        if len(parts) < 2:
            raise ValueError(f"Invalid indicator format: '{indicator_string}'. Expected format: 'IndicatorName{settings}:timeframe[:source]'")
            
        # Extract the indicator definition (name + parameters)
        ind_def = parts[0]
        
        # Extract the timeframe
        timeframe = parts[1]
        
        # Extract the source, default to 'close' if not provided
        source = parts[2] if len(parts) >= 3 else "close"
        
        # Find the actual prefix by looking at what's registered
        # e.g., 'SMA' from 'SMA20'
        prefix = None
        for registered_prefix in cls._registry.keys():
            if ind_def.startswith(registered_prefix):
                prefix = registered_prefix
                break
                
        if prefix is None:
            # Try direct match in case the format is unusual
            if ind_def in cls._registry:
                prefix = ind_def
            else:
                raise ValueError(f"No indicator class found for '{ind_def}'")
        
        # Look for the indicator class in the registry
        indicator_class = cls._registry[prefix]
        
        # Create the indicator
        indicator = indicator_class.from_command(ind_def, timeframe, source)
        
        return indicator
    
    @classmethod
    def get_all_indicators(cls, pretty=False) -> Dict[str, Dict[str, Any]]:
        """Get all registered indicators with their documentation.
        
        Args:
            pretty: If True, returns formatted colored strings for display
        """
        if pretty:
            return {
                prefix: {
                    'class': indicator_class,
                    'description': indicator_class.description(pretty=True),
                    'format': indicator_class.format_str(pretty=True),
                    'examples': indicator_class.examples(pretty=True),
                    'params': indicator_class.params(pretty=True)
                }
                for prefix, indicator_class in cls._registry.items()
            }
        else:
            return {
                prefix: {
                    'class': indicator_class,
                    'description': indicator_class.description(),
                    'format': indicator_class.format_str(),
                    'examples': indicator_class.examples(),
                    'params': indicator_class.params()
                }
                for prefix, indicator_class in cls._registry.items()
            }

class Indicator(ABC):
    """Base class for all indicators."""
    
    # Class must define this attribute for auto-registration
    PREFIX = None
    
    # COMPONENTS is a dict mapping component names to nested indicator instances
    COMPONENTS = {}
    
    def __init__(self, name: str, timeframe: str, source: str = 'close'):
        self.name = name
        self.timeframe = timeframe
        self.source = source
        
        # Will be set by calculate
        self.buffer = None
        self.position = 0
        self.buffer_size = 0
        self.is_filled = False
        
        # Dependencies (other indicators that this one depends on)
        self.dependencies = []
    
    @classmethod
    @abstractmethod
    def from_command(cls, ind_def: str, timeframe: str, source: str) -> 'Indicator':
        """Create an indicator instance from command line definition."""
        # This is an abstract method that each indicator must implement
        pass
    
    @classmethod
    @abstractmethod
    def description(cls, pretty=False) -> str:
        """Description of the indicator.
        
        Args:
            pretty: If True, returns a formatted colored string
        """
        raise NotImplementedError("Indicator subclasses must implement the description method")
    
    @classmethod
    @abstractmethod
    def format_str(cls, pretty=False) -> str:
        """Format string for command line usage.
        
        Args:
            pretty: If True, returns a formatted colored string
        """
        raise NotImplementedError("Indicator subclasses must implement the format_str method")
    
    @classmethod
    @abstractmethod
    def examples(cls, pretty=False) -> List[str]:
        """Examples of command line usage.
        
        Args:
            pretty: If True, returns formatted colored strings
        """
        raise NotImplementedError("Indicator subclasses must implement the examples method")
    
    @classmethod
    @abstractmethod
    def params(cls, pretty=False) -> List[dict]:
        """Parameters description.
        
        Args:
            pretty: If True, returns parameters with formatted colored descriptions
        """
        raise NotImplementedError("Indicator subclasses must implement the params method")
    
    @classmethod
    def get_info_formatted(cls) -> Dict[str, Any]:
        """Get all indicator metadata with pretty formatting applied."""
        return {
            'description': cls.description(pretty=True),
            'format': cls.format_str(pretty=True),
            'examples': cls.examples(pretty=True),
            'params': cls.params(pretty=True)
        }
    
    @abstractmethod
    def _calculate_full(self, data: np.ndarray) -> np.ndarray:
        """Calculate indicator values from price data.
        
        This is a private method that performs a full calculation over a dataset.
        This should not be called directly by users - use update_with_data instead.
        """
        pass
    
    def update_with_data(self, data: np.ndarray) -> None:
        """Update indicator values with new data.
        
        This is the primary public method to calculate and update the indicator with new data.
        It will automatically determine whether to do a full calculation or incremental update.
        
        Args:
            data: Price data in format compatible with the indicator
        """
        # Determine if dependencies need updating
        for dep in self.dependencies:
            dep.update_with_data(data)

        # Call subclass implementation to calculate using the input data
        # (may use different calculation method depending on specific indicator)
        self.values = self._calculate_full(data)

        # Keep track of how many data points we last processed
        # (some indicators may use this for optimization)
        if hasattr(data, '__len__'):
            self._last_data_length = len(data)
        else:
            self._last_data_length = 0
    
    def add_dependency(self, indicator: 'Indicator') -> None:
        """Add a dependency on another indicator."""
        if indicator.timeframe != self.timeframe:
            raise ValueError(f"Indicator timeframes must match: {self.timeframe} != {indicator.timeframe}")
        self.dependencies.append(indicator)
    
    def methods(self) -> None:
        """Display all labeled accessor methods available for this indicator."""
        methods_found = False
        
        # Header with indicator name, type, and timeframe
        print(f"Available methods for {format_method(self.__class__.__name__)} ({format_subheader(self.name)}) on timeframe {format_subheader(self.timeframe)}:")
        
        # Find and print all methods marked with @indicator_method decorator
        for attr_name in dir(self):
            attr = getattr(self, attr_name)
            if callable(attr) and hasattr(attr, '_is_indicator_method'):
                methods_found = True
                
                # Get the method signature as a string
                import inspect
                signature = str(inspect.signature(attr))
                
                # Print the method name and signature
                print(f"  {format_method(attr_name + signature)}")
                
                # Print the description if available
                if hasattr(attr, '_method_description') and attr._method_description:
                    # Wrap description text at 80 chars and indent properly
                    desc_lines = textwrap.wrap(attr._method_description, width=80)
                    for line in desc_lines:
                        print(f"    {format_description('Description: ' + line)}")
                
                # Print the return type if available
                if hasattr(attr, '_method_returns') and attr._method_returns:
                    # Wrap returns text at 80 chars and indent properly
                    returns_lines = textwrap.wrap(attr._method_returns, width=80)
                    for line in returns_lines:
                        print(f"    {format_description('Returns: ' + line)}")
                
                print()  # Empty line between methods
        
        if not methods_found:
            print(f"  {format_error('No labeled methods found for this indicator.')}")
            print(f"  {format_error('You can use the @indicator_method decorator to label methods.')}")
        
        # Print source information if available
        if hasattr(self, 'source') and self.source:
            print(f"\n{format_subheader('Source:')} {format_description(self.source)}")
        
        print(f"\n{format_subheader('Available components:')}")
        if self.COMPONENTS:
            for component in self.COMPONENTS:
                print(f"  - {format_component(component)}")
        else:
            print(f"  {format_error('No components defined')}")
        
        print(f"\n{format_description('Components can be accessed through:')}")
        print(f"  - {format_method(f'get(\"<component>\", offset)')}")
        print(f"  - {format_method(f'series(\"<component>\", length, start_pos)')}")
    
    def get_value(self, index: int = -1) -> Any:
        """Get indicator value at specific index (default: latest).
        
        This method handles index conversion to ensure consistent access pattern:
        - For index >= 0: 0 is the most recent value (maps to array[-1])
        - For index < 0: -1 is the most recent value (direct array indexing)
        
        Args:
            index: Index of the value to retrieve. 0 is most recent, -1 is previous, etc.
            
        Returns:
            The value at the specified index or None if not available
        """
        try:
            # First, check if we have a buffer-based implementation
            if hasattr(self, 'buffer') and self.buffer is not None:
                # If buffer is used, check if the requested index is available
                if not hasattr(self, 'is_filled') or not self.is_filled:
                    # Buffer is not fully filled yet, fall back to regular values array
                    pass
                else:
                    # Buffer is filled, use it for faster access
                    buffer_len = len(self.buffer)
                    
                    # Convert the index to buffer position
                    if index >= 0:
                        # 0 is the most recent, convert to position in circular buffer
                        buffer_idx = (self.position - 1 - index) % buffer_len
                    else:
                        # Negative indices count backward from most recent
                        buffer_idx = (self.position - 1 + index + 1) % buffer_len
                        
                    # Fetch the value from buffer
                    if not np.isnan(self.buffer[buffer_idx]):
                        return self.buffer[buffer_idx]
                
            # Check for _last_value first
            if hasattr(self, '_last_value'):
                if index == -1 or index == 0:
                    return self._last_value
                return None
            
            if not hasattr(self, 'values') or self.values is None:
                return None
            
            if hasattr(self.values, '__len__') and len(self.values) == 0:
                return None
                
            # Convert positive index to negative for proper access
            # For index >= 0, convert to corresponding negative index
            # This ensures 0 always means "most recent" entry
            if index >= 0:
                # Get the value from the end of the array
                # For index=0, get the last element
                # For index=1, get the second-to-last element, etc.
                actual_index = -1 - index
                
                # Ensure index is within bounds
                if abs(actual_index) > len(self.values):
                    return None
                    
                return self.values[actual_index]
            else:
                # For negative indices, directly access the array
                # Make sure index is in bounds
                if abs(index) > len(self.values):
                    return None
                    
                return self.values[index]
        except Exception as e:
            print(f"Error in get_value: {e}")
            return None
    
    def get(self, component: str = None, offset: int = 0) -> Any:
        """Get a specific component value at the given offset.
        
        Args:
            component: Component name to retrieve. If None, returns the full value.
            offset: Offset from the most recent candle. 0 is current, -1 is previous, etc.
            
        Returns:
            The component value or None if not available
        """
        try:
            # Get the full value at the specified offset
            value = self.get_value(offset)
            if value is None:
                return None
                
            # If no component specified, return the full value
            if component is None:
                return value
            
            # Handle dictionary values
            if isinstance(value, dict):
                if component in value:
                    return value[component]
                elif 'lines' in value and component in value['lines']:
                    return value['lines'][component]
                return None
            
            # Check if this component is defined in the class's COMPONENTS dictionary
            if hasattr(self.__class__, 'COMPONENTS') and component in self.__class__.COMPONENTS:
                component_index = self.__class__.COMPONENTS.get(component)
                
                # If component_index is None, assume this is a structured array field name
                if component_index is None:
                    # Handle structured arrays (numpy.void or dtype with names)
                    if hasattr(value, 'dtype') and hasattr(value.dtype, 'names') and value.dtype.names:
                        if component in value.dtype.names:
                            return value[component]
                        return None
                # Otherwise, use the index to access the component from a tuple/array
                elif hasattr(value, '__getitem__'):
                    if not hasattr(value, '__len__') or len(value) > component_index:
                        return value[component_index]
                    return None
                    
            # Handle structured arrays (like Bollinger Bands)
            if hasattr(value, 'dtype') and hasattr(value.dtype, 'names') and value.dtype.names:
                if component in value.dtype.names:
                    return value[component]
                return None
                
            # Handle list/array access using component as index
            if hasattr(value, '__getitem__'):
                # Try to convert component to an integer index if possible
                try:
                    idx = int(component)
                    if len(value) > idx:
                        return value[idx]
                except (ValueError, TypeError):
                    # If component is not a valid index, subclasses should handle it
                    pass
                    
            # For simple scalar values
            if not hasattr(value, '__len__'):
                return value
                
            return None
        except Exception as e:
            return None
    
    def series(self, component: str = None, length: int = 1, start_pos: int = 0) -> List[Any]:
        """Get a series of indicator values or components.
        
        Args:
            component: Component name to retrieve. If None, returns the full value.
            length: Number of values to return
            start_pos: Starting position. 0 is current, negative values are historical
            
        Returns:
            List of values with specified length
        """
        # Fast path for buffer-based retrieval when no component is specified
        if component is None and hasattr(self, 'buffer') and self.buffer is not None and hasattr(self, 'is_filled') and self.is_filled:
            buffer_len = len(self.buffer)
            result = []
            
            # Calculate end position
            end_pos = start_pos + length - 1
            
            # Retrieve values directly from buffer for better performance
            for i in range(start_pos, end_pos + 1):
                if i >= 0:
                    # 0 is the most recent, convert to position in circular buffer
                    buffer_idx = (self.position - 1 - i) % buffer_len
                else:
                    # Negative indices count backward from most recent
                    buffer_idx = (self.position - 1 + i + 1) % buffer_len
                
                # Get value from buffer
                value = self.buffer[buffer_idx]
                
                # Check if value is NaN - handle both simple and structured arrays
                is_nan_value = False
                if hasattr(value, 'dtype') and hasattr(value.dtype, 'names') and value.dtype.names:
                    # For structured arrays, check if all fields are NaN
                    is_nan_value = all(np.isnan(value[field]) for field in value.dtype.names if np.issubdtype(value[field].dtype, np.number))
                elif np.issubdtype(np.asarray(value).dtype, np.number):
                    # For simple numeric arrays or values
                    is_nan_value = np.isnan(value)
                
                if not is_nan_value:
                    result.append(value)
                else:
                    result.append(None)
                    
            return result
        
        # Standard path using get method for more complex cases
        result = []
        end_pos = start_pos + length - 1
        
        for i in range(start_pos, end_pos + 1):
            result.append(self.get(component, i))
            
        return result
    
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
        elif isinstance(value, dict):
            # Handle dictionary values
            result = []
            for k, v in value.items():
                if isinstance(v, dict):
                    # Handle nested dictionaries
                    for k2, v2 in v.items():
                        if isinstance(v2, float) or isinstance(v2, np.float64):
                            result.append((f"{k}.{k2}", f"{v2:.6f}"))
                        else:
                            result.append((f"{k}.{k2}", str(v2)))
                elif isinstance(v, float) or isinstance(v, np.float64):
                    result.append((k, f"{v:.6f}"))
                else:
                    result.append((k, str(v)))
            return result
        else:
            # Default formatting for other types
            return [("", str(value))]
    
    def __str__(self) -> str:
        if self.source == 'close':  # Only show source if it's not the default
            return f"{self.name}({self.timeframe})"
        return f"{self.name}[{self.source}]({self.timeframe})"

    def update_buffer(self, latest_value):
        """Update the circular buffer with the latest calculated value.
        
        Args:
            latest_value: The most recent calculated value to add to the buffer
        """
        if not hasattr(self, 'buffer') or self.buffer is None:
            # Create a default buffer if one doesn't exist
            buffer_capacity = 200  # Default size
            self.buffer = np.full(buffer_capacity, np.nan)
            self.position = 0
            self.is_filled = False
            
        # Store the value in the buffer
        self.buffer[self.position] = latest_value
        
        # Update the position (circular buffer)
        self.position = (self.position + 1) % len(self.buffer)
        
        # Mark as filled once we have filled the buffer once
        if not hasattr(self, 'is_filled') or not self.is_filled:
            if np.count_nonzero(~np.isnan(self.buffer)) >= len(self.buffer):
                self.is_filled = True
