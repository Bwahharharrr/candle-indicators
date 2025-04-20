#!/usr/bin/env python
"""
Helper utilities for candle_indicators.
"""

# Import UI components
from candle_indicators.utils.ui import (
    INFO, WARNING, SUCCESS, 
    COLOR_DIR, COLOR_TYPE, COLOR_DESC, COLOR_VAR, 
    Style, format_header
)

def display_indicators(indicators=None):
    """
    Display indicators with numbering and proper indentation.
    This function prints a formatted list of indicators
    with their descriptions, formats, examples, and parameters.
    
    Args:
        indicators: Optional dictionary of indicators to display.
                   If None, displays all registered indicators.
    """
    # Import here to avoid circular imports
    from candle_indicators import IndicatorRegistry
    
    # Get all available indicators with pretty formatting if not provided
    if indicators is None:
        indicators = IndicatorRegistry.get_all_indicators(pretty=True)
    
    # Display count and available indicators
    print(f"\n{INFO} {len(indicators)} Available Indicators:")
    
    # Display all indicators with numbering and proper indentation
    total_indicators = len(indicators)
    for idx, (prefix, info) in enumerate(indicators.items(), 1):
        indicator_class = info['class']
        
        # Format the header with indicator number, total count, and indicator name
        header = f"({idx}/{total_indicators}) {prefix} - {indicator_class.__name__} - {info['description']}"
        print(f"\n{format_header(header)}")
        
        # Get metadata from the info dictionary - already formatted
        format_str = info['format']
        examples = info['examples']
        params = info['params']
        
        # Display format string with indentation
        print(f"    {COLOR_VAR}Format:{Style.RESET_ALL} {format_str}")

        # Display parameters with indentation
        if params:
            print(f"    {COLOR_VAR}Parameters:{Style.RESET_ALL}")
            for param in params:
                print(f"      • {COLOR_TYPE}{param['name']}{Style.RESET_ALL}: {param['description']}")


        # Display examples with indentation
        if examples:
            print(f"    {COLOR_VAR}Examples:{Style.RESET_ALL}")
            for example in examples:
                print(f"      • {COLOR_DESC}{example}{Style.RESET_ALL}")
        
            
    print(f"\n{SUCCESS} Usage:")
    print(f"  {COLOR_DIR}indicator = IndicatorRegistry.create_indicator(Format, Timeframe, Source){Style.RESET_ALL}")
    print(f"{WARNING} Note: All indicators should be managed externally after creation")
