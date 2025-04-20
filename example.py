#!/usr/bin/env python
"""
Simple example script that imports and initiates the candle_indicators module.
This script demonstrates how to import and use the candle_indicators package.
"""

# Standard library imports
import sys
import argparse
import numpy as np 
from colorama import Fore
from pprint import pprint

# Import the candle_indicators module
import candle_indicators
from candle_indicators import Indicator, IndicatorRegistry
from candle_indicators.utils.helpers import display_indicators
from candle_indicators.utils.ui import (
    INFO, WARNING, SUCCESS, ERROR,
    COLOR_DIR, COLOR_TYPE, COLOR_DESC, COLOR_VAR, COLOR_REQ, COLOR_FILE, COLOR_NEW, COLOR_HELPER,
    Style, format_header
)
from candle_iterator import create_candle_iterator

# Custom help formatter to suppress the "options:" header
class CustomHelpFormatter(argparse.RawTextHelpFormatter):
    def _format_action_invocation(self, action):
        if not action.option_strings:
            default = self._get_default_metavar_for_positional(action)
            metavar, = self._metavar_formatter(action, default)(1)
            return metavar
        else:
            return ', '.join(action.option_strings)
            
    def _format_usage(self, usage, actions, groups, prefix):
        return ""  # Skip usage section
        
    def _format_action(self, action):
        if isinstance(action, argparse._HelpAction):
            return ""  # Skip help entry
        return super()._format_action(action)
        
    def _fill_text(self, text, width, indent):
        return text  # Don't fill or wrap text
        
    def _split_lines(self, text, width):
        return text.splitlines()  # Don't split lines

# ----------------------------------------------------------------------
# 6) MAIN
# ----------------------------------------------------------------------
def parse_args():
    """
    Parses command-line arguments with a color-coded help output.
    """
    # Define column widths for perfect alignment
    arg_width = 25
    type_width = 10
    req_width = 15
    
    # Format each line with consistent spacing
    arg_lines = [
        f"  {COLOR_VAR}--show-indicators{Style.RESET_ALL}{' ' * (arg_width - 18)}{COLOR_TYPE}(flag){Style.RESET_ALL}{' ' * (type_width - 6)}{COLOR_HELPER}{' ' * (req_width - 9)} {COLOR_DESC}Display all available indicators and exit{Style.RESET_ALL}",
        f"  {COLOR_VAR}--exchange{Style.RESET_ALL}{' ' * (arg_width - 11)}{COLOR_TYPE}(str){Style.RESET_ALL}{' ' * (type_width - 5)}{COLOR_REQ}{' ' * (req_width - 11)} {COLOR_DESC}Exchange name (e.g., BITFINEX){Style.RESET_ALL}",
        f"  {COLOR_VAR}--ticker{Style.RESET_ALL}{' ' * (arg_width - 9)}{COLOR_TYPE}(str){Style.RESET_ALL}{' ' * (type_width - 5)}{COLOR_REQ}{' ' * (req_width - 11)} {COLOR_DESC}Trading pair (e.g., tBTCUSD){Style.RESET_ALL}",
        f"  {COLOR_VAR}--timeframe{Style.RESET_ALL}{' ' * (arg_width - 12)}{COLOR_TYPE}(str){Style.RESET_ALL}{' ' * (type_width - 5)}{COLOR_REQ}{' ' * (req_width - 11)} {COLOR_DESC}Base timeframe (e.g., 1m, 1h, 1D){Style.RESET_ALL}",
        f"  {COLOR_VAR}--start{Style.RESET_ALL}{' ' * (arg_width - 8)}{COLOR_TYPE}(str){Style.RESET_ALL}{' ' * (type_width - 5)}{' ' * req_width} {COLOR_DESC}Start date (YYYY-MM-DD or YYYY-MM-DD HH:MM){Style.RESET_ALL}",
        f"  {COLOR_VAR}--end{Style.RESET_ALL}{' ' * (arg_width - 6)}{COLOR_TYPE}(str){Style.RESET_ALL}{' ' * (type_width - 5)}{' ' * req_width} {COLOR_DESC}End date (YYYY-MM-DD or YYYY-MM-DD HH:MM){Style.RESET_ALL}",
        f"  {COLOR_VAR}--data-dir{Style.RESET_ALL}{' ' * (arg_width - 11)}{COLOR_TYPE}(str){Style.RESET_ALL}{' ' * (type_width - 5)}{' ' * req_width} {COLOR_DESC}Base directory for candle data (default: ~/.corky){Style.RESET_ALL}",
    ]
    
    # Create the full description with aligned arguments and proper line breaks
    description = f"""
{INFO} Candle Indicators Example Script {Style.RESET_ALL}

{"\n".join(arg_lines)}

{INFO} Example Usage: {Style.RESET_ALL}
  {COLOR_FILE}python example.py --show-indicators{Style.RESET_ALL}
  {COLOR_FILE}python example.py --exchange BITFINEX --ticker tBTCUSD --timeframe 1h{Style.RESET_ALL}
  {COLOR_FILE}python example.py --exchange BITFINEX --ticker tBTCUSD --timeframe 1h --start "2024-01-01" --end "2024-02-01"{Style.RESET_ALL}
"""
    
    # Create parser with custom formatter, but don't add the help automatically
    parser = argparse.ArgumentParser(
        description=description,
        formatter_class=CustomHelpFormatter,
        add_help=False  # Don't add the help option automatically
    )
    
    # Add custom help option that just prints our description and exits
    parser.add_argument('-h', '--help', action='store_true', help=argparse.SUPPRESS)
    
    parser.add_argument("--show-indicators", action="store_true", 
                      help="Display all available indicators and exit")
    parser.add_argument("--exchange", help="Exchange name (e.g., BITFINEX)")
    parser.add_argument("--ticker", help="Trading pair (e.g., tBTCUSD)")
    parser.add_argument("--timeframe", help="Base timeframe (e.g., 1h)")
    parser.add_argument("--start", help="Start date (YYYY-MM-DD or YYYY-MM-DD HH:MM)")
    parser.add_argument("--end", help="End date (YYYY-MM-DD or YYYY-MM-DD HH:MM)")
    parser.add_argument("--data-dir", default="~/.corky",
                      help="Base directory for candle data")
    
    # Parse args
    args = parser.parse_args()
    
    # Handle help argument manually
    if args.help or len(sys.argv) == 1:
        print(description)
        sys.exit(0)
    
    # If show_indicators is set, we don't need the other required arguments
    if not args.show_indicators:
        # Check for required arguments
        if not all([args.exchange, args.ticker, args.timeframe]):
            print(f"\n{ERROR} Missing required parameters! Please specify all required parameters.\n")
            print(description)
            sys.exit(1)
        
    return args

def main():
    """
    Main function that demonstrates the initialization of candle_indicators.
    """
    # Parse command-line arguments
    args = parse_args()
    
    # Get all available indicators with pretty formatting
    indicators = IndicatorRegistry.get_all_indicators(pretty=True)

    # Check if we should only display indicators
    if args.show_indicators:
        # Display all available indicators
        display_indicators(indicators)
        return
     
    # 1 
    # sma = IndicatorRegistry.create_indicator(f"SMA20:{args.timeframe}:close")

    # 2
    # bbands = IndicatorRegistry.create_indicator(f"BB20:{args.timeframe}:close")

    # 3
    # cwema = IndicatorRegistry.create_indicator(f"CWEMA20:{args.timeframe}:close")

    # 4 
    # ema = IndicatorRegistry.create_indicator(f"EMA20:{args.timeframe}:close")

    # 5
    # macd = IndicatorRegistry.create_indicator(f"MACD12,26,9:{args.timeframe}:close")

    # 6 
    rsi = IndicatorRegistry.create_indicator(f"RSI14:{args.timeframe}:close")

        
    try:
        for closure in create_candle_iterator(
            exchange=args.exchange,
            ticker=args.ticker,
            base_timeframe=args.timeframe,
            aggregation_timeframes=[args.timeframe],
            start_date=args.start,
            end_date=args.end,
            data_dir=args.data_dir
        ):
            # Print the closure information
            closure.print()
            
            # Get the candle for the specified timeframe
            candle = closure.get_candle(args.timeframe)

            # 1 
            # sma.update_with_data(np.array([candle.close]))
            # print(sma.series(None,5))
            
            # 2 
            # bbands.update_with_data(np.array([candle.close]))

            #3 
            # cwema.update_with_data(np.array([candle.close]))

            # 4
            # ema.update_with_data(np.array([candle.close]))

            # 5
            # macd.update_with_data(np.array([candle.close]))

            # 6
            # rsi.update_with_data(np.array([candle.close]))

            # print(rsi.series(None,5))
            
    except ValueError as e:
        print(f"{ERROR} {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
