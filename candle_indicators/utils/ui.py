"""
Color and UI utilities for candle_indicators.
"""

# ----------------------------------------------------------------------
# 1) COLOR & LOGGING SETUP
# ----------------------------------------------------------------------
try:
    import colorama
    from colorama import Fore, Style
    colorama.init(autoreset=True)
except ImportError:
    class Fore:
        CYAN = ""
        GREEN = ""
        YELLOW = ""
        RED = ""
        MAGENTA = ""
        WHITE = ""
        BLUE = ""  # Adding this for completeness
        RESET = ""
    class Style:
        RESET_ALL = ""
        BRIGHT = ""
        DIM = ""
        NORMAL = ""

INFO = Fore.GREEN + "[INFO]" + Style.RESET_ALL
WARNING = Fore.YELLOW + "[WARNING]" + Style.RESET_ALL
ERROR = Fore.RED + "[ERROR]" + Style.RESET_ALL
SUCCESS = Fore.GREEN + "[SUCCESS]" + Style.RESET_ALL
UPDATE = Fore.MAGENTA + "[UPDATE]" + Style.RESET_ALL

# Additional color definitions for CLI output
COLOR_DIR = Fore.CYAN
COLOR_FILE = Fore.YELLOW
COLOR_TIMESTAMPS = Fore.MAGENTA
COLOR_ROWS = Fore.RED
COLOR_NEW = Fore.WHITE
COLOR_VAR = Fore.CYAN
COLOR_TYPE = Fore.YELLOW
COLOR_DESC = Fore.MAGENTA
COLOR_REQ = Fore.RED + "[REQUIRED]" + Style.RESET_ALL
COLOR_HELPER = Fore.GREEN + "[HELPER]" + Style.RESET_ALL

# Formatting functions previously in utils/ui/formatting.py
def format_header(text):
    return f"{Style.BRIGHT}{Fore.CYAN}{text}{Style.RESET_ALL}"

def format_subheader(text):
    return f"{Style.BRIGHT}{Fore.BLUE}{text}{Style.RESET_ALL}"

def format_method(text):
    return f"{Style.BRIGHT}{Fore.GREEN}{text}{Style.RESET_ALL}"

def format_description(text):
    return f"{Fore.WHITE}{text}{Style.RESET_ALL}"

def format_error(text):
    return f"{Style.BRIGHT}{Fore.RED}{text}{Style.RESET_ALL}"

def format_component(text):
    return f"{Fore.YELLOW}{text}{Style.RESET_ALL}"

def format_value(text):
    return f"{Fore.MAGENTA}{text}{Style.RESET_ALL}"

def format_example_header(text):
    return f"{Style.BRIGHT}{Fore.CYAN}{text}{Style.RESET_ALL}"
