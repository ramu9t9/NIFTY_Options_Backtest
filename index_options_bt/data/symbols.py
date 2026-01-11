"""
Symbol parsing utilities for NIFTY options, futures, and index symbols.

Supports:
- INDEX: "NIFTY 50"
- FUT: NIFTY{DDMMMYY}FUT (e.g., NIFTY28NOV25FUT)
- OPT: NIFTY{DDMMMYY}{STRIKE}CE/PE (e.g., NIFTY25NOV2526000CE)
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Literal
import re

# Month abbreviations mapping
_MONTH_MAP = {
    "JAN": 1, "FEB": 2, "MAR": 3, "APR": 4, "MAY": 5, "JUN": 6,
    "JUL": 7, "AUG": 8, "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12,
}


@dataclass
class InstrumentId:
    """Parsed instrument identifier"""
    kind: Literal["INDEX", "FUT", "OPT"]
    underlying: str  # "NIFTY"
    expiry: Optional[datetime] = None  # date only (no time)
    strike: Optional[int] = None
    cp: Optional[Literal["C", "P"]] = None  # Call or Put
    original_symbol: str = ""


def parse_symbol(symbol: str) -> InstrumentId:
    """
    Parse a NIFTY symbol into its components.
    
    Args:
        symbol: Symbol string (e.g., "NIFTY 50", "NIFTY28NOV25FUT", "NIFTY25NOV2526000CE")
        
    Returns:
        InstrumentId dataclass with parsed fields
        
    Examples:
        >>> parse_symbol("NIFTY 50")
        InstrumentId(kind='INDEX', underlying='NIFTY', ...)
        
        >>> parse_symbol("NIFTY28NOV25FUT")
        InstrumentId(kind='FUT', underlying='NIFTY', expiry=datetime(2025, 11, 28), ...)
        
        >>> parse_symbol("NIFTY25NOV2526000CE")
        InstrumentId(kind='OPT', underlying='NIFTY', expiry=datetime(2025, 11, 25), strike=26000, cp='C', ...)
    """
    symbol = symbol.strip()
    original = symbol
    
    # Check for index symbol (exact match)
    if symbol == "NIFTY 50":
        return InstrumentId(
            kind="INDEX",
            underlying="NIFTY",
            original_symbol=original,
        )
    
    # Parse future: NIFTY{DDMMMYY}FUT
    fut_match = re.match(r"^NIFTY(\d{2})([A-Z]{3})(\d{2})FUT$", symbol)
    if fut_match:
        day_str, month_str, year_str = fut_match.groups()
        day = int(day_str)
        month = _MONTH_MAP.get(month_str.upper())
        year = 2000 + int(year_str)
        
        if month is None:
            raise ValueError(f"Invalid month in future symbol: {symbol}")
        
        try:
            expiry = datetime(year, month, day).date()
            return InstrumentId(
                kind="FUT",
                underlying="NIFTY",
                expiry=datetime(expiry.year, expiry.month, expiry.day),
                original_symbol=original,
            )
        except ValueError as e:
            raise ValueError(f"Invalid expiry date in future symbol: {symbol}") from e
    
    # Parse option: NIFTY{DDMMMYY}{STRIKE}CE/PE
    opt_match = re.match(r"^NIFTY(\d{2})([A-Z]{3})(\d{2})(\d+)CE$", symbol)
    if opt_match:
        day_str, month_str, year_str, strike_str = opt_match.groups()
        day = int(day_str)
        month = _MONTH_MAP.get(month_str.upper())
        year = 2000 + int(year_str)
        strike = int(strike_str)
        
        if month is None:
            raise ValueError(f"Invalid month in option symbol: {symbol}")
        
        try:
            expiry = datetime(year, month, day).date()
            return InstrumentId(
                kind="OPT",
                underlying="NIFTY",
                expiry=datetime(expiry.year, expiry.month, expiry.day),
                strike=strike,
                cp="C",
                original_symbol=original,
            )
        except ValueError as e:
            raise ValueError(f"Invalid expiry date in option symbol: {symbol}") from e
    
    opt_match = re.match(r"^NIFTY(\d{2})([A-Z]{3})(\d{2})(\d+)PE$", symbol)
    if opt_match:
        day_str, month_str, year_str, strike_str = opt_match.groups()
        day = int(day_str)
        month = _MONTH_MAP.get(month_str.upper())
        year = 2000 + int(year_str)
        strike = int(strike_str)
        
        if month is None:
            raise ValueError(f"Invalid month in option symbol: {symbol}")
        
        try:
            expiry = datetime(year, month, day).date()
            return InstrumentId(
                kind="OPT",
                underlying="NIFTY",
                expiry=datetime(expiry.year, expiry.month, expiry.day),
                strike=strike,
                cp="P",
                original_symbol=original,
            )
        except ValueError as e:
            raise ValueError(f"Invalid expiry date in option symbol: {symbol}") from e
    
    raise ValueError(f"Unrecognized symbol format: {symbol}")


def is_option_symbol(symbol: str) -> bool:
    """Check if symbol is an option"""
    try:
        parsed = parse_symbol(symbol)
        return parsed.kind == "OPT"
    except ValueError:
        return False


def is_future_symbol(symbol: str) -> bool:
    """Check if symbol is a future"""
    try:
        parsed = parse_symbol(symbol)
        return parsed.kind == "FUT"
    except ValueError:
        return False


def is_index_symbol(symbol: str) -> bool:
    """Check if symbol is the index"""
    try:
        parsed = parse_symbol(symbol)
        return parsed.kind == "INDEX"
    except ValueError:
        return False


def option_contract_id(symbol: str) -> str:
    """
    Generate a stable contract identifier for an option symbol.
    Useful for portfolio/position keys.
    
    Args:
        symbol: Option symbol string
        
    Returns:
        Stable contract ID string (e.g., "NIFTY-2025-11-25-26000-C")
    """
    parsed = parse_symbol(symbol)
    if parsed.kind != "OPT":
        raise ValueError(f"Not an option symbol: {symbol}")
    
    if parsed.expiry is None or parsed.strike is None or parsed.cp is None:
        raise ValueError(f"Incomplete option symbol: {symbol}")
    
    expiry_str = parsed.expiry.strftime("%Y-%m-%d")
    return f"{parsed.underlying}-{expiry_str}-{parsed.strike}-{parsed.cp}"

