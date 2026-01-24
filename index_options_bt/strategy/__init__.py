"""
Strategy module with deterministic discovery via explicit imports.

Strategies are registered via @register_strategy decorator when modules are imported.
This __init__.py imports all known strategy modules to ensure registration happens.
"""

# Import strategy modules to trigger registration
# Add new strategy modules here as they are implemented

try:
    from .example_breakout import ExampleBreakoutStrategy
except ImportError:
    # Strategy not implemented yet
    pass

try:
    from .ema_crossover import EMACrossoverStrategy
except ImportError:
    # Strategy not implemented yet
    pass

try:
    from .nifty_micro_scalp import NiftyMicroScalpStrategy
except ImportError:
    # Strategy not implemented yet
    pass

try:
    from .pre_rally_pattern import PreRallyPatternStrategy
except ImportError:
    # Strategy not implemented yet
    pass

try:
    from .strangle import ShortStrangleStrategy
except ImportError:
    # Strategy not implemented yet
    pass

try:
    from .oi_momentum import OIMomentumStrategy
except ImportError:
    # Strategy not implemented yet
    pass

try:
    from .institutional_flow_momentum import InstitutionalFlowMomentumStrategy
except ImportError:
    # Strategy not implemented yet
    pass

# Export registry functions
from .registry import (
    register_strategy,
    list_strategies,
    get_strategy,
    discover_strategies,
)

# Export base classes
from .base import Strategy, Intent

__all__ = [
    "Strategy",
    "Intent",
    "register_strategy",
    "list_strategies",
    "get_strategy",
    "discover_strategies",
]

