"""
Strategy registry for pluggable strategies.

Strategies register themselves using @register_strategy decorator.
Registry provides lookup and instantiation with friendly error messages.
"""

from typing import Dict, Type, List, Optional, Any
import logging

from .base import Strategy

logger = logging.getLogger(__name__)

# Global registry: name -> Strategy class
_strategy_registry: Dict[str, Type[Strategy]] = {}


def register_strategy(name: str):
    """
    Decorator to register a strategy class.
    
    Args:
        name: Strategy name (must be unique)
        
    Example:
        @register_strategy("breakout")
        class BreakoutStrategy(Strategy):
            ...
    """
    def decorator(cls: Type[Strategy]):
        if name in _strategy_registry:
            logger.warning(f"Strategy '{name}' is already registered. Overwriting.")
        _strategy_registry[name] = cls
        logger.debug(f"Registered strategy: {name} -> {cls.__name__}")
        return cls
    return decorator


def list_strategies() -> List[str]:
    """
    List all registered strategy names.
    
    Returns:
        List of strategy names
    """
    return sorted(_strategy_registry.keys())


def get_strategy(name: str, params: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> Strategy:
    """
    Get a strategy instance by name.
    
    Args:
        name: Strategy name (must be registered)
        params: Strategy parameters from config
        context: Optional context (for strategies that need initialization state)
        
    Returns:
        Strategy instance
        
    Raises:
        ValueError: If strategy name is unknown
    """
    if name not in _strategy_registry:
        available = list_strategies()
        if available:
            available_str = ", ".join(available)
            raise ValueError(
                f"Unknown strategy: '{name}'. Available strategies: {available_str}\n"
                f"Make sure the strategy module is imported (add to index_options_bt/strategy/__init__.py)"
            )
        else:
            raise ValueError(
                f"Unknown strategy: '{name}'. No strategies are registered.\n"
                f"Make sure strategy modules are imported (add to index_options_bt/strategy/__init__.py)"
            )
    
    strategy_cls = _strategy_registry[name]
    
    try:
        return strategy_cls(params)
    except Exception as e:
        raise ValueError(f"Failed to instantiate strategy '{name}': {e}") from e


def discover_strategies():
    """
    Discover strategies by importing known strategy modules.
    
    This is called automatically when the strategy package is imported,
    but can be called manually to trigger discovery.
    """
    # Import strategy package to trigger __init__.py imports
    from . import __init__ as strategy_init  # noqa: F401
    
    logger.info(f"Discovered {len(_strategy_registry)} strategies: {list_strategies()}")

