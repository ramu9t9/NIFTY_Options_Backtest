"""
Tests for strategy registry.
"""

import pytest
from typing import Dict, Any

from index_options_bt.strategy.base import Strategy, Intent
from index_options_bt.strategy.registry import (
    register_strategy,
    list_strategies,
    get_strategy,
    discover_strategies,
)
from index_options_bt.data.models import MarketSnapshot
from datetime import datetime, timezone


class MockStrategy(Strategy):
    """Mock strategy for testing"""
    
    def generate_intents(self, snapshot: MarketSnapshot, context: Dict[str, Any] = None):
        return []


def test_register_strategy():
    """Test strategy registration"""
    # Clear registry (in real usage, strategies register on import)
    from index_options_bt.strategy import registry
    original_registry = registry._strategy_registry.copy()
    
    try:
        # Register a test strategy
        @register_strategy("test_strategy")
        class TestStrategy(MockStrategy):
            pass
        
        # Check it's registered
        strategies = list_strategies()
        assert "test_strategy" in strategies
        
        # Get it
        strategy = get_strategy("test_strategy", {})
        assert isinstance(strategy, TestStrategy)
        
    finally:
        # Restore original registry
        registry._strategy_registry.clear()
        registry._strategy_registry.update(original_registry)


def test_get_strategy_unknown():
    """Test that unknown strategy raises friendly error"""
    with pytest.raises(ValueError, match="Unknown strategy"):
        get_strategy("nonexistent_strategy", {})


def test_list_strategies():
    """Test listing strategies"""
    strategies = list_strategies()
    assert isinstance(strategies, list)
    # May be empty if no strategies registered, which is fine


def test_discover_strategies():
    """Test strategy discovery"""
    # Should not raise
    discover_strategies()

