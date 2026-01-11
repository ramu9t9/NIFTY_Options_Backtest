"""
Tests for config loader with overrides.
"""

import json
import os
import tempfile
import pytest
from pathlib import Path

from index_options_bt.config import load_config, apply_env_overrides, apply_cli_overrides, RunConfig


@pytest.fixture
def temp_config_json():
    """Create a temporary JSON config file"""
    config_dict = {
        "data": {
            "provider": "sqlite",
            "sqlite_path": "/tmp/test.db",
            "table": "ltp_ticks",
            "symbol_index": "NIFTY 50",
        },
        "engine": {
            "start": "2025-10-01",
            "end": "2025-10-15",
            "bar_size": "15s",
        },
        "execution": {
            "fill_on": "bidask",
            "slippage_bps": 2.0,
        },
        "strategy": {
            "name": "breakout",
            "params": {
                "lookback_bars": 20,
            },
        },
    }
    
    fd, path = tempfile.mkstemp(suffix=".json")
    with open(fd, "w") as f:
        json.dump(config_dict, f)
    
    yield path
    
    Path(path).unlink()


def test_load_config_json(temp_config_json):
    """Test loading JSON config"""
    config = load_config(temp_config_json)
    assert isinstance(config, RunConfig)
    assert config.data.provider == "sqlite"
    assert config.data.sqlite_path == "/tmp/test.db"
    assert config.engine.bar_size == "15s"


def test_load_config_yaml():
    """Test loading YAML config (if PyYAML available)"""
    try:
        import yaml
    except ImportError:
        pytest.skip("PyYAML not available")
    
    config_dict = {
        "data": {
            "provider": "sqlite",
            "sqlite_path": "/tmp/test.db",
        },
        "engine": {
            "start": "2025-10-01",
            "end": "2025-10-15",
        },
        "strategy": {
            "name": "breakout",
            "params": {},
        },
    }
    
    fd, path = tempfile.mkstemp(suffix=".yaml")
    with open(fd, "w") as f:
        yaml.dump(config_dict, f)
    
    try:
        config = load_config(path)
        assert isinstance(config, RunConfig)
    finally:
        Path(path).unlink()


def test_apply_env_overrides(temp_config_json):
    """Test environment variable overrides"""
    # Set environment variable
    os.environ["IOBT__execution__slippage_bps"] = "5"
    
    try:
        config = load_config(temp_config_json)
        config = apply_env_overrides(config)
        
        # Check override was applied
        assert config.execution.slippage_bps == 5
        
    finally:
        # Cleanup
        os.environ.pop("IOBT__execution__slippage_bps", None)


def test_apply_cli_overrides(temp_config_json):
    """Test CLI --set overrides"""
    config = load_config(temp_config_json)
    
    # Apply CLI override
    config = apply_cli_overrides(config, ["execution.slippage_bps=10"])
    
    # Check override was applied
    assert config.execution.slippage_bps == 10
    
    # Test nested override
    config = apply_cli_overrides(config, ["selector.liquidity.min_bid=2.0"])
    assert config.selector.liquidity.min_bid == 2.0


def test_apply_cli_overrides_typed(temp_config_json):
    """Test that CLI overrides parse types correctly"""
    config = load_config(temp_config_json)
    
    # Integer
    config = apply_cli_overrides(config, ["risk.max_contracts=8"])
    assert config.risk.max_contracts == 8
    assert isinstance(config.risk.max_contracts, int)
    
    # Float
    config = apply_cli_overrides(config, ["execution.slippage_bps=3.5"])
    assert config.execution.slippage_bps == 3.5
    assert isinstance(config.execution.slippage_bps, float)
    
    # Boolean
    config = apply_cli_overrides(config, ["execution.partial_fills=true"])
    assert config.execution.partial_fills is True
    
    config = apply_cli_overrides(config, ["execution.partial_fills=false"])
    assert config.execution.partial_fills is False
    
    # JSON parsing
    config = apply_cli_overrides(config, ['selector.liquidity={"min_bid": 5.0, "max_spread_pct": 0.25}'])
    assert config.selector.liquidity.min_bid == 5.0
    assert config.selector.liquidity.max_spread_pct == 0.25

