"""
Tests for run artifacts writing.
"""

import json
import tempfile
import pytest
from pathlib import Path
import pandas as pd
from datetime import datetime

from index_options_bt.run.artifacts import RunArtifacts, generate_run_id


@pytest.fixture
def temp_run_dir():
    """Create a temporary run directory"""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


def test_generate_run_id_deterministic():
    """Test deterministic run ID generation"""
    config1 = {"data": {"provider": "sqlite"}, "engine": {"bar_size": "15s"}}
    config2 = {"data": {"provider": "sqlite"}, "engine": {"bar_size": "15s"}}
    
    run_id1 = generate_run_id(config1, mode="deterministic")
    run_id2 = generate_run_id(config2, mode="deterministic")
    
    # Should be the same for identical configs
    assert run_id1 == run_id2
    assert run_id1.startswith("run-")
    
    # Different config should give different ID
    config3 = {"data": {"provider": "sqlite"}, "engine": {"bar_size": "5s"}}
    run_id3 = generate_run_id(config3, mode="deterministic")
    assert run_id3 != run_id1


def test_generate_run_id_timestamp():
    """Test timestamp-based run ID generation"""
    config = {"data": {"provider": "sqlite"}}
    
    run_id = generate_run_id(config, mode="timestamp")
    
    assert run_id.startswith("run-")
    # Should have format: run-YYYYMMDD-HHMMSS-xxx
    parts = run_id.split("-")
    assert len(parts) >= 3


def test_run_artifacts_init(temp_run_dir):
    """Test RunArtifacts initialization"""
    config = {"data": {"provider": "sqlite"}, "engine": {"bar_size": "15s"}}
    run_id = "test-run-001"
    
    artifacts = RunArtifacts(temp_run_dir, run_id, config)
    
    assert artifacts.run_dir.exists()
    assert (artifacts.run_dir / "run.log").exists()
    
    artifacts.close()


def test_write_config_resolved(temp_run_dir):
    """Test writing resolved config"""
    config = {"data": {"provider": "sqlite"}, "engine": {"bar_size": "15s"}}
    run_id = "test-run-001"
    
    artifacts = RunArtifacts(temp_run_dir, run_id, config)
    
    # Write JSON config
    artifacts.write_config_resolved(format="json")
    assert (artifacts.run_dir / "config_resolved.json").exists()
    
    # Verify content
    with open(artifacts.run_dir / "config_resolved.json") as f:
        loaded = json.load(f)
        assert loaded["data"]["provider"] == "sqlite"
    
    artifacts.close()


def test_write_manifest(temp_run_dir):
    """Test writing manifest.json"""
    config = {"data": {"provider": "sqlite"}}
    run_id = "test-run-001"
    
    artifacts = RunArtifacts(temp_run_dir, run_id, config)
    
    metadata = {"total_trades": 10, "start": "2025-10-01"}
    artifacts.write_manifest(metadata)
    
    assert (artifacts.run_dir / "manifest.json").exists()
    
    # Verify content
    with open(artifacts.run_dir / "manifest.json") as f:
        manifest = json.load(f)
        assert manifest["run_id"] == run_id
        assert manifest["total_trades"] == 10
    
    artifacts.close()


def test_write_equity_curve(temp_run_dir):
    """Test writing equity_curve.csv"""
    config = {"data": {"provider": "sqlite"}}
    run_id = "test-run-001"
    
    artifacts = RunArtifacts(temp_run_dir, run_id, config)
    
    equity_curve = pd.DataFrame({
        "timestamp": [datetime(2025, 10, 1), datetime(2025, 10, 2)],
        "equity": [100000.0, 101000.0],
        "drawdown": [0.0, -0.01],
        "cumulative_return": [0.0, 0.01],
    })
    
    artifacts.write_equity_curve(equity_curve)
    
    assert (artifacts.run_dir / "equity_curve.csv").exists()
    
    # Verify content
    loaded = pd.read_csv(artifacts.run_dir / "equity_curve.csv")
    assert len(loaded) == 2
    assert "equity" in loaded.columns
    
    artifacts.close()


def test_write_trades(temp_run_dir):
    """Test writing trades.csv"""
    config = {"data": {"provider": "sqlite"}}
    run_id = "test-run-001"
    
    artifacts = RunArtifacts(temp_run_dir, run_id, config)
    
    trades = pd.DataFrame({
        "trade_id": [1, 2],
        "entry_time": [datetime(2025, 10, 1, 9, 15), datetime(2025, 10, 1, 10, 0)],
        "exit_time": [datetime(2025, 10, 1, 9, 30), datetime(2025, 10, 1, 10, 15)],
        "contract": ["NIFTY25NOV2526000CE", "NIFTY25NOV2526000PE"],
        "pnl": [100.0, -50.0],
    })
    
    artifacts.write_trades(trades)
    
    assert (artifacts.run_dir / "trades.csv").exists()
    
    # Verify content
    loaded = pd.read_csv(artifacts.run_dir / "trades.csv")
    assert len(loaded) == 2
    assert "pnl" in loaded.columns
    
    artifacts.close()


def test_write_metrics(temp_run_dir):
    """Test writing metrics.json"""
    config = {"data": {"provider": "sqlite"}}
    run_id = "test-run-001"
    
    artifacts = RunArtifacts(temp_run_dir, run_id, config)
    
    metrics = {
        "total_return": 10.5,
        "max_drawdown": -5.2,
        "sharpe": 1.8,
        "win_rate": 60.0,
    }
    
    artifacts.write_metrics(metrics)
    
    assert (artifacts.run_dir / "metrics.json").exists()
    
    # Verify content
    with open(artifacts.run_dir / "metrics.json") as f:
        loaded = json.load(f)
        assert loaded["total_return"] == 10.5
        assert loaded["sharpe"] == 1.8
    
    artifacts.close()


def test_write_selection_csv(temp_run_dir):
    """selection.csv should always be created (empty with headers allowed)."""
    config = {"data": {"provider": "sqlite"}}
    run_id = "test-run-001"
    artifacts = RunArtifacts(temp_run_dir, run_id, config)
    try:
        artifacts.write_selection(pd.DataFrame())
        assert (artifacts.run_dir / "selection.csv").exists()
        df = pd.read_csv(artifacts.run_dir / "selection.csv")
        assert "timestamp" in df.columns
        assert "selected_symbol" in df.columns
    finally:
        artifacts.close()

