"""
Run artifacts: standardized output files for each backtest run.
"""

import json
import hashlib
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, Literal, Dict, Any
import pandas as pd

logger = logging.getLogger(__name__)


class RunArtifacts:
    """
    Manages run artifacts (output files) for a backtest run.
    
    Each run writes to: runs/<run_id>/
    - config_resolved.yaml|json
    - manifest.json
    - equity_curve.csv
    - trades.csv
    - positions.csv
    - metrics.json
    - report.png (optional)
    - run.log
    """
    
    def __init__(
        self,
        run_dir: Path,
        run_id: str,
        config: Dict[str, Any],
    ):
        """
        Initialize run artifacts writer.
        
        Args:
            run_dir: Root directory for runs (e.g., Path("runs"))
            run_id: Unique run ID (deterministic hash or timestamp-based)
            config: Resolved RunConfig as dictionary
        """
        self.run_dir = run_dir / run_id
        self.run_id = run_id
        self.config = config
        
        # Create run directory
        self.run_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging to file
        self.log_file = self.run_dir / "run.log"
        self._setup_logging()
    
    def _setup_logging(self):
        """Setup file logging for this run"""
        file_handler = logging.FileHandler(self.log_file, encoding="utf-8")
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        self._file_handler = file_handler
    
    def close(self):
        """Close file handlers"""
        if hasattr(self, "_file_handler"):
            logger.removeHandler(self._file_handler)
    
    def write_config_resolved(self, format: Literal["yaml", "json"] = "json"):
        """Write resolved configuration file"""
        if format == "yaml":
            try:
                import yaml
                with open(self.run_dir / "config_resolved.yaml", "w", encoding="utf-8") as f:
                    yaml.dump(self.config, f, default_flow_style=False, sort_keys=False)
            except ImportError:
                logger.warning("PyYAML not available, using JSON for config")
                format = "json"
        
        if format == "json":
            with open(self.run_dir / "config_resolved.json", "w", encoding="utf-8") as f:
                json.dump(self.config, f, indent=2, default=str)
    
    def write_manifest(self, metadata: Dict[str, Any]):
        """Write manifest.json with run metadata"""
        manifest = {
            "run_id": self.run_id,
            "created_at": datetime.utcnow().isoformat() + "Z",
            "config": self.config,
            **metadata,
        }
        
        with open(self.run_dir / "manifest.json", "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, default=str)
    
    def write_equity_curve(self, equity_curve: pd.DataFrame):
        """
        Write equity_curve.csv
        
        Expected columns: timestamp, equity, cash, market_value, drawdown, cumulative_return, unrealized_pnl, realized_pnl
        """
        if equity_curve.empty:
            equity_curve = pd.DataFrame(
                columns=[
                    "timestamp",
                    "equity",
                    "cash",
                    "market_value",
                    "unrealized_pnl",
                    "realized_pnl",
                    "drawdown",
                    "cumulative_return",
                    "delta",
                    "gamma",
                    "theta",
                    "vega",
                ]
            )
        
        equity_curve.to_csv(self.run_dir / "equity_curve.csv", index=False)
    
    def write_trades(self, trades: pd.DataFrame):
        """
        Write trades.csv
        
        Expected columns:
        trade_id, contract_id, symbol, entry_time, exit_time, qty, entry_price, exit_price, realized_pnl
        """
        if trades.empty:
            trades = pd.DataFrame(
                columns=[
                    "trade_id",
                    "contract_id",
                    "symbol",
                    "entry_time",
                    "exit_time",
                    "qty",
                    "entry_price",
                    "exit_price",
                    "realized_pnl",
                ]
            )
        
        trades.to_csv(self.run_dir / "trades.csv", index=False)
    
    def write_positions(self, positions: pd.DataFrame):
        """
        Write positions.csv (optional but preferred)
        
        Expected columns: timestamp, contract, quantity, entry_price, mtm_price, unrealized_pnl, ...
        """
        if positions.empty:
            positions = pd.DataFrame(
                columns=[
                    "timestamp",
                    "contract_id",
                    "symbol",
                    "qty",
                    "avg_price",
                    "mtm_price",
                    "unrealized_pnl",
                    "realized_pnl",
                    "hold_bars",
                ]
            )
        
        positions.to_csv(self.run_dir / "positions.csv", index=False)

    def write_selection(self, selection: pd.DataFrame) -> None:
        """
        Write selection.csv (debug contract selection decisions).

        Columns:
        timestamp, strategy, intent_direction, option_type, selected_symbol, side, qty,
        selector_mode, spot, expiry, strike, delta, bid, ask, mid, spread_pct, dte
        """
        if selection.empty:
            selection = pd.DataFrame(
                columns=[
                    "timestamp",
                    "strategy",
                    "intent_direction",
                    "option_type",
                    "selected_symbol",
                    "side",
                    "qty",
                    "selector_mode",
                    "spot",
                    "expiry",
                    "strike",
                    "delta",
                    "bid",
                    "ask",
                    "mid",
                    "spread_pct",
                    "dte",
                ]
            )
        selection.to_csv(self.run_dir / "selection.csv", index=False)
    
    def write_metrics(self, metrics: Dict[str, Any]):
        """Write metrics.json"""
        with open(self.run_dir / "metrics.json", "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, default=str)
    
    def write_report_plot(self, fig, format: str = "png"):
        """
        Write report plot (if matplotlib available)
        
        Args:
            fig: matplotlib figure object
            format: Image format (png, jpeg, etc.)
        """
        try:
            import matplotlib
            matplotlib.use("Agg")  # Non-interactive backend
            fig.savefig(self.run_dir / f"report.{format}", dpi=150, bbox_inches="tight")
            logger.info(f"Saved report plot: report.{format}")
        except ImportError:
            logger.warning("matplotlib not available, skipping plot generation")
        except Exception as e:
            logger.error(f"Failed to save plot: {e}")


def generate_run_id(
    config: Dict[str, Any],
    mode: Literal["deterministic", "timestamp"] = "timestamp",
) -> str:
    """
    Generate run ID.
    
    Args:
        config: Resolved RunConfig as dictionary
        mode: "deterministic" (hash of config) or "timestamp" (YYYYMMDD-HHMMSS-<suffix>)
        
    Returns:
        Run ID string
    """
    if mode == "deterministic":
        # Hash of canonical JSON representation
        config_json = json.dumps(config, sort_keys=True, default=str)
        hash_obj = hashlib.sha256(config_json.encode("utf-8"))
        hash_hex = hash_obj.hexdigest()[:12]  # 12 chars
        return f"run-{hash_hex}"
    
    elif mode == "timestamp":
        # Timestamp-based: YYYYMMDD-HHMMSS-<suffix>
        now = datetime.utcnow()
        timestamp_str = now.strftime("%Y%m%d-%H%M%S")
        # Add short random suffix to avoid collisions
        import random
        suffix = random.randint(100, 999)
        return f"run-{timestamp_str}-{suffix}"
    
    else:
        raise ValueError(f"Invalid run_id_mode: {mode}")

