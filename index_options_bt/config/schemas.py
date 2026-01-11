"""
Configuration schemas using Pydantic for validation and type safety.
"""

from datetime import datetime
from typing import Optional, Literal, Dict, Any
from pathlib import Path
from pydantic import BaseModel, Field, field_validator
from zoneinfo import ZoneInfo


class DataConfig(BaseModel):
    """Data source configuration"""
    provider: Literal["sqlite", "csv"] = Field(default="sqlite", description="Data provider type")
    sqlite_path: Optional[str] = Field(default=None, description="Path to SQLite database")
    table: str = Field(default="ltp_ticks", description="Table name in database")
    symbol_index: str = Field(default="NIFTY 50", description="Index symbol name")
    
    # CSV provider fields (optional)
    csv_spot_path: Optional[str] = Field(default=None, description="Path to spot CSV file")
    csv_options_path: Optional[str] = Field(default=None, description="Path to options chain CSV file")
    
    @field_validator("sqlite_path", mode="before")
    @classmethod
    def validate_sqlite_path(cls, v, info):
        """Validate sqlite_path is provided if provider is sqlite"""
        if info.data.get("provider") == "sqlite" and not v:
            raise ValueError("sqlite_path is required when provider='sqlite'")
        return v


class EngineConfig(BaseModel):
    """Engine configuration (single source of truth for bar_size)"""
    start: str = Field(description="Start date (ISO format, e.g., '2025-10-01')")
    end: str = Field(description="End date (ISO format, e.g., '2025-10-15')")
    bar_size: str = Field(default="15s", description="Bar size frequency (e.g., '5s', '15s', '30s', '1min')")
    tz_display: str = Field(default="Asia/Kolkata", description="Display timezone")
    normalize_to_utc: bool = Field(default=True, description="Normalize timestamps to UTC internally")
    
    # Session settings (optional, defaults used if not specified)
    session_start_ist: Optional[str] = Field(default="09:15", description="Session start time in IST (HH:MM)")
    session_end_ist: Optional[str] = Field(default="15:30", description="Session end time in IST (HH:MM)")
    
    @field_validator("start", "end", mode="before")
    @classmethod
    def validate_date_string(cls, v):
        """Ensure start/end are strings (keep as string, don't convert to datetime)"""
        if isinstance(v, datetime):
            # If already a datetime, convert back to string
            return v.strftime("%Y-%m-%d")
        if isinstance(v, str):
            return v
        raise ValueError(f"Invalid date format: {v}. Expected ISO date string (YYYY-MM-DD)")
    
    @field_validator("bar_size", mode="before")
    @classmethod
    def validate_bar_size(cls, v):
        """Validate bar_size is a valid pandas frequency"""
        valid_suffixes = ["s", "sec", "min", "h", "H"]
        if not any(v.endswith(s) for s in valid_suffixes):
            raise ValueError(f"Invalid bar_size: {v}. Must be pandas frequency (e.g., '5s', '15s', '1min')")
        return v
    
    def get_start_datetime(self) -> datetime:
        """Get start datetime in UTC"""
        if isinstance(self.start, str):
            dt = datetime.fromisoformat(self.start.replace("Z", "+00:00"))
        else:
            dt = self.start
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=ZoneInfo(self.tz_display))
        if self.normalize_to_utc:
            return dt.astimezone(ZoneInfo("UTC"))
        return dt
    
    def get_end_datetime(self) -> datetime:
        """Get end datetime in UTC"""
        if isinstance(self.end, str):
            try:
                dt = datetime.fromisoformat(self.end.replace("Z", "+00:00"))
            except ValueError:
                dt = datetime.strptime(self.end, "%Y-%m-%d")
                dt = dt.replace(tzinfo=ZoneInfo(self.tz_display))
        else:
            dt = self.end
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=ZoneInfo(self.tz_display))
        if self.normalize_to_utc:
            return dt.astimezone(ZoneInfo("UTC"))
        return dt


class ExecutionConfig(BaseModel):
    """Execution model configuration"""
    fill_on: Literal["bidask", "mid"] = Field(default="bidask", description="Fill price method")
    fallback_price: Literal["mid", "ltp"] = Field(default="mid", description="Fallback price if bid/ask missing")
    slippage_bps: float = Field(default=2.0, description="Slippage in basis points (1bp = 0.01%)")
    commission_per_contract: float = Field(default=0.0, description="Commission per contract")
    fees_per_contract: float = Field(default=0.0, description="Exchange fees per contract")
    partial_fills: bool = Field(default=False, description="Allow partial fills")


class StrikeWindow(BaseModel):
    """Strike window around spot."""

    kind: Literal["count", "pct"] = Field(default="count", description="count=±N strikes, pct=±pct around spot")
    value: float = Field(default=5, description="count: N strikes; pct: proportion (0.02=2%)")


class LiquidityFilters(BaseModel):
    """Liquidity filters for selection."""

    min_bid: float = 1.0
    max_spread_pct: float = 0.20
    min_oi: float = 0
    min_volume: float = 0


class SelectorConfig(BaseModel):
    """Contract selector configuration"""
    mode: Literal["atm", "delta", "dte"] = Field(default="atm", description="Selection mode")
    expiry_preference: Literal["weekly", "monthly", "nearest"] = Field(default="weekly", description="Expiry preference")
    contract_multiplier: int = Field(default=1, description="Contract multiplier / lot size (e.g., 50 for NIFTY options)")
    strike_window: Optional[StrikeWindow] = Field(default_factory=StrikeWindow, description="Strike window filter")
    target_dte: Optional[int] = Field(default=None, description="Target days to expiry (for dte mode)")
    target_delta: Optional[float] = Field(default=None, description="Target delta (for delta mode, e.g., 0.25)")
    liquidity: LiquidityFilters = Field(default_factory=LiquidityFilters, description="Liquidity filters")


class RiskConfig(BaseModel):
    """Risk management configuration"""
    max_contracts: int = Field(default=4, description="Maximum contracts per position")
    max_notional: Optional[float] = Field(default=None, description="Maximum notional exposure")
    max_loss_per_trade: Optional[float] = Field(default=None, description="Maximum loss per trade")
    max_loss_per_day: Optional[float] = Field(default=None, description="Maximum loss per day")
    margin_per_contract: Optional[float] = Field(default=None, description="Margin requirement per contract")


class ReportingConfig(BaseModel):
    """Reporting configuration"""
    run_dir_root: str = Field(default="runs", description="Root directory for run outputs")
    save_csv: bool = Field(default=True, description="Save CSV files")
    save_plot: bool = Field(default=True, description="Save matplotlib plots (if available)")
    save_log: bool = Field(default=True, description="Save run log")


class StrategyConfig(BaseModel):
    """Strategy configuration"""
    name: str = Field(description="Strategy name (must be registered)")
    params: Dict[str, Any] = Field(default_factory=dict, description="Strategy parameters")


class RunConfig(BaseModel):
    """Complete run configuration"""
    data: DataConfig
    engine: EngineConfig
    execution: ExecutionConfig = Field(default_factory=ExecutionConfig)
    selector: SelectorConfig = Field(default_factory=SelectorConfig)
    risk: RiskConfig = Field(default_factory=RiskConfig)
    reporting: ReportingConfig = Field(default_factory=ReportingConfig)
    strategy: StrategyConfig

