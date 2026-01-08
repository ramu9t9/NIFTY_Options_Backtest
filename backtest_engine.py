"""
Backtest Engine for NIFTY Options Paper Trading Strategy

Replicates the proven backtest results:
- 114 trades
- 57.89% win rate
- Profit Factor: 2.24

Strategy:
1. Build 30-second candles from tick data
2. Detect trend signals (0.11% cumulative move)
3. Analyze patterns using Greeks (60-second window)
4. Simulate trades with target/stop-loss/time exits
"""

import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Dict, List, Optional, Tuple
import pandas as pd
import logging

# Reuse from live_trading_engine.py
from live_trading_engine import (
    THRESHOLDS_DEFAULT,
    calculate_fast_indicators,
    detect_fast_patterns,
    determine_direction_from_patterns,
    calculate_transaction_cost,
    get_atm_strike,
    Candle,
)

logger = logging.getLogger(__name__)
IST = timezone(timedelta(hours=5, minutes=30))


@dataclass
class TrendSignal:
    """Represents a detected trend signal"""
    signal_time: datetime
    spot_price: float
    trend_direction: str  # "UP" or "DOWN"
    cumulative_move_pct: float
    trend_number: int


@dataclass
class TradeSignal:
    """Represents a trade entry signal with patterns"""
    entry_time: datetime
    spot_price: float
    predicted_direction: str  # "CE" or "PE"
    option_symbol: str
    strike: int
    expiry: str
    patterns_detected: List[Dict[str, Any]]
    patterns_count: int
    trend_number: int


@dataclass
class TradeResult:
    """Represents a completed trade with P&L"""
    order_id: int
    trend_number: int
    entry_time: datetime
    exit_time: datetime
    option_symbol: str
    strike: int
    option_type: str  # "CE" or "PE"
    predicted_direction: str
    entry_price: float
    exit_price: float
    exit_reason: str  # "TARGET", "STOP_LOSS", "TIME"
    gross_pnl: float
    transaction_cost: float
    net_pnl: float
    pnl_points: float
    hold_time_minutes: float
    target_hit: bool = False
    stop_hit: bool = False
    patterns_count: int = 0
    patterns_str: str = ""


@dataclass
class BacktestResults:
    """Aggregated backtest results"""
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    total_gross_pnl: float
    total_transaction_cost: float
    total_net_pnl: float
    profit_factor: float
    avg_win: float
    avg_loss: float
    max_win: float
    max_loss: float
    avg_hold_minutes: float
    trades: List[TradeResult] = field(default_factory=list)
    
    # Additional metrics
    trends_detected: int = 0
    patterns_analyzed: int = 0
    signals_generated: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for display"""
        return {
            'Total Trades': self.total_trades,
            'Winning Trades': self.winning_trades,
            'Losing Trades': self.losing_trades,
            'Win Rate': f"{self.win_rate:.2f}%",
            'Total Net P&L': f"₹{self.total_net_pnl:,.2f}",
            'Profit Factor': f"{self.profit_factor:.2f}",
            'Avg Win': f"₹{self.avg_win:,.2f}",
            'Avg Loss': f"₹{self.avg_loss:,.2f}",
            'Max Win': f"₹{self.max_win:,.2f}",
            'Max Loss': f"₹{self.max_loss:,.2f}",
            'Avg Hold Time': f"{self.avg_hold_minutes:.1f} min",
            'Trends Detected': self.trends_detected,
            'Patterns Analyzed': self.patterns_analyzed,
            'Signals Generated': self.signals_generated,
        }


class BacktestEngine:
    """
    Backtest engine that replicates live trading strategy on historical data.
    
    Database: nifty_local.db (tick data at 5-second intervals)
    Strategy: 3-stage approach (trend → patterns → trade)
    """
    
    def __init__(self, db_path: str, config: Optional[Dict[str, Any]] = None):
        """
        Initialize backtest engine.
        
        Args:
            db_path: Path to SQLite database (nifty_local.db)
            config: Strategy configuration parameters
        """
        self.db_path = db_path
        self.config = config or {}
        
        # Strategy parameters
        self.candle_interval_seconds = self.config.get('candle_interval_seconds', 30)
        self.movement_threshold = self.config.get('movement_threshold', 0.11)
        self.pattern_window_seconds = self.config.get('pattern_window_seconds', 60)
        
        # Trade parameters
        self.lot_size = self.config.get('lot_size', 3750)  # 50 lots × 75
        self.target_pct = self.config.get('target_pct', 10.0)
        self.stop_pct = self.config.get('stop_pct', 5.0)
        self.max_hold_minutes = self.config.get('max_hold_minutes', 3.0)
        
        # Pattern thresholds
        self.thresholds = self.config.get('thresholds', dict(THRESHOLDS_DEFAULT))
        
        # Progress callback
        self.progress_callback: Optional[Callable[[str, float], None]] = None
        
        # State
        self.trades: List[TradeResult] = []
        self.order_counter = 0
        
    def run(self, start_date: str, end_date: str) -> BacktestResults:
        """
        Run backtest for given date range.
        
        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            
        Returns:
            BacktestResults with all trades and metrics
        """
        logger.info(f"Starting backtest from {start_date} to {end_date}")
        self._update_progress("Loading NIFTY 50 data...", 0.0)
        
        # Connect to database
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        
        try:
            # Step 1: Load NIFTY 50 tick data
            nifty_df = self._load_nifty_data(conn, start_date, end_date)
            if nifty_df.empty:
                logger.warning("No NIFTY 50 data found for date range")
                return self._create_empty_results()
            
            logger.info(f"Loaded {len(nifty_df)} NIFTY 50 ticks")
            self._update_progress(f"Building candles from {len(nifty_df)} ticks...", 0.1)
            
            # Step 2: Build 30-second candles
            candles_df = self._build_candles(nifty_df)
            logger.info(f"Built {len(candles_df)} candles")
            self._update_progress(f"Detecting trends from {len(candles_df)} candles...", 0.2)
            
            # Step 3: Detect trend signals (with data availability pre-check)
            trend_signals = self._detect_trend_signals(candles_df, conn)
            logger.info(f"Detected {len(trend_signals)} trend signals")
            self._update_progress(f"Analyzing {len(trend_signals)} trend signals...", 0.3)
            
            # Step 4: Analyze patterns and simulate trades
            for i, signal in enumerate(trend_signals):
                progress = 0.3 + (0.6 * (i + 1) / len(trend_signals))
                self._update_progress(
                    f"Processing signal {i+1}/{len(trend_signals)} at {signal.signal_time.strftime('%Y-%m-%d %H:%M:%S')}",
                    progress
                )
                
                # Analyze patterns
                trade_signal = self._analyze_patterns(conn, signal)
                if trade_signal:
                    # Simulate trade
                    trade_result = self._simulate_trade(conn, trade_signal)
                    if trade_result:
                        self.trades.append(trade_result)
                        logger.info(
                            f"Trade #{trade_result.order_id}: {trade_result.option_type} "
                            f"Entry={trade_result.entry_price:.2f} Exit={trade_result.exit_price:.2f} "
                            f"P&L=₹{trade_result.net_pnl:,.2f} ({trade_result.exit_reason})"
                        )
            
            # Step 5: Calculate results
            self._update_progress("Calculating results...", 0.95)
            results = self._calculate_results(len(trend_signals))
            
            self._update_progress("Backtest complete!", 1.0)
            logger.info(f"Backtest complete: {results.total_trades} trades, {results.win_rate:.2f}% win rate")
            
            return results
            
        finally:
            conn.close()
    
    def _load_nifty_data(self, conn: sqlite3.Connection, start_date: str, end_date: str) -> pd.DataFrame:
        """Load NIFTY 50 tick data for date range"""
        query = """
        SELECT ts, ltp
        FROM ltp_ticks
        WHERE symbol = 'NIFTY 50'
          AND date(ts) >= date(?)
          AND date(ts) <= date(?)
        ORDER BY ts
        """
        
        df = pd.read_sql_query(query, conn, params=(start_date, end_date))
        if not df.empty:
            df['ts'] = pd.to_datetime(df['ts'], format='mixed', errors='coerce', utc=True)
            df = df.dropna(subset=['ts'])
        
        return df
    
    def _build_candles(self, ticks_df: pd.DataFrame) -> pd.DataFrame:
        """
        Build 30-second OHLC candles from tick data.
        
        Args:
            ticks_df: DataFrame with 'ts' and 'ltp' columns
            
        Returns:
            DataFrame with candle data (filtered for complete candles)
        """
        # Group by 30-second intervals using standard clock boundaries
        ticks_df['bucket'] = ticks_df['ts'].dt.floor(f'{self.candle_interval_seconds}s')
        
        candles = ticks_df.groupby('bucket').agg({
            'ltp': ['first', 'max', 'min', 'last', 'count']
        }).reset_index()
        
        candles.columns = ['ts', 'open', 'high', 'low', 'close', 'tick_count']
        
        # IMPROVEMENT #1: Filter out incomplete candles (< 3 ticks)
        # This prevents false signals from partial data at start/end of day
        min_ticks = 3
        initial_count = len(candles)
        candles = candles[candles['tick_count'] >= min_ticks].copy()
        filtered_count = initial_count - len(candles)
        if filtered_count > 0:
            logger.debug(f"Filtered {filtered_count} incomplete candles (< {min_ticks} ticks)")
        
        # Calculate percentage change and direction
        candles['pct_change'] = ((candles['close'] - candles['open']) / candles['open']) * 100.0
        candles['direction'] = candles['pct_change'].apply(
            lambda x: 1 if x > 0.01 else (-1 if x < -0.01 else 0)
        )
        
        return candles
    
    def _detect_trend_signals(self, candles_df: pd.DataFrame, conn: sqlite3.Connection) -> List[TrendSignal]:
        """
        Detect trend signals using cumulative move tracking.
        
        IMPROVEMENT #3: Added data availability pre-check to eliminate invalid signals.
        
        Implements same logic as TrendSignalDetector in live_trading_engine.py
        """
        signals = []
        
        # State tracking
        current_direction: Optional[int] = None
        direction_start_price: Optional[float] = None
        trend_counter = 0
        already_signaled = False
        skipped_no_data = 0
        
        for idx, row in candles_df.iterrows():
            direction = row['direction']
            close_price = row['close']
            open_price = row['open']
            ts = row['ts']
            
            # Check for direction change or neutral
            if direction != current_direction or direction == 0:
                # Reset tracking when direction changes or goes neutral
                # NOTE: This is CORRECT - we only reset on direction CHANGE,
                # not on every candle. Continuous trends maintain same baseline.
                current_direction = direction
                direction_start_price = open_price
                already_signaled = False
                continue
            
            # Skip if already signaled in this trend
            if already_signaled or direction_start_price is None:
                continue
            
            # Calculate cumulative move
            cumulative_pct = ((close_price - direction_start_price) / direction_start_price) * 100.0
            
            # Check threshold
            if abs(cumulative_pct) >= self.movement_threshold:
                # IMPROVEMENT #3: Pre-check options data availability
                # This eliminates signals we can't trade (23 in original run)
                window_start = ts - timedelta(seconds=self.pattern_window_seconds)
                options_df = self._load_options_window(conn, window_start, ts, close_price)
                
                if not options_df.empty:
                    # Valid signal with data available
                    trend_counter += 1
                    already_signaled = True
                    
                    # Create signal
                    trend_direction = "UP" if direction == 1 else "DOWN"
                    signal = TrendSignal(
                        signal_time=ts,
                        spot_price=close_price,
                        trend_direction=trend_direction,
                        cumulative_move_pct=cumulative_pct,
                        trend_number=trend_counter
                    )
                    signals.append(signal)
                else:
                    # Skip this trend - no options data available
                    skipped_no_data += 1
                    already_signaled = True  # Don't retry this trend
                    logger.debug(f"Skipping trend at {ts} - no options data available")
        
        if skipped_no_data > 0:
            logger.info(f"Skipped {skipped_no_data} trends due to missing options data")
        
        return signals
    
    def _analyze_patterns(self, conn: sqlite3.Connection, signal: TrendSignal) -> Optional[TradeSignal]:
        """
        Analyze patterns for a trend signal.
        
        IMPROVEMENT #2: Pattern window is now INDEPENDENT of signal candle.
        Window ends BEFORE the signal candle to avoid circular logic.
        
        Args:
            conn: Database connection
            signal: Trend signal to analyze
            
        Returns:
            TradeSignal if patterns found, else None
        """
        # IMPROVEMENT #2: Independent pattern window
        # Window ends BEFORE the signal candle (no overlap)
        # If signal is at 09:16:00, pattern window is 09:14:30-09:15:30
        window_end = signal.signal_time - timedelta(seconds=self.candle_interval_seconds)
        window_start = window_end - timedelta(seconds=self.pattern_window_seconds)
        
        options_df = self._load_options_window(conn, window_start, window_end, signal.spot_price)
        
        if options_df.empty:
            logger.warning(f"No options data for signal at {signal.signal_time}")
            return None
        
        # Calculate indicators
        metrics = calculate_fast_indicators(options_df)
        if not metrics:
            logger.warning(f"Failed to calculate indicators for signal at {signal.signal_time}")
            return None
        
        # Detect patterns
        patterns = detect_fast_patterns(metrics, self.thresholds)
        if not patterns:
            logger.debug(f"No patterns detected for signal at {signal.signal_time}")
            return None
        
        # Convert patterns list to dict format for direction determination 
        # detect_fast_patterns returns List[Dict] with 'pattern_type', 'value', 'direction'
        # determine_direction_from_patterns expects Dict[str, float]
        patterns_dict = {p['pattern_type']: p['value'] for p in patterns}
        
        # Determine trade direction using the dict format
        predicted_direction = determine_direction_from_patterns(patterns_dict)
        option_type = "CE" if predicted_direction == "BULLISH" else "PE"
        
        # Get ATM strike and option symbol
        atm_strike = get_atm_strike(signal.spot_price)
        expiry = self._get_nearest_expiry(conn, signal.signal_time)
        if not expiry:
            logger.warning(f"No expiry found for signal at {signal.signal_time}")
            return None
        
        option_symbol = f"NIFTY{expiry}{atm_strike}{option_type}"
        
        # Create trade signal
        trade_signal = TradeSignal(
            entry_time=signal.signal_time,
            spot_price=signal.spot_price,
            predicted_direction=option_type,
            option_symbol=option_symbol,
            strike=atm_strike,
            expiry=expiry,
            patterns_detected=patterns,
            patterns_count=len(patterns),
            trend_number=signal.trend_number
        )
        
        return trade_signal
    
    def _load_options_window(
        self, 
        conn: sqlite3.Connection, 
        start: datetime, 
        end: datetime,
        spot_price: float
    ) -> pd.DataFrame:
        """Load options data for 60-second window"""
        # Get ATM strike to filter relevant options
        atm_strike = get_atm_strike(spot_price)
        
        query = """
        SELECT ts, symbol, ltp, volume, oi, iv, delta, gamma, theta, vega
        FROM ltp_ticks
        WHERE symbol LIKE 'NIFTY%'
          AND symbol != 'NIFTY 50'
          AND ts >= ? AND ts < ?
        ORDER BY ts
        """
        
        df = pd.read_sql_query(query, conn, params=(start.isoformat(), end.isoformat()))
        
        if not df.empty:
            df['ts'] = pd.to_datetime(df['ts'], format='mixed', errors='coerce', utc=True)
            df = df.dropna(subset=['ts'])
        
        return df
    
    def _get_nearest_expiry(self, conn: sqlite3.Connection, signal_time: datetime) -> Optional[str]:
        """Get nearest weekly expiry from available option symbols"""
        query = """
        SELECT DISTINCT symbol
        FROM ltp_ticks
        WHERE symbol LIKE 'NIFTY%'
          AND symbol != 'NIFTY 50'
          AND date(ts) = date(?)
        LIMIT 100
        """
        
        cursor = conn.execute(query, (signal_time.isoformat(),))
        symbols = [row[0] for row in cursor.fetchall()]
        
        if not symbols:
            return None
        
        # Extract expiries from symbols (format: NIFTYDDMMMYY...)
        import re
        expiry_pattern = re.compile(r'NIFTY(\d{2}[A-Z]{3}\d{2})')
        
        expiries = set()
        for symbol in symbols:
            match = expiry_pattern.match(symbol)
            if match:
                expiries.add(match.group(1))
        
        if not expiries:
            return None
        
        # Return nearest expiry (smallest date)
        return sorted(expiries)[0]
    
    def _simulate_trade(self, conn: sqlite3.Connection, signal: TradeSignal) -> Optional[TradeResult]:
        """
        Simulate trade execution with exit logic.
        
        Args:
            conn: Database connection
            signal: Trade signal
            
        Returns:
            TradeResult if trade executed, else None
        """
        # Get entry price
        entry_price = self._get_entry_price(conn, signal.option_symbol, signal.entry_time)
        if entry_price is None:
            logger.warning(f"No entry price found for {signal.option_symbol} at {signal.entry_time}")
            return None
        
        # Calculate exit thresholds
        target_price = entry_price * (1 + self.target_pct / 100)
        stop_price = entry_price * (1 - self.stop_pct / 100)
        max_exit_time = signal.entry_time + timedelta(minutes=self.max_hold_minutes)
        
        # Monitor for exit
        exit_result = self._monitor_exit(
            conn, 
            signal.option_symbol, 
            signal.entry_time, 
            max_exit_time,
            target_price,
            stop_price
        )
        
        if not exit_result:
            logger.warning(f"No exit found for {signal.option_symbol}")
            return None
        
        exit_time, exit_price, exit_reason = exit_result
        
        # Calculate P&L
        gross_pnl = (exit_price - entry_price) * self.lot_size
        buy_value = entry_price * self.lot_size
        sell_value = exit_price * self.lot_size
        transaction_cost = calculate_transaction_cost(buy_value, sell_value)
        net_pnl = gross_pnl - transaction_cost
        pnl_points = exit_price - entry_price
        hold_minutes = (exit_time - signal.entry_time).total_seconds() / 60.0
        
        # Create trade result
        self.order_counter += 1
        
        patterns_str = ", ".join([f"{p['pattern_type']}:{p['value']:.2f}" for p in signal.patterns_detected])
        
        trade = TradeResult(
            order_id=self.order_counter,
            trend_number=signal.trend_number,
            entry_time=signal.entry_time,
            exit_time=exit_time,
            option_symbol=signal.option_symbol,
            strike=signal.strike,
            option_type=signal.predicted_direction,
            predicted_direction=signal.predicted_direction,
            entry_price=entry_price,
            exit_price=exit_price,
            exit_reason=exit_reason,
            gross_pnl=gross_pnl,
            transaction_cost=transaction_cost,
            net_pnl=net_pnl,
            pnl_points=pnl_points,
            hold_time_minutes=hold_minutes,
            target_hit=(exit_reason == "TARGET"),
            stop_hit=(exit_reason == "STOP_LOSS"),
            patterns_count=signal.patterns_count,
            patterns_str=patterns_str
        )
        
        return trade
    
    def _get_entry_price(self, conn: sqlite3.Connection, symbol: str, entry_time: datetime) -> Optional[float]:
        """Get entry price for option at signal time"""
        query = """
        SELECT ltp
        FROM ltp_ticks
        WHERE symbol = ?
          AND ts >= ?
        ORDER BY ts
        LIMIT 1
        """
        
        cursor = conn.execute(query, (symbol, entry_time.isoformat()))
        row = cursor.fetchone()
        
        return row[0] if row else None
    
    def _monitor_exit(
        self,
        conn: sqlite3.Connection,
        symbol: str,
        entry_time: datetime,
        max_exit_time: datetime,
        target_price: float,
        stop_price: float
    ) -> Optional[Tuple[datetime, float, str]]:
        """
        Monitor option price for exit conditions.
        
        Returns:
            (exit_time, exit_price, exit_reason) or None
        """
        query = """
        SELECT ts, ltp
        FROM ltp_ticks
        WHERE symbol = ?
          AND ts >= ? AND ts <= ?
        ORDER BY ts
        """
        
        cursor = conn.execute(query, (symbol, entry_time.isoformat(), max_exit_time.isoformat()))
        
        for row in cursor:
            ts_str, ltp = row
            ts = pd.to_datetime(ts_str, format='mixed', errors='coerce', utc=True)
            
            # Check target
            if ltp >= target_price:
                return (ts, ltp, "TARGET")
            
            # Check stop loss
            if ltp <= stop_price:
                return (ts, ltp, "STOP_LOSS")
        
        # Time exit - get last available price
        cursor.execute(
            "SELECT ts, ltp FROM ltp_ticks WHERE symbol = ? AND ts <= ? ORDER BY ts DESC LIMIT 1",
            (symbol, max_exit_time.isoformat())
        )
        row = cursor.fetchone()
        if row:
            ts_str, ltp = row
            ts = pd.to_datetime(ts_str, format='mixed', errors='coerce', utc=True)
            return (ts, ltp, "TIME")
        
        return None
    
    def _calculate_results(self, trends_detected: int) -> BacktestResults:
        """Calculate aggregated backtest results"""
        if not self.trades:
            return self._create_empty_results()
        
        # Calculate metrics
        total_trades = len(self.trades)
        winning_trades = sum(1 for t in self.trades if t.net_pnl > 0)
        losing_trades = sum(1 for t in self.trades if t.net_pnl < 0)
        win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0.0
        
        total_gross_pnl = sum(t.gross_pnl for t in self.trades)
        total_transaction_cost = sum(t.transaction_cost for t in self.trades)
        total_net_pnl = sum(t.net_pnl for t in self.trades)
        
        wins = [t.net_pnl for t in self.trades if t.net_pnl > 0]
        losses = [abs(t.net_pnl) for t in self.trades if t.net_pnl < 0]
        
        avg_win = sum(wins) / len(wins) if wins else 0.0
        avg_loss = sum(losses) / len(losses) if losses else 0.0
        max_win = max(wins) if wins else 0.0
        max_loss = max(losses) if losses else 0.0
        
        profit_factor = sum(wins) / sum(losses) if losses else float('inf')
        avg_hold_minutes = sum(t.hold_time_minutes for t in self.trades) / total_trades if total_trades > 0 else 0.0
        
        return BacktestResults(
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            win_rate=win_rate,
            total_gross_pnl=total_gross_pnl,
            total_transaction_cost=total_transaction_cost,
            total_net_pnl=total_net_pnl,
            profit_factor=profit_factor,
            avg_win=avg_win,
            avg_loss=avg_loss,
            max_win=max_win,
            max_loss=max_loss,
            avg_hold_minutes=avg_hold_minutes,
            trades=self.trades.copy(),
            trends_detected=trends_detected,
            patterns_analyzed=len([t for t in self.trades if t.patterns_count > 0]),
            signals_generated=total_trades
        )
    
    def _create_empty_results(self) -> BacktestResults:
        """Create empty results object"""
        return BacktestResults(
            total_trades=0,
            winning_trades=0,
            losing_trades=0,
            win_rate=0.0,
            total_gross_pnl=0.0,
            total_transaction_cost=0.0,
            total_net_pnl=0.0,
            profit_factor=0.0,
            avg_win=0.0,
            avg_loss=0.0,
            max_win=0.0,
            max_loss=0.0,
            avg_hold_minutes=0.0,
            trades=[]
        )
    
    def _update_progress(self, message: str, progress: float) -> None:
        """Update progress callback"""
        if self.progress_callback:
            self.progress_callback(message, progress)
        logger.info(f"Progress {progress*100:.0f}%: {message}")
