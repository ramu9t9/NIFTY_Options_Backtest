#!/usr/bin/env python3
"""
Broadcaster Data Writer for Replay Mode
Connects to Centralize Data Centre broadcaster and writes data to replay database.
"""

import os
import sys
import time
import json
import signal
import argparse
import logging
from datetime import datetime, timezone
from typing import Dict, Any, Optional

import websocket
from sqlalchemy import create_engine, text
from sqlalchemy.pool import NullPool

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Database schema (same as live mode)
DDL_LTP_SQLITE = """
CREATE TABLE IF NOT EXISTS ltp_ticks (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  symbol TEXT NOT NULL,
  token  TEXT NOT NULL,
  ts TEXT NOT NULL,
  ltp REAL,
  bid REAL,
  ask REAL,
  volume INTEGER,
  oi INTEGER,
  delta REAL,
  gamma REAL,
  theta REAL,
  vega REAL,
  iv REAL,
  source TEXT DEFAULT 'broadcaster'
);
"""

DDL_UNIQUE_INDEX = """
CREATE UNIQUE INDEX IF NOT EXISTS ux_ltp_symbol_ts ON ltp_ticks(symbol, ts);
"""


class BroadcasterDataWriter:
    """Receives data from broadcaster and writes to replay database."""
    
    def __init__(self, broadcaster_url: str, db_path: str):
        self.broadcaster_url = broadcaster_url
        self.db_path = db_path
        self.engine = None
        self.ws = None
        self.running = True
        self.records_written = 0
        self.last_write_time = None
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully."""
        logger.info(f"\n🛑 Received signal {signum} - initiating graceful shutdown...")
        self.running = False
        if self.ws:
            self.ws.close()
        logger.info("✅ Signal handler completed")
        
    def init_db(self):
        """Initialize database with schema."""
        try:
            logger.info(f"🔄 Initializing database: {self.db_path}")
            self.engine = create_engine(f"sqlite:///{self.db_path}", poolclass=NullPool, future=True)
            
            with self.engine.begin() as conn:
                # Create table
                conn.exec_driver_sql(DDL_LTP_SQLITE)
                
                # Apply SQLite optimizations
                conn.exec_driver_sql("PRAGMA journal_mode=WAL;")
                conn.exec_driver_sql("PRAGMA synchronous=NORMAL;")
                conn.exec_driver_sql("PRAGMA temp_store=MEMORY;")
                conn.exec_driver_sql("PRAGMA mmap_size=268435456;")  # 256MB
                conn.exec_driver_sql("PRAGMA cache_size=-200000;")   # ~200MB
                
                # Create unique index
                conn.exec_driver_sql(DDL_UNIQUE_INDEX)
            
            logger.info(f"✅ Database ready at {self.db_path}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Database initialization error: {e}")
            return False
    
    def _write_to_db(self, data: Dict[str, Any]):
        """Write broadcaster data to database."""
        try:
            if not data:
                return
                
            rows = []
            timestamp = data.get('timestamp', datetime.now(timezone.utc).isoformat())
            
            # Process market data
            market_data = data.get('data', {})
            
            for symbol, tick_data in market_data.items():
                row = {
                    'symbol': symbol,
                    'token': tick_data.get('token', ''),
                    'ts': timestamp,
                    'ltp': tick_data.get('ltp'),
                    'bid': tick_data.get('bid'),
                    'ask': tick_data.get('ask'),
                    'volume': tick_data.get('volume'),
                    'oi': tick_data.get('oi'),
                    'delta': tick_data.get('delta'),
                    'gamma': tick_data.get('gamma'),
                    'theta': tick_data.get('theta'),
                    'vega': tick_data.get('vega'),
                    'iv': tick_data.get('iv'),
                    'source': 'broadcaster'
                }
                rows.append(row)
            
            if rows:
                self._insert_rows(rows)
                self.records_written += len(rows)
                self.last_write_time = datetime.now()
                
                if self.records_written % 100 == 0:  # Log every 100 records
                    logger.info(f"📊 Written {self.records_written} records total")
                    
        except Exception as e:
            logger.error(f"❌ Error writing to database: {e}")
    
    def _insert_rows(self, rows):
        """Insert rows with upsert logic."""
        if not rows:
            return
            
        cols = list(rows[0].keys())
        placeholders = ", ".join([f":{c}" for c in cols])
        update_cols = [c for c in cols if c not in ("id", "symbol", "ts")]
        set_clause = ", ".join([f"{c}=excluded.{c}" for c in update_cols]) or "symbol=excluded.symbol"
        
        sql = (
            f"INSERT INTO ltp_ticks ({', '.join(cols)}) VALUES ({placeholders}) "
            f"ON CONFLICT(symbol, ts) DO UPDATE SET {set_clause}"
        )
        
        with self.engine.begin() as conn:
            conn.execute(text(sql), rows)
    
    def on_message(self, ws, message):
        """Handle incoming broadcaster messages."""
        try:
            data = json.loads(message)
            self._write_to_db(data)
        except json.JSONDecodeError as e:
            logger.error(f"❌ JSON decode error: {e}")
        except Exception as e:
            logger.error(f"❌ Error processing message: {e}")
    
    def on_error(self, ws, error):
        """Handle WebSocket errors."""
        logger.error(f"❌ WebSocket error: {error}")
    
    def on_close(self, ws, close_status_code, close_msg):
        """Handle WebSocket close."""
        logger.info(f"🔌 WebSocket closed: {close_status_code} - {close_msg}")
        logger.info(f"📊 Total records written: {self.records_written}")
    
    def on_open(self, ws):
        """Handle WebSocket open."""
        logger.info(f"✅ Connected to broadcaster: {self.broadcaster_url}")
        logger.info("📡 Receiving data...")
    
    def start(self):
        """Start receiving data from broadcaster."""
        if not self.init_db():
            logger.error("❌ Failed to initialize database")
            return
        
        logger.info("=" * 60)
        logger.info("🎯 BROADCASTER DATA WRITER - REPLAY MODE")
        logger.info("=" * 60)
        logger.info(f"📡 Broadcaster: {self.broadcaster_url}")
        logger.info(f"💾 Database: {self.db_path}")
        logger.info("=" * 60)
        
        # Create WebSocket connection
        self.ws = websocket.WebSocketApp(
            self.broadcaster_url,
            on_message=self.on_message,
            on_error=self.on_error,
            on_close=self.on_close,
            on_open=self.on_open
        )
        
        # Run forever with auto-reconnect
        while self.running:
            try:
                self.ws.run_forever()
                if self.running:
                    logger.info("🔄 Reconnecting in 5 seconds...")
                    time.sleep(5)
            except KeyboardInterrupt:
                logger.info("\n🛑 Keyboard interrupt received")
                break
            except Exception as e:
                logger.error(f"❌ Error: {e}")
                if self.running:
                    logger.info("🔄 Reconnecting in 5 seconds...")
                    time.sleep(5)
        
        logger.info("✅ Broadcaster data writer stopped")
        logger.info(f"📊 Final count: {self.records_written} records written")


def main():
    parser = argparse.ArgumentParser(description='Broadcaster Data Writer for Replay Mode')
    parser.add_argument('--broadcaster-url', type=str, default='ws://localhost:8765',
                        help='Broadcaster WebSocket URL (default: ws://localhost:8765)')
    parser.add_argument('--db-path', type=str, 
                        default='g:/Projects/NIFTY_Options_Backtest/data/nifty_replay.db',
                        help='Path to replay database')
    
    args = parser.parse_args()
    
    writer = BroadcasterDataWriter(args.broadcaster_url, args.db_path)
    writer.start()


if __name__ == "__main__":
    main()
