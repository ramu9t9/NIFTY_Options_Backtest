#!/usr/bin/env python3
"""
Local Test Variant — SQLite
NIFTY Second-by-Second Stream (LTP + OI) → SQLite (nifty_local.db)

- Same behavior as the cloud version, but defaults to SQLite so you can test locally.
- DB_URL can be omitted; defaults to sqlite:///./nifty_local.db
"""

import os
import time
import threading
import json
import signal
import sys
from datetime import datetime, timezone, timedelta
import logging

import pandas as pd
import requests
from sqlalchemy import create_engine, text
from sqlalchemy.pool import NullPool

from SmartApi import SmartConnect
import pyotp

# -------- Config --------
ANGEL_API_KEY     = "cuDLz9MO"
ANGEL_USER_ID     = "m812631"
ANGEL_PIN         = 9029
ANGEL_TOTP_SECRET = "2KMWT6YBLNGNNPZOMSBQXFOPVE"

DB_URL        = os.getenv("DB_URL", "sqlite:///g:/Projects/NIFTY_Options_Backtest/data/nifty_live.db").strip()
ATM_WINDOW    = int(os.getenv("ATM_WINDOW", "5"))
OI_POLL_SECS  = int(os.getenv("OI_POLL_SECS", "30"))
DATA_STORAGE_INTERVAL_SECS = int(os.getenv("DATA_STORAGE_INTERVAL_SECS", "5"))  # Data storage interval in seconds
SYMBOL_BASE   = os.getenv("SYMBOL_BASE", "NIFTY").upper()

# Configuration Guide:
# ===================
# To change settings, set environment variables or modify the defaults above:
#
# DATA_STORAGE_INTERVAL_SECS: How often to write data to database (default: 5 seconds)
#   - Lower values (1-3s): More frequent updates, higher database load
#   - Higher values (5-10s): Less frequent updates, lower database load
#   - Recommended: 5 seconds for most use cases
#
# OI_POLL_SECS: How often to poll for OI data (default: 30 seconds)
#   - OI data doesn't change as frequently as LTP
#   - Lower values increase API calls to Angel One
#   - Recommended: 30 seconds
#
# ATM_WINDOW: Number of strikes around ATM to monitor (default: 5)
#   - ±5 strikes = 11 total strikes (5 below + ATM + 5 above)
#   - More strikes = more data but higher resource usage
#   - Recommended: 5 for most analysis needs
#
# Example environment variables:
#   export DATA_STORAGE_INTERVAL_SECS=3
#   export OI_POLL_SECS=15
#   export ATM_WINDOW=3

# Credentials are hardcoded for this deployment

engine = create_engine(DB_URL, poolclass=NullPool, future=True)

# Setup logging with timestamps
def setup_logging():
    """Setup logging with timestamps"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    return logging.getLogger(__name__)

logger = setup_logging()

def _is_mysql():
    return DB_URL.lower().startswith("mysql")

def _is_sqlite():
    return DB_URL.lower().startswith("sqlite")

DDL_LTP_PG = """
CREATE TABLE IF NOT EXISTS ltp_ticks (
  id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
  symbol VARCHAR(64) NOT NULL,
  token  VARCHAR(32) NOT NULL,
  ts TIMESTAMP WITH TIME ZONE NOT NULL,
  ltp DOUBLE PRECISION,
  bid DOUBLE PRECISION,
  ask DOUBLE PRECISION,
  volume BIGINT,
  oi BIGINT,
  delta DOUBLE PRECISION,
  gamma DOUBLE PRECISION,
  theta DOUBLE PRECISION,
  vega DOUBLE PRECISION,
  iv DOUBLE PRECISION,
  source VARCHAR(16) DEFAULT 'ws'
);
"""

DDL_OI_PG = """
CREATE TABLE IF NOT EXISTS oi_snapshots (
  id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
  symbol VARCHAR(64) NOT NULL,
  token  VARCHAR(32) NOT NULL,
  ts TIMESTAMP WITH TIME ZONE NOT NULL,
  oi BIGINT,
  volume BIGINT,
  delta DOUBLE PRECISION,
  gamma DOUBLE PRECISION,
  theta DOUBLE PRECISION,
  vega DOUBLE PRECISION,
  iv DOUBLE PRECISION
);
"""

DDL_LTP_MYSQL = """
CREATE TABLE IF NOT EXISTS ltp_ticks (
  id BIGINT AUTO_INCREMENT PRIMARY KEY,
  symbol VARCHAR(64) NOT NULL,
  token  VARCHAR(32) NOT NULL,
  ts TIMESTAMP(6) NOT NULL,
  ltp DOUBLE,
  bid DOUBLE,
  ask DOUBLE,
  volume BIGINT,
  oi BIGINT,
  delta DOUBLE,
  gamma DOUBLE,
  theta DOUBLE,
  vega DOUBLE,
  iv DOUBLE,
  source VARCHAR(16) DEFAULT 'ws'
) ENGINE=InnoDB;
"""

DDL_OI_MYSQL = """
CREATE TABLE IF NOT EXISTS oi_snapshots (
  id BIGINT AUTO_INCREMENT PRIMARY KEY,
  symbol VARCHAR(64) NOT NULL,
  token  VARCHAR(32) NOT NULL,
  ts TIMESTAMP(6) NOT NULL,
  oi BIGINT,
  volume BIGINT,
  delta DOUBLE,
  gamma DOUBLE,
  theta DOUBLE,
  vega DOUBLE,
  iv DOUBLE
) ENGINE=InnoDB;
"""

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
  source TEXT DEFAULT 'ws'
);
"""

DDL_OI_SQLITE = """
CREATE TABLE IF NOT EXISTS oi_snapshots (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  symbol TEXT NOT NULL,
  token  TEXT NOT NULL,
  ts TEXT NOT NULL,
  oi INTEGER,
  volume INTEGER,
  delta REAL,
  gamma REAL,
  theta REAL,
  vega REAL,
  iv REAL
);
"""

def init_db():
    max_retries = 3
    for attempt in range(max_retries):
        try:
            logger.info(f"🔄 Initializing database (attempt {attempt + 1}/{max_retries})...")
            
            # Create tables
            with engine.begin() as conn:
                if _is_mysql():
                    conn.exec_driver_sql(DDL_LTP_MYSQL)
                    conn.exec_driver_sql(DDL_OI_MYSQL)
                elif _is_sqlite():
                    conn.exec_driver_sql(DDL_LTP_SQLITE)
                    conn.exec_driver_sql(DDL_OI_SQLITE)
                else:
                    conn.exec_driver_sql(DDL_LTP_PG)
                    conn.exec_driver_sql(DDL_OI_PG)
            
            # Apply SQLite PRAGMAs + create unique indexes
            if _is_sqlite():
                with engine.begin() as conn:
                    # Durability & performance for frequent writes
                    conn.exec_driver_sql("PRAGMA journal_mode=WAL;")
                    conn.exec_driver_sql("PRAGMA synchronous=NORMAL;")
                    conn.exec_driver_sql("PRAGMA temp_store=MEMORY;")
                    conn.exec_driver_sql("PRAGMA mmap_size=268435456;")   # 256MB
                    conn.exec_driver_sql("PRAGMA cache_size=-200000;")    # ~200MB

                    # Duplicate guard (upsert target)
                    conn.exec_driver_sql(
                        "CREATE UNIQUE INDEX IF NOT EXISTS ux_ltp_symbol_ts ON ltp_ticks(symbol, ts);"
                    )
                    conn.exec_driver_sql(
                        "CREATE UNIQUE INDEX IF NOT EXISTS ux_oi_symbol_ts  ON oi_snapshots(symbol, ts);"
                    )

            logger.info(f"✅ DB ready at {DB_URL}")
            return True
            
        except Exception as e:
            print(f"❌ Database initialization error (attempt {attempt + 1}): {e}")
            if attempt < max_retries - 1:
                logger.info("🔄 Retrying database initialization...")
                time.sleep(2)
                continue
            else:
                print("❌ Failed to initialize database after all retries")
                raise RuntimeError(f"Database initialization failed: {e}")
    
    return False

# ------- Angel login & symbols -------
obj = SmartConnect(api_key=ANGEL_API_KEY)
auth_data = None
jwt_token = None
feed_token = None

def angel_login():
    global auth_data, jwt_token, feed_token
    totp = pyotp.TOTP(ANGEL_TOTP_SECRET).now()
    auth_data = obj.generateSession(ANGEL_USER_ID, ANGEL_PIN, totp)
    if not auth_data.get("status"):
        raise RuntimeError(f"Login failed: {auth_data}")
    jwt_token = auth_data.get("data", {}).get("jwtToken")
    feed_token = obj.getfeedToken()
    if not feed_token:
        raise RuntimeError("No feed token")
    logger.info(f"✅ Logged in as {ANGEL_USER_ID}")

SYMBOL_TO_TOKEN = {}
TOKEN_TO_SYMBOL = {}
NIFTY_INDEX_TOKEN = None
current_expiry_short = None

def fetch_instruments_and_map():
    global SYMBOL_TO_TOKEN, TOKEN_TO_SYMBOL, NIFTY_INDEX_TOKEN, current_expiry_short

    url = "https://margincalculator.angelbroking.com/OpenAPI_File/files/OpenAPIScripMaster.json"
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    df = pd.DataFrame(r.json())

    idx = df[(df['name'] == 'NIFTY 50') & (df['instrumenttype'] == 'INDEX') & (df['exch_seg'] == 'NSE')]
    if not idx.empty:
        NIFTY_INDEX_TOKEN = str(idx.iloc[0]['token'])
        SYMBOL_TO_TOKEN['NIFTY 50'] = NIFTY_INDEX_TOKEN
        TOKEN_TO_SYMBOL[NIFTY_INDEX_TOKEN] = 'NIFTY 50'
    else:
        NIFTY_INDEX_TOKEN = '99926000'
        SYMBOL_TO_TOKEN['NIFTY 50'] = NIFTY_INDEX_TOKEN
        TOKEN_TO_SYMBOL[NIFTY_INDEX_TOKEN] = 'NIFTY 50'

    # Get NIFTY Options
    opt = df[(df['name'] == SYMBOL_BASE) & (df['instrumenttype'] == 'OPTIDX')].copy()
    
    # Get NIFTY Futures
    fut = df[(df['name'] == SYMBOL_BASE) & (df['instrumenttype'] == 'FUTIDX')].copy()
    
    # Get current expiry using the same logic as Enhanced_OI_Monitor_CLEAN
    current_expiry = get_current_expiry(df)
    if current_expiry:
        current_expiry_short = current_expiry
        logger.info(f"📅 Using current expiry: {current_expiry_short}")
    else:
        # Fallback to next expiry if current not found
        exp_dates = []
        for s in sorted(opt['expiry'].unique()):
            try:
                exp_dates.append(datetime.strptime(s, "%d%b%Y"))
            except Exception:
                pass
        exp_dates = sorted([d for d in exp_dates if d > datetime.now()])
        chosen = exp_dates[0] if exp_dates else (datetime.now())
        current_expiry_short = chosen.strftime("%d%b%y").upper()
        logger.info(f"📅 Using fallback expiry: {current_expiry_short}")

    # Convert expiry format for matching
    # The API returns expiry in format like "21AUG2025" but symbols use "21AUG25"
    expiry_for_matching = current_expiry_short
    if len(current_expiry_short) == 7:  # Format: 21AUG25
        # Dynamically determine year based on expiry date
        try:
            current_year = datetime.now().year
            current_date = datetime.now()
            expiry_yy = current_expiry_short[-2:]  # Last 2 digits (e.g., "25")
            
            # Try current year first
            exp_date_current = datetime.strptime(current_expiry_short[:-2] + str(current_year)[-2:], '%d%b%y')
            
            # Try next year
            exp_date_next = datetime.strptime(current_expiry_short[:-2] + str(current_year + 1)[-2:], '%d%b%y')
            
            # Choose the expiry that is in the future (or today)
            if exp_date_current >= current_date:
                year = str(current_year)
            elif exp_date_next >= current_date:
                year = str(current_year + 1)
            else:
                # Both are in past, use next year (shouldn't happen for valid expiries)
                year = str(current_year + 1)
            
            expiry_for_matching = current_expiry_short[:-2] + year  # Convert to 21AUG2025 or 21AUG2026
        except ValueError:
            # Fallback to current year if parsing fails
            expiry_for_matching = current_expiry_short[:-2] + str(datetime.now().year)
    
    # Filter for current expiry - Options
    current_options = opt[opt['expiry'] == expiry_for_matching]
    
    # Filter for current expiry - Futures (use same expiry as options to ensure consistency)
    # This ensures on expiry day, both options and futures use the same expiry date
    current_futures = fut[fut['expiry'] == expiry_for_matching]
    
    # If no futures found with the same expiry as options, fall back to month-based logic
    if current_futures.empty and not fut.empty:
        print(f"⚠️ No futures found with expiry {expiry_for_matching}, falling back to month-based selection")
        # Get available futures expiries
        futures_expiries = sorted(fut['expiry'].unique())
        print(f"📅 Available futures expiries: {futures_expiries}")
        
        # Find current month's futures expiry first
        current_time = datetime.now()
        current_month = current_time.month
        current_year = current_time.year
        
        # Try to find current month's futures first (including today if it's expiry day)
        current_month_futures_expiry = None
        next_month_futures_expiry = None
        
        for exp in futures_expiries:
            try:
                exp_date = datetime.strptime(exp, "%d%b%Y")
                # Include today's expiry if it matches (for expiry day scenarios)
                if exp_date.date() >= current_time.date():
                    # Check current month
                    if exp_date.month == current_month and exp_date.year == current_year:
                        current_month_futures_expiry = exp
                        break
                    # Check next month (handle year transition)
                    elif not next_month_futures_expiry:
                        next_month_num = (current_month % 12) + 1
                        next_month_year = current_year if next_month_num > current_month else current_year + 1
                        if exp_date.month == next_month_num and exp_date.year == next_month_year:
                            next_month_futures_expiry = exp
            except:
                continue
        
        # Use current month's futures if available, otherwise next month's, otherwise nearest
        if current_month_futures_expiry:
            current_futures = fut[fut['expiry'] == current_month_futures_expiry]
            print(f"📈 Using current month futures expiry: {current_month_futures_expiry}")
        elif next_month_futures_expiry:
            current_futures = fut[fut['expiry'] == next_month_futures_expiry]
            print(f"📈 Using next month futures expiry: {next_month_futures_expiry}")
        else:
            # Fallback to nearest available
            nearest_futures_expiry = None
            for exp in futures_expiries:
                try:
                    exp_date = datetime.strptime(exp, "%d%b%Y")
                    if exp_date.date() >= current_time.date():
                        nearest_futures_expiry = exp
                        break
                except:
                    continue
            
            if nearest_futures_expiry:
                current_futures = fut[fut['expiry'] == nearest_futures_expiry]
                print(f"📈 Using nearest futures expiry: {nearest_futures_expiry}")
            else:
                # If no future expiry found, use the latest available
                current_futures = fut[fut['expiry'] == futures_expiries[-1]]
                print(f"📈 Using latest futures expiry: {futures_expiries[-1]}")
    elif not current_futures.empty:
        print(f"✅ Using futures with same expiry as options: {expiry_for_matching}")
    
    # If no exact match, try to find the closest expiry
    if current_options.empty:
        available_expiries = sorted(opt['expiry'].unique())
        print(f"Available expiries: {available_expiries[:5]}...")  # Show first 5
        for exp in available_expiries:
            if exp > datetime.now().strftime("%d%b%Y"):
                expiry_for_matching = exp
                current_options = opt[opt['expiry'] == expiry_for_matching]
                current_futures = fut[fut['expiry'] == expiry_for_matching]
                print(f"📅 Using closest available expiry: {expiry_for_matching}")
                break
    
    # Map Options symbols to tokens
    for _, row in current_options.iterrows():
        sym, tok = str(row['symbol']), str(row['token'])
        SYMBOL_TO_TOKEN[sym] = tok
        TOKEN_TO_SYMBOL[tok] = sym

    # Map Futures symbols to tokens
    for _, row in current_futures.iterrows():
        sym, tok = str(row['symbol']), str(row['token'])
        SYMBOL_TO_TOKEN[sym] = tok
        TOKEN_TO_SYMBOL[tok] = sym

    logger.info(f"✅ Symbols mapped for expiry {expiry_for_matching} (total {len(SYMBOL_TO_TOKEN)})")
    print(f"📊 Options count: {len(current_options)}")
    print(f"📈 Futures count: {len(current_futures)}")
    
    # Show some sample symbols
    sample_symbols = list(SYMBOL_TO_TOKEN.keys())[:5]
    print(f"📋 Sample symbols: {sample_symbols}")
    
    # Show futures symbols if available
    future_symbols = [s for s in SYMBOL_TO_TOKEN.keys() if 'FUT' in s]
    if future_symbols:
        print(f"🚀 Available Futures: {future_symbols}")

def get_current_expiry(instrument_df=None, index_name='NIFTY'):
    """Get the current month expiry date - same logic as Enhanced_OI_Monitor_CLEAN."""
    try:
        if instrument_df is None:
            print("⚠️ Warning: get_current_expiry called without instrument_df")
            return None
            
        current_time = datetime.now()
        options_df = instrument_df[(instrument_df['name'] == index_name) & (instrument_df['instrumenttype'] == 'OPTIDX')]
        
        if options_df.empty:
            return None
        
        # Get unique expiries and sort
        unique_expiries = sorted(options_df['expiry'].unique())
        expiry_dates = []
        
        for exp in unique_expiries:
            try:
                date = datetime.strptime(exp, '%d%b%Y')
                expiry_dates.append((exp, date))
            except ValueError:
                continue
        
        expiry_dates.sort(key=lambda x: x[1])
        current_expiry = None
        
        # First, try to find today's expiry (if it's a Thursday)
        today = current_time.date()
        for exp_str, exp_date in expiry_dates:
            if exp_date.date() == today:
                current_expiry = exp_str
                print(f"📅 Found today's expiry: {current_expiry}")
                break
        
        # If no today's expiry, find the next Thursday expiry
        if not current_expiry:
            for exp_str, exp_date in expiry_dates:
                if exp_date > current_time and exp_date.weekday() == 3:  # Thursday
                    current_expiry = exp_str
                    print(f"📅 Found next Thursday expiry: {current_expiry}")
                    break
        
        # If still no Thursday expiry found, take the next available
        if not current_expiry:
            for exp_str, exp_date in expiry_dates:
                if exp_date > current_time:
                    current_expiry = exp_str
                    print(f"📅 Found next available expiry: {current_expiry}")
                    break
        
        # If still no expiry, take the latest available
        if not current_expiry and expiry_dates:
            current_expiry = expiry_dates[-1][0]
            print(f"📅 Using latest available expiry: {current_expiry}")
        
        if not current_expiry:
            current_expiry = current_time.strftime('%d%b%Y').upper()
            print(f"📅 Using current date as expiry: {current_expiry}")
        
        # Convert to short format for symbol matching
        try:
            exp_date = datetime.strptime(current_expiry, '%d%b%Y')
            current_expiry_str = exp_date.strftime('%d%b%y').upper()  # Format: 14AUG25
            print(f"📅 Detected current expiry: {current_expiry_str}")
            return current_expiry_str
        except ValueError:
            print(f"📅 Using expiry as is: {current_expiry}")
            return current_expiry
        
    except Exception as e:
        print(f"❌ Error getting expiry: {e}")
        return None

def is_market_open():
    """
    Check if Indian market is currently open.
    
    Market Hours: Monday to Friday, 9:15 AM to 3:30 PM IST
    """
    # Get current UTC time and convert to IST
    utc_now = datetime.now(timezone.utc)
    ist_now = utc_now + timedelta(hours=5, minutes=30)
    
    # Check if it's a weekday (Monday = 0, Sunday = 6)
    if ist_now.weekday() >= 5:  # Saturday or Sunday
        print(f"📅 Weekend detected: {ist_now.strftime('%A')}")
        return False
    
    # Check if current time is between 9:15 and 15:30 IST
    current_time = ist_now.time()
    market_start = datetime.strptime("09:15:00", "%H:%M:%S").time()
    market_end = datetime.strptime("15:30:00", "%H:%M:%S").time()
    
    is_open = market_start <= current_time <= market_end
    logger.info(f"🕐 Market check: {ist_now.strftime('%H:%M:%S')} IST - {'OPEN' if is_open else 'CLOSED'}")
    
    return is_open

def is_trading_day_from_csv(csv_path=None):
    """
    Check if today is a trading day using the appropriate calendar file.
    Tries NSE_Market_Calendar_2025.csv first (for 2025 dates), then falls back to 2024_25.
    """
    try:
        import csv
        from datetime import date
        today = (datetime.now(timezone.utc) + timedelta(hours=5, minutes=30)).date()
        
        # Determine which calendar file to use based on current year
        if csv_path is None:
            # Try 2025 calendar first (for dates in 2025+)
            calendar_2025 = "/opt/nifty-data-collector/data/NSE_Market_Calendar_2025.csv"
            calendar_2024_25 = "/opt/nifty-data-collector/data/NSE_Market_Calendar_2024_25.csv"
            
            # For 2025 and beyond, use 2025 calendar; otherwise use 2024_25
            if today.year >= 2025:
                csv_path = calendar_2025
            else:
                csv_path = calendar_2024_25
        
        with open(csv_path, "r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                ds = str(row.get("Date", "")).strip()
                itd = str(row.get("Is_Trading_Day", "")).strip().upper()
                
                # Try parsing CSV date and comparing date objects (more robust)
                csv_date = None
                # Try ISO format first (2024_25 calendar: 2024-04-01)
                try:
                    csv_date = datetime.strptime(ds, "%Y-%m-%d").date()
                except:
                    # Try M/D/YYYY format (2025 calendar: 12/31/2025 or 1/1/2025)
                    try:
                        csv_date = datetime.strptime(ds, "%m/%d/%Y").date()
                    except:
                        # Try DD-MM-YYYY format
                        try:
                            csv_date = datetime.strptime(ds, "%d-%m-%Y").date()
                        except:
                            # Try DD/MM/YYYY format
                            try:
                                csv_date = datetime.strptime(ds, "%d/%m/%Y").date()
                            except:
                                pass
                
                # If date matches today, return trading status
                if csv_date and csv_date == today:
                    return itd != "FALSE"  # trading day if not FALSE
    except FileNotFoundError:
        # If 2025 calendar not found, try 2024_25 as fallback
        if csv_path and "2025.csv" in csv_path:
            print(f"⚠️ 2025 calendar not found, trying 2024_25 calendar...")
            return is_trading_day_from_csv("/opt/nifty-data-collector/data/NSE_Market_Calendar_2024_25.csv")
        print(f"⚠️ Holiday CSV check failed: Calendar file not found")
    except Exception as e:
        print(f"⚠️ Holiday CSV check failed: {e}")
    # If CSV missing/error, default to weekday/time checks
    return True

def get_spot_ltp():
    """
    Get NIFTY spot LTP with fallback mechanisms.
    
    Returns:
        float: Current NIFTY spot price or None if unavailable
    """
    if NIFTY_INDEX_TOKEN:
        try:
            print(f"🔍 Fetching NIFTY spot price using token: {NIFTY_INDEX_TOKEN}")
            resp = obj.getMarketData("LTP", {"NSE": [NIFTY_INDEX_TOKEN]})
            print(f"📊 API Response: {resp}")
            
            if isinstance(resp, dict) and resp.get("data"):
                data = resp["data"]
                if isinstance(data, dict) and data.get("fetched"):
                    fetched_data = data["fetched"]
                    if fetched_data and len(fetched_data) > 0:
                        ltp = fetched_data[0].get("ltp")
                        if ltp: 
                            print(f"✅ NIFTY spot LTP: {ltp}")
                            return float(ltp)
                        else:
                            print(f"❌ No LTP in fetched data: {fetched_data}")
                    else:
                        print(f"❌ No fetched data available")
                else:
                    print(f"❌ Unexpected data structure: {data}")
            else:
                print(f"❌ Invalid response format: {resp}")
        except Exception as e:
            print(f"❌ Error fetching spot LTP: {e}")
    
    # Fallback: Use a default value for testing when market is closed
    print("⚠️ Using fallback NIFTY price (market may be closed)")
    return 24750.0  # Default fallback price

def pick_watchlist(atm_window=ATM_WINDOW):
    """
    Automatically select strike prices based on ATM (At-The-Money) calculation.
    
    Selection Logic:
    1. Get current NIFTY spot price
    2. Round to nearest 50 to find ATM strike (e.g., 24750 if spot is 24723)
    3. Select strikes within ±ATM_WINDOW range (default ±5 strikes)
    4. Include both CE (Call) and PE (Put) options for each strike
    5. Include NIFTY Futures for current expiry
    
    Example: If ATM is 24750 and ATM_WINDOW=5:
    - Selected strikes: 24500, 24550, 24600, 24650, 24700, 24750, 24800, 24850, 24900, 24950, 25000
    - Total symbols: 23 (11 strikes × 2 options each + 1 future)
    """
    ltp = get_spot_ltp()
    if not ltp:
        raise RuntimeError("No spot LTP yet")
    atm = int(round(ltp / 50.0) * 50)
    exp = current_expiry_short
    desired = set()
    
    # Add NIFTY Options
    for off in range(-atm_window, atm_window + 1):
        strike = atm + off * 50
        ce = f"{SYMBOL_BASE}{exp}{int(strike):05d}CE"
        pe = f"{SYMBOL_BASE}{exp}{int(strike):05d}PE"
        if ce in SYMBOL_TO_TOKEN: desired.add(ce)
        if pe in SYMBOL_TO_TOKEN: desired.add(pe)
    
    # Add NIFTY Spot
    desired.add("NIFTY 50")
    
    # Add NIFTY Futures (find any available futures)
    futures_symbols = [s for s in SYMBOL_TO_TOKEN.keys() if 'FUT' in s]
    if futures_symbols:
        # Add the first available futures symbol
        desired.add(futures_symbols[0])
        print(f"📈 Added NIFTY Future: {futures_symbols[0]}")
    
    return sorted(desired)

# ------- Simplified Data Collection -------
RUN_FLAG = threading.Event()

def fetch_market_data(symbols):
    """
    Fetch all market data using getMarketData("FULL") for all symbols.
    Returns complete data including LTP, Volume, OI, and Greeks.
    """
    print(f"📊 Fetching market data for {len(symbols)} symbols...")
    
    # Get tokens for different exchanges
    nfo_tokens = [SYMBOL_TO_TOKEN[s] for s in symbols if s != "NIFTY 50" and s in SYMBOL_TO_TOKEN]
    nse_tokens = [SYMBOL_TO_TOKEN[s] for s in symbols if s == "NIFTY 50" and s in SYMBOL_TO_TOKEN]
    
    all_data = {}
    
    try:
        # Fetch NFO data (options and futures)
        if nfo_tokens:
            resp_nfo = obj.getMarketData("FULL", {"NFO": nfo_tokens})
            if resp_nfo and isinstance(resp_nfo, dict) and resp_nfo.get('status'):
                data = resp_nfo.get("data", {})
                fetched = data.get("fetched") or data.get("data") or data.get("ltpData") or []
                for item in fetched:
                    if isinstance(item, dict):
                        symbol = item.get("tradingSymbol")
                        if symbol:
                            all_data[symbol] = item
                print(f"✅ Fetched {len(fetched)} NFO records")
        
        # Fetch NSE data (NIFTY 50 spot)
        if nse_tokens:
            resp_nse = obj.getMarketData("FULL", {"NSE": nse_tokens})
            if resp_nse and isinstance(resp_nse, dict) and resp_nse.get('status'):
                data = resp_nse.get("data", {})
                fetched = data.get("fetched") or data.get("data") or data.get("ltpData") or []
                
                # Debug: Log NSE data structure
                if fetched and len(fetched) > 0:
                    print(f"🔍 Debug - NSE data structure: {fetched[0]}")
                    print(f"🔍 Debug - NSE tradingSymbol: {fetched[0].get('tradingSymbol')}")
                
                for item in fetched:
                    if isinstance(item, dict):
                        symbol = item.get("tradingSymbol")
                        if symbol:
                            all_data[symbol] = item
                print(f"✅ Fetched {len(fetched)} NSE records")
        
        print(f"📊 Total market data records: {len(all_data)}")
        return all_data
        
    except Exception as e:
        print(f"❌ Error fetching market data: {e}")
        return {}

# ------- Simplified Data Collection -------

# ------- Signal Handling -------
def signal_handler(signum, frame):
    """Handle shutdown signals gracefully."""
    print(f"\n🛑 Received signal {signum} - initiating graceful shutdown...")
    RUN_FLAG.clear()
    print("✅ Signal handler completed")

# Register signal handlers
signal.signal(signal.SIGTERM, signal_handler)
signal.signal(signal.SIGINT, signal_handler)

# ------- Configurable interval writer -------
def insert_rows(table, rows):
    if not rows:
        return
    cols = list(rows[0].keys())
    placeholders = ", ".join([f":{c}" for c in cols])

    if _is_sqlite() and table in ("ltp_ticks", "oi_snapshots"):
        # Upsert on (symbol, ts)
        update_cols = [c for c in cols if c not in ("id", "symbol", "ts")]
        set_clause = ", ".join([f"{c}=excluded.{c}" for c in update_cols]) or "symbol=excluded.symbol"
        sql = (
            f"INSERT INTO {table} ({', '.join(cols)}) VALUES ({placeholders}) "
            f"ON CONFLICT(symbol, ts) DO UPDATE SET {set_clause}"
        )
    else:
        # Plain insert for MySQL/Postgres (or define vendor-specific upsert if needed)
        sql = f"INSERT INTO {table} ({', '.join(cols)}) VALUES ({placeholders})"

    with engine.begin() as conn:
        conn.execute(text(sql), rows)


def simplified_data_writer_loop(symbols):
    """
    Simplified data collection using ONLY getMarketData("FULL").
    
    This function:
    1. Fetches all data using getMarketData("FULL") every 5 seconds
    2. Extracts LTP, Volume, OI, and Greeks from single API call
    3. Applies proper scaling for options
    4. Writes directly to database
    5. Synchronizes to exact 5-second intervals (12:44:00, 12:44:05, 12:44:10, etc.)
    """
    print(f"⏱️ Simplified {DATA_STORAGE_INTERVAL_SECS}-second writer started")
    iteration_count = 0
    
    while RUN_FLAG.is_set():
        try:
            # Check if market is still open - if not, exit gracefully
            if not is_market_open():
                logger.info("🏁 Market has closed - stopping data collection")
                logger.info("📊 Final iteration completed, exiting gracefully...")
                RUN_FLAG.clear()
                break
            
            iteration_count += 1
            print(f"🔄 Writer iteration {iteration_count} starting...")
            
            t0 = time.time()
            # Synchronize to exact 5-second intervals (12:44:00, 12:44:05, 12:44:10, etc.)
            now_utc = datetime.now(timezone.utc)
            # Calculate the next exact 5-second boundary
            current_second = now_utc.second
            next_boundary_second = ((current_second // DATA_STORAGE_INTERVAL_SECS) + 1) * DATA_STORAGE_INTERVAL_SECS
            
            if next_boundary_second >= 60:
                # Move to next minute
                next_boundary_second = 0
                next_boundary = now_utc.replace(second=0, microsecond=0) + timedelta(minutes=1)
            else:
                next_boundary = now_utc.replace(second=next_boundary_second, microsecond=0)
            
            # Use the exact boundary timestamp
            now_iso = next_boundary.isoformat()
            batch = []

            # Fetch all market data in one call
            market_data = fetch_market_data(symbols)
            print(f"📊 Fetched data for {len(market_data)} symbols")

            # Fetch Greeks data for options (separate call)
            print(f"🔍 Starting Greeks data fetch...")
            greeks_data = fetch_greeks_data()
            print(f"📊 Fetched Greeks data for {len(greeks_data)} symbols")
            if greeks_data:
                print(f"✅ Greeks data keys: {list(greeks_data.keys())[:5]}...")  # Show first 5 keys
            else:
                print(f"⚠️ No Greeks data found")

            # Process each symbol
            for s in symbols:
                try:
                    # Try exact match first, then case-insensitive match
                    data = market_data.get(s)
                    if not data:
                        # Try case-insensitive match for NIFTY 50
                        if s == "NIFTY 50":
                            for key, value in market_data.items():
                                if key.lower() == "nifty 50":
                                    data = value
                                    print(f"✅ Found NIFTY 50 data with key: {key}")
                                    break
                    
                    if not data:
                        print(f"⚠️ No data for {s}")
                        continue
                    
                    tok = SYMBOL_TO_TOKEN.get(s, "")
                    
                    # Extract basic data from getMarketData response
                    raw_ltp = data.get('ltp')
                    volume = int(data.get('totaltradedvolume') or data.get('tradeVolume') or data.get('volume') or 0)
                    oi = int(data.get('opnInterest') or data.get('oi') or 0)
                    
                    # Get Greeks data from separate call (if available)
                    greeks = greeks_data.get(s, {})
                    delta = float(greeks.get('delta', 0.0))
                    gamma = float(greeks.get('gamma', 0.0))
                    theta = float(greeks.get('theta', 0.0))
                    vega = float(greeks.get('vega', 0.0))
                    iv = float(greeks.get('iv', 0.0)) # Get IV
                    
                    # Log Greeks data for options
                    if 'CE' in s or 'PE' in s:
                        logger.info(f"🔍 {s}: Δ={delta:.4f}, Γ={gamma:.4f}, Θ={theta:.4f}, ν={vega:.4f}, IV={iv:.4f}")
                    
                    # Store raw LTP values exactly as received from API - no scaling
                    final_ltp = None
                    if raw_ltp is not None:
                        final_ltp = float(raw_ltp)
                        logger.info(f"🔍 {s}: Raw={raw_ltp}, Stored={final_ltp}")
                    
                    row = {
                        "symbol": s,
                        "token": tok,
                        "ts": now_iso,
                        "ltp": final_ltp,
                        "bid": 0.0,  # Not available in getMarketData
                        "ask": 0.0,  # Not available in getMarketData
                        "volume": volume,
                        "oi": oi,
                        "delta": delta,
                        "gamma": gamma,
                        "theta": theta,
                        "vega": vega,
                        "iv": iv, # Add IV to the row
                        "source": "api",
                    }
                    batch.append(row)

                except Exception as e:
                    print(f"❌ Error processing symbol {s}: {e}")
                    continue

            # Write to database only if market is open
            if batch:
                if is_market_open():
                    try:
                        logger.info(f"💾 Writing {len(batch)} records to database...")
                        insert_rows("ltp_ticks", batch)
                        logger.info(f"✅ Successfully wrote {len(batch)} records")
                    except Exception as e:
                        print(f"❌ DB insert error: {e}")
                        import traceback
                        traceback.print_exc()
                else:
                    print("⏸️ Market closed - skipping database write (data would be stale)")
            else:
                print("⚠️ No data to write in this iteration")
                
            # Calculate sleep time to synchronize with exact 5-second boundaries
            dt = time.time() - t0
            # Calculate time until next exact boundary
            current_time = time.time()
            next_boundary_time = next_boundary.timestamp()
            sleep_time = max(0.0, next_boundary_time - current_time)
            
            logger.info(f"⏱️ Iteration {iteration_count} completed in {dt:.2f}s, sleeping for {sleep_time:.2f}s until {next_boundary.strftime('%H:%M:%S')}")
            time.sleep(sleep_time)
            
        except Exception as e:
            print(f"❌ Critical error in writer loop iteration {iteration_count}: {e}")
            import traceback
            traceback.print_exc()
            print("🔄 Continuing to next iteration...")
            time.sleep(1)  # Brief pause before retry

# ------- Greeks fetching -------
GREEKS_CACHE = {}
GREEKS_CACHE_LOCK = threading.Lock()
LAST_GREEKS_UPDATE = 0
GREEKS_UPDATE_INTERVAL = 60  # Update Greeks every 60 seconds

def fetch_greeks_data():
    """
    Fetch live Greeks data using optionGreek API.
    Returns DataFrame with Greeks data for current expiry.
    """
    global LAST_GREEKS_UPDATE
    
    # Check if market is open before making API calls
    if not is_market_open():
        print("⏸️ Market closed - returning cached Greeks data")
        with GREEKS_CACHE_LOCK:
            return GREEKS_CACHE.copy()
    
    current_time = time.time()
    if current_time - LAST_GREEKS_UPDATE < GREEKS_UPDATE_INTERVAL:
        # Return cached data if recent
        with GREEKS_CACHE_LOCK:
            return GREEKS_CACHE.copy()
    
    try:
        # Convert expiry format from 14AUG25 to 14AUG2025 or 14AUG2026 (dynamically determine year)
        if len(current_expiry_short) == 7:  # Format: 14AUG25
            try:
                current_year = datetime.now().year
                current_date = datetime.now()
                
                # Try current year first
                exp_date_current = datetime.strptime(current_expiry_short[:-2] + str(current_year)[-2:], '%d%b%y')
                
                # Try next year
                exp_date_next = datetime.strptime(current_expiry_short[:-2] + str(current_year + 1)[-2:], '%d%b%y')
                
                # Choose the expiry that is in the future (or today)
                if exp_date_current >= current_date:
                    year = str(current_year)
                elif exp_date_next >= current_date:
                    year = str(current_year + 1)
                else:
                    # Both are in past, use next year (shouldn't happen for valid expiries)
                    year = str(current_year + 1)
                
                expiry_for_greeks = current_expiry_short[:-2] + year
            except ValueError:
                # Fallback to current year if parsing fails
                expiry_for_greeks = current_expiry_short[:-2] + str(datetime.now().year)
        else:
            expiry_for_greeks = current_expiry_short
        
        print(f"🔄 Fetching Greeks data for expiry: {expiry_for_greeks}")
        
        greek_param = {"name": "NIFTY", "expirydate": expiry_for_greeks}
        greek_res = obj.optionGreek(greek_param)
        
        if greek_res.get('status', False):
            data = greek_res.get('data', [])
            if data:
                df_greeks = pd.DataFrame(data)
                df_greeks['strike'] = pd.to_numeric(df_greeks['strikePrice'], errors='coerce')
                df_greeks['optionType'] = df_greeks['optionType'].str.upper()
                
                # Convert Greeks to numeric
                for col in ['delta', 'gamma', 'vega', 'theta', 'impliedVolatility']:
                    if col in df_greeks.columns:
                        df_greeks[col] = pd.to_numeric(df_greeks[col], errors='coerce')
                
                # Create symbol mapping for Greeks
                greeks_dict = {}
                for _, row in df_greeks.iterrows():
                    strike = int(row['strike'])
                    option_type = row['optionType']
                    symbol = f"NIFTY{current_expiry_short}{strike:05d}{option_type}"
                    
                    greeks_dict[symbol] = {
                        'delta': row.get('delta', 0.0),
                        'gamma': row.get('gamma', 0.0),
                        'theta': row.get('theta', 0.0),
                        'vega': row.get('vega', 0.0),
                        'iv': row.get('impliedVolatility', 0.0)
                    }
                
                # Update cache
                with GREEKS_CACHE_LOCK:
                    GREEKS_CACHE.clear()
                    GREEKS_CACHE.update(greeks_dict)
                
                LAST_GREEKS_UPDATE = current_time
                print(f"✅ Greeks data updated for {len(greeks_dict)} symbols")
                return greeks_dict
            else:
                print("❌ No Greeks data in API response")
        else:
            error_msg = greek_res.get('message', 'Unknown error')
            print(f"❌ Greeks API failed: {error_msg}")
            
    except Exception as e:
        print(f"❌ Error fetching Greeks: {e}")
    
    # Return cached data if available
    with GREEKS_CACHE_LOCK:
        return GREEKS_CACHE.copy()

def get_greeks_for_symbol(symbol):
    """Get Greeks data for a specific symbol."""
    greeks_data = fetch_greeks_data()
    return greeks_data.get(symbol, {'delta': 0.0, 'gamma': 0.0, 'theta': 0.0, 'vega': 0.0, 'iv': 0.0})

# ------- Main -------
def main():
    # Create logs directory at start so prints always land somewhere when run outside Supervisor
    os.makedirs("/opt/nifty-data-collector/logs", exist_ok=True)
    
    print("🕐 Checking market status...")
    # Holiday guard first
    if not is_trading_day_from_csv():
        print("📅 Today is a non-trading day per calendar CSV — exiting.")
        return

    if not is_market_open():
        print("🏁 Market is currently closed — exiting")
        return
    
    print("🟢 Market is open — proceeding with data collection")
    
    # Display configuration summary
    print("=" * 60)
    print("🎯 NIFTY STREAM CONFIGURATION")
    print("=" * 60)
    print(f"📊 Symbol Base: {SYMBOL_BASE}")
    print(f"🎯 ATM Window: ±{ATM_WINDOW} strikes")
    print(f"⏱️ Data Storage Interval: {DATA_STORAGE_INTERVAL_SECS} seconds")
    print(f"🔄 OI Polling Interval: {OI_POLL_SECS} seconds")
    print(f"💾 Database: {DB_URL}")
    print(f"🔑 User ID: {ANGEL_USER_ID}")
    print("=" * 60)
    
    # Check market hours again for status display
    market_status = "🟢 OPEN" if is_market_open() else "🔴 CLOSED"
    print(f"📈 Market Status: {market_status}")
    
    # Initialize database with retry logic
    try:
        if not init_db():
            print("❌ Database initialization failed - exiting")
            return
    except Exception as e:
        print(f"❌ Critical database error: {e}")
        return
    
    # Angel login with retry logic
    max_login_retries = 3
    for login_attempt in range(max_login_retries):
        try:
            angel_login()
            break
        except Exception as e:
            print(f"❌ Login error (attempt {login_attempt + 1}/{max_login_retries}): {e}")
            if login_attempt < max_login_retries - 1:
                print("🔄 Retrying login...")
                time.sleep(5)
                continue
            else:
                print("❌ Failed to login after all retries - exiting")
                return
    
    # Fetch instruments with retry logic
    max_instrument_retries = 3
    for instrument_attempt in range(max_instrument_retries):
        try:
            fetch_instruments_and_map()
            break
        except Exception as e:
            print(f"❌ Instrument mapping error (attempt {instrument_attempt + 1}/{max_instrument_retries}): {e}")
            if instrument_attempt < max_instrument_retries - 1:
                print("🔄 Retrying instrument mapping...")
                time.sleep(5)
                continue
            else:
                print("❌ Failed to map instruments after all retries - exiting")
                return
    
    try:
        symbols = pick_watchlist(ATM_WINDOW)
        print(f"🎯 Watchlist ({len(symbols)}): {symbols}")

        RUN_FLAG.set()

        try:
            simplified_data_writer_loop(symbols)
        except KeyboardInterrupt:
            print("Interrupted, stopping...")
        finally:
            logger.info("🔄 Cleaning up and closing connections...")
            RUN_FLAG.clear()
            try:
                # Close database connection if exists
                if 'engine' in globals():
                    engine.dispose()
                    logger.info("💾 Database connection closed")
                
                # Clear any remaining connections
                import sqlite3
                try:
                    sqlite3.connect(DB_URL.replace('sqlite:///', '')).close()
                    logger.info("💾 SQLite connections cleared")
                except:
                    pass
                    
                logger.info("✅ Cleanup completed - program exiting gracefully")
            except Exception as e:
                print(f"⚠️ Error during cleanup: {e}")
                import traceback
                traceback.print_exc()
    except Exception as e:
        print(f"❌ Error in main execution: {e}")
        print("💡 This might be due to market being closed or API issues.")
        print("   Try running during market hours (9:15 AM - 3:30 PM IST) for full functionality.")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()