"""
Database diagnostic script to check data availability
"""
import sqlite3
from datetime import datetime

db_path = "G:/Projects/Centralize Data Centre/data/nifty_local.db"

try:
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    print("=" * 70)
    print("DATABASE DIAGNOSTIC")
    print("=" * 70)
    
    # Check NIFTY 50 data
    cursor.execute("SELECT MIN(ts), MAX(ts), COUNT(*) FROM ltp_ticks WHERE symbol='NIFTY 50'")
    result = cursor.fetchone()
    print(f"\nNIFTY 50 Index Data:")
    print(f"  Min Date: {result[0]}")
    print(f"  Max Date: {result[1]}")
    print(f"  Total Records: {result[2]:,}")
    
    # Check total symbols
    cursor.execute("SELECT COUNT(DISTINCT symbol) FROM ltp_ticks")
    print(f"\nTotal Unique Symbols: {cursor.fetchone()[0]:,}")
    
    # Top symbols
    cursor.execute("""
        SELECT symbol, COUNT(*) as cnt 
        FROM ltp_ticks 
        GROUP BY symbol 
        ORDER BY cnt DESC 
        LIMIT 15
    """)
    print(f"\nTop 15 Symbols by Record Count:")
    for row in cursor.fetchall():
        print(f"  {row[0]}: {row[1]:,} records")
    
    # Check recent NIFTY 50 data
    cursor.execute("""
        SELECT ts, ltp 
        FROM ltp_ticks 
        WHERE symbol='NIFTY 50' 
        ORDER BY ts DESC 
        LIMIT 10
    """)
    print(f"\nLatest 10 NIFTY 50 Ticks:")
    for row in cursor.fetchall():
        print(f"  {row[0]}: LTP={row[1]}")
    
    # Check options data
    cursor.execute("""
        SELECT COUNT(*) 
        FROM ltp_ticks 
        WHERE symbol LIKE 'NIFTY%CE' OR symbol LIKE 'NIFTY%PE'
    """)
    print(f"\nNIFTY Options Records: {cursor.fetchone()[0]:,}")
    
    # Sample options
    cursor.execute("""
        SELECT DISTINCT symbol 
        FROM ltp_ticks 
        WHERE (symbol LIKE 'NIFTY%CE' OR symbol LIKE 'NIFTY%PE')
        LIMIT 10
    """)
    print(f"\nSample Option Symbols:")
    for row in cursor.fetchall():
        print(f"  {row[0]}")
    
    conn.close()
    print("\n" + "=" * 70)
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
