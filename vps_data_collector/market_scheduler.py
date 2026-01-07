#!/usr/bin/env python3
"""
Market-Aware NIFTY Data Collection Scheduler
===========================================

This script manages the NIFTY data collection service based on market hours:
- Starts data collection at market open (09:15 IST)
- Stops data collection at market close (15:30 IST)
- Skips weekends and holidays automatically
- Handles supervisor service management

Usage:
    python3 market_scheduler.py start   # Start data collection (market open)
    python3 market_scheduler.py stop    # Stop data collection (market close)
    python3 market_scheduler.py status  # Check current status
"""

import sys
import os
import subprocess
import time
from datetime import datetime, timedelta
import csv
import logging

# Configuration
PROJECT_DIR = "/opt/nifty-data-collector"
LOG_DIR = f"{PROJECT_DIR}/logs"
CALENDAR_FILE = f"{PROJECT_DIR}/data/NSE_Market_Calendar_2025.csv"
SERVICE_NAME = "nifty-data-collector"

# Ensure log directory exists
os.makedirs(LOG_DIR, exist_ok=True)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"{LOG_DIR}/market_scheduler.log"),
        logging.StreamHandler()
    ]
)

def load_holiday_calendar():
    """Load NSE holiday calendar from CSV file."""
    holidays = set()
    try:
        with open(CALENDAR_FILE, 'r') as file:
            reader = csv.DictReader(file)
            for row in reader:
                if row.get('Is_Trading_Day', '').strip().upper() == 'FALSE':
                    date_str = row.get('Date', '').strip()
                    if date_str:
                        # Parse date (assuming format: DD-MM-YYYY or similar)
                        try:
                            # Try different date formats
                            for fmt in ['%m/%d/%Y', '%d-%m-%Y', '%Y-%m-%d', '%d/%m/%Y']:
                                try:
                                    holiday_date = datetime.strptime(date_str, fmt).date()
                                    holidays.add(holiday_date)
                                    break
                                except ValueError:
                                    continue
                        except ValueError:
                            logging.warning(f"Could not parse date: {date_str}")
        
        logging.info(f"Loaded {len(holidays)} holiday dates from calendar")
        return holidays
    except FileNotFoundError:
        logging.error(f"Holiday calendar file not found: {CALENDAR_FILE}")
        return set()
    except Exception as e:
        logging.error(f"Error loading holiday calendar: {e}")
        return set()

def is_market_open_today():
    """Check if market should be open today."""
    # Get current IST time
    utc_now = datetime.utcnow()
    ist_now = utc_now + timedelta(hours=5, minutes=30)
    today = ist_now.date()
    
    # Check if it's a weekday (Monday=0, Sunday=6)
    if ist_now.weekday() >= 5:  # Saturday=5, Sunday=6
        logging.info("Market closed: Weekend")
        return False
    
    # Check if it's a holiday
    holidays = load_holiday_calendar()
    if today in holidays:
        logging.info("Market closed: Holiday")
        return False
    
    logging.info("Market should be open today")
    return True

def get_service_status():
    """Get the current status of the data collection service."""
    try:
        result = subprocess.run(['supervisorctl', 'status', SERVICE_NAME], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            status_line = result.stdout.strip()
            if 'RUNNING' in status_line:
                return 'RUNNING'
            elif 'STOPPED' in status_line:
                return 'STOPPED'
            elif 'FATAL' in status_line:
                return 'FATAL'
            elif 'STARTING' in status_line:
                return 'STARTING'
            elif 'STOPPING' in status_line:
                return 'STOPPING'
            else:
                return 'UNKNOWN'
        else:
            logging.error(f"Failed to get service status: {result.stderr}")
            return 'ERROR'
    except subprocess.TimeoutExpired:
        logging.error("Timeout getting service status")
        return 'TIMEOUT'
    except Exception as e:
        logging.error(f"Error checking service status: {e}")
        return 'ERROR'

def start_data_collection():
    """Start the data collection service."""
    if not is_market_open_today():
        logging.info("Market is closed today - not starting data collection")
        return False
    
    # Get current IST time
    utc_now = datetime.utcnow()
    ist_now = utc_now + timedelta(hours=5, minutes=30)
    current_time = ist_now.strftime('%H:%M:%S')
    
    # Check if we're within market hours (09:15 - 15:30)
    market_start = "09:15:00"
    market_end = "15:30:00"
    
    if not (market_start <= current_time <= market_end):
        logging.info(f"Outside market hours ({current_time}) - not starting data collection")
        return False
    
    # Check current service status
    status = get_service_status()
    if status == 'RUNNING':
        logging.info("Data collection service is already running")
        return True
    
    # Retry logic for service start
    max_retries = 3
    for attempt in range(max_retries):
        try:
            logging.info(f"Starting NIFTY data collection service... (attempt {attempt + 1}/{max_retries})")
            result = subprocess.run(['supervisorctl', 'start', SERVICE_NAME], 
                                  capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                logging.info("✅ Data collection service start command successful")
                
                # Wait and verify it's running
                for check in range(6):  # Check for up to 30 seconds
                    time.sleep(5)
                    current_status = get_service_status()
                    if current_status == 'RUNNING':
                        logging.info("✅ Service confirmed running")
                        return True
                    elif current_status == 'FATAL':
                        logging.error("❌ Service entered FATAL state")
                        break
                    else:
                        logging.info(f"⏳ Service status: {current_status}, waiting...")
                
                logging.error("❌ Service failed to start properly after 30 seconds")
                if attempt < max_retries - 1:
                    logging.info("🔄 Retrying service start...")
                    continue
                return False
            else:
                logging.error(f"❌ Failed to start service: {result.stderr}")
                if attempt < max_retries - 1:
                    logging.info("🔄 Retrying service start...")
                    continue
                return False
                
        except subprocess.TimeoutExpired:
            logging.error(f"❌ Timeout starting service (attempt {attempt + 1})")
            if attempt < max_retries - 1:
                continue
            return False
        except Exception as e:
            logging.error(f"❌ Error starting data collection: {e}")
            if attempt < max_retries - 1:
                continue
            return False
    
    return False

def stop_data_collection():
    """Stop the data collection service."""
    status = get_service_status()
    if status == 'STOPPED':
        logging.info("Data collection service is already stopped")
        return True
    
    # Retry logic for service stop
    max_retries = 3
    for attempt in range(max_retries):
        try:
            logging.info(f"Stopping NIFTY data collection service... (attempt {attempt + 1}/{max_retries})")
            result = subprocess.run(['supervisorctl', 'stop', SERVICE_NAME], 
                                  capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                logging.info("✅ Data collection service stop command successful")
                
                # Wait and verify it's stopped
                for check in range(6):  # Check for up to 30 seconds
                    time.sleep(5)
                    current_status = get_service_status()
                    if current_status == 'STOPPED':
                        logging.info("✅ Service confirmed stopped")
                        return True
                    elif current_status == 'FATAL':
                        logging.error("❌ Service entered FATAL state")
                        break
                    else:
                        logging.info(f"⏳ Service status: {current_status}, waiting...")
                
                logging.warning("⚠️ Service may not have stopped cleanly after 30 seconds")
                if attempt < max_retries - 1:
                    logging.info("🔄 Retrying service stop...")
                    continue
                return False
            else:
                logging.error(f"❌ Failed to stop service: {result.stderr}")
                if attempt < max_retries - 1:
                    logging.info("🔄 Retrying service stop...")
                    continue
                return False
                
        except subprocess.TimeoutExpired:
            logging.error(f"❌ Timeout stopping service (attempt {attempt + 1})")
            if attempt < max_retries - 1:
                continue
            return False
        except Exception as e:
            logging.error(f"❌ Error stopping data collection: {e}")
            if attempt < max_retries - 1:
                continue
            return False
    
    return False

def show_status():
    """Show current system and market status."""
    # Get current IST time
    utc_now = datetime.utcnow()
    ist_now = utc_now + timedelta(hours=5, minutes=30)
    
    print(f"\n🕐 Current Time: {ist_now.strftime('%Y-%m-%d %H:%M:%S')} IST")
    print(f"📅 Day: {ist_now.strftime('%A')}")
    
    # Market status
    market_open = is_market_open_today()
    current_time = ist_now.strftime('%H:%M:%S')
    market_start = "09:15:00"
    market_end = "15:30:00"
    
    if not market_open:
        if ist_now.weekday() >= 5:
            print("📈 Market Status: 🔴 CLOSED (Weekend)")
        else:
            print("📈 Market Status: 🔴 CLOSED (Holiday)")
    elif market_start <= current_time <= market_end:
        print("📈 Market Status: 🟢 OPEN")
    else:
        print("📈 Market Status: 🔴 CLOSED (Outside market hours)")
    
    # Service status
    service_status = get_service_status()
    if service_status == 'RUNNING':
        print("🤖 Service Status: ✅ RUNNING")
    elif service_status == 'STOPPED':
        print("🤖 Service Status: ⏹️ STOPPED")
    elif service_status == 'FATAL':
        print("🤖 Service Status: ❌ FATAL")
    else:
        print(f"🤖 Service Status: ⚠️ {service_status}")
    
    # Next market session
    if market_open and current_time < market_start:
        print(f"⏰ Market Opens: Today at {market_start}")
    elif market_open and current_time > market_end:
        print("⏰ Next Market: Tomorrow 09:15 IST")
    elif not market_open:
        # Find next trading day
        next_day = ist_now + timedelta(days=1)
        holidays = load_holiday_calendar()
        
        while next_day.weekday() >= 5 or next_day.date() in holidays:
            next_day += timedelta(days=1)
        
        print(f"⏰ Next Market: {next_day.strftime('%A %d %B %Y')} 09:15 IST")
    
    print()

def main():
    """Main scheduler function."""
    if len(sys.argv) != 2:
        print("Usage: python3 market_scheduler.py [start|stop|status]")
        sys.exit(1)
    
    command = sys.argv[1].lower()
    
    if command == 'start':
        logging.info("Market scheduler: START command received")
        success = start_data_collection()
        sys.exit(0 if success else 1)
        
    elif command == 'stop':
        logging.info("Market scheduler: STOP command received")
        success = stop_data_collection()
        sys.exit(0 if success else 1)
        
    elif command == 'status':
        show_status()
        sys.exit(0)
        
    else:
        print(f"Unknown command: {command}")
        print("Usage: python3 market_scheduler.py [start|stop|status]")
        sys.exit(1)

if __name__ == "__main__":
    main()
