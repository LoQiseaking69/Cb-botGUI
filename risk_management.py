import os
import logging
import sqlite3
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Logging setup
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Risk Parameters
MAX_POSITION_SIZE = 0.1  # Maximum allowed trade size per order
MAX_CONCURRENT_TRADES = 5  # Max open positions
STOP_LOSS_PERCENT = 0.02  # Stop-loss at 2% drop
TAKE_PROFIT_PERCENT = 0.05  # Take profit at 5% gain
POSITION_COOLDOWN = 60  # Time before a new trade can be placed on the same asset (in seconds)

# Database Connection
DB_FILE = "trades.db"

def check_open_positions(asset):
    """Check open positions to prevent overtrading."""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM trades WHERE asset = ? AND profit_loss IS NULL", (asset,))
    open_positions = cursor.fetchone()[0]
    conn.close()
    return open_positions < MAX_CONCURRENT_TRADES

def enforce_risk_management(asset, price, trade_size, side):
    """Prevent trades that exceed risk limits."""
    if trade_size > MAX_POSITION_SIZE:
        logging.warning(f"Trade size {trade_size} exceeds maximum allowed position size.")
        return False

    if not check_open_positions(asset):
        logging.warning(f"Max concurrent trades reached for {asset}. Skipping trade.")
        return False

    return True

def check_stop_loss_take_profit(asset, entry_price):
    """Check if an active trade should be closed based on risk limits."""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("SELECT price, side FROM trades WHERE asset = ? AND profit_loss IS NULL", (asset,))
    positions = cursor.fetchall()
    conn.close()

    for price, side in positions:
        if side == "buy":
            stop_loss = price * (1 - STOP_LOSS_PERCENT)
            take_profit = price * (1 + TAKE_PROFIT_PERCENT)
            if entry_price <= stop_loss or entry_price >= take_profit:
                return True
        elif side == "sell":
            stop_loss = price * (1 + STOP_LOSS_PERCENT)
            take_profit = price * (1 - TAKE_PROFIT_PERCENT)
            if entry_price >= stop_loss or entry_price <= take_profit:
                return True

    return False
