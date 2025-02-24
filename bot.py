import os
import time
import hmac
import hashlib
import requests
import json
import numpy as np
import pandas as pd
import tensorflow as tf
import asyncio
import logging
import threading
import websockets
import dotenv
import pickle
import sqlite3
from collections import deque
from datetime import datetime
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, LayerNormalization
from sklearn.preprocessing import MinMaxScaler

# Load environment variables
dotenv.load_dotenv()

# Logging setup
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# API Credentials
API_KEY = os.getenv("COINBASE_API_KEY")
API_PASSPHRASE = os.getenv("COINBASE_API_PASSPHRASE")
API_SECRET = os.getenv("COINBASE_API_SECRET")
API_URL = "https://api.pro.coinbase.com"

if not API_KEY or not API_SECRET or not API_PASSPHRASE:
    logging.error("Missing API credentials! Ensure .env file is properly set.")
    exit()

# Trading parameters
LOOKBACK = 50
STOP_LOSS_PERCENT = 0.02
TAKE_PROFIT_PERCENT = 0.05
LEARNING_RATE = 0.001
EPOCHS = 5  # Reduced for incremental learning
BATCH_SIZE = 16
SCALER_FILE = "scaler.pkl"
MODEL_FILE = "lstm_model.h5"

# Trading state
trading_active = False
data_buffer = deque(maxlen=1000)

# Initialize Database
def init_db():
    conn = sqlite3.connect("trades.db")
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            asset TEXT,
            side TEXT,
            amount REAL,
            price REAL,
            profit_loss REAL
        )
    """)
    conn.commit()
    conn.close()

def log_trade(asset, side, amount, price, profit_loss):
    """Safely log trades only if execution was successful."""
    if price > 0:
        conn = sqlite3.connect("trades.db")
        cursor = conn.cursor()
        cursor.execute("INSERT INTO trades (timestamp, asset, side, amount, price, profit_loss) VALUES (?, ?, ?, ?, ?, ?)", 
                       (datetime.utcnow().isoformat(), asset, side, amount, price, profit_loss))
        conn.commit()
        conn.close()
    else:
        logging.error(f"Trade execution failed, not logging trade.")

def get_headers(method, path, body=""):
    """Create secure headers for Coinbase API authentication."""
    timestamp = str(int(time.time()))
    message = timestamp + method + path + body
    signature = hmac.new(API_SECRET.encode(), message.encode(), hashlib.sha256).hexdigest()
    return {
        "CB-ACCESS-KEY": API_KEY,
        "CB-ACCESS-SIGN": signature,
        "CB-ACCESS-TIMESTAMP": timestamp,
        "CB-ACCESS-PASSPHRASE": API_PASSPHRASE,
        "Content-Type": "application/json"
    }

def get_historical_data(asset, granularity=300):
    """Fetch historical market data from Coinbase."""
    path = f"/products/{asset}-USD/candles?granularity={granularity}"
    response = requests.get(API_URL + path, headers=get_headers("GET", path))
    if response.status_code == 200:
        data = response.json()
        df = pd.DataFrame(data, columns=["time", "low", "high", "open", "close", "volume"])
        df["time"] = pd.to_datetime(df["time"], unit="s")
        df.sort_values("time", inplace=True)
        return df
    else:
        logging.error(f"Failed to fetch market data: {response.text}")
        return None

def train_or_update_model():
    """Train or update model using real historical data for incremental learning."""
    df = get_historical_data("BTC")
    if df is None:
        return

    X_train, y_train, scaler = preprocess_data(df)

    if os.path.exists(MODEL_FILE):
        logging.info("Updating existing model with new data...")
        model = load_model(MODEL_FILE)
    else:
        logging.info("Training new LSTM model from scratch...")
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=(LOOKBACK, 1)),
            LayerNormalization(),
            Dropout(0.2),
            LSTM(50, return_sequences=False),
            Dense(25, activation="relu"),
            Dense(1)
        ])
        model.compile(optimizer=tf.keras.optimizers.Adam(LEARNING_RATE), loss="mse")

    model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=1)
    model.save(MODEL_FILE)
    pickle.dump(scaler, open(SCALER_FILE, "wb"))

    logging.info("Model training/update complete!")

def preprocess_data(df):
    """Prepare data for training and prediction."""
    df["close"] = df["close"].astype(float)
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df["close"].values.reshape(-1, 1))

    X, y = [], []
    for i in range(LOOKBACK, len(scaled_data)):
        X.append(scaled_data[i-LOOKBACK:i])
        y.append(scaled_data[i])
    
    return np.array(X), np.array(y), scaler

def predict_price():
    """Predict the next price using the trained model."""
    if len(data_buffer) < LOOKBACK or not os.path.exists(MODEL_FILE):
        logging.warning("Insufficient data or model missing! Training now...")
        train_or_update_model()
    
    model = load_model(MODEL_FILE)
    scaler = pickle.load(open(SCALER_FILE, "rb"))
    recent_data = np.array(data_buffer)[-LOOKBACK:].reshape(-1, 1)
    scaled_data = scaler.transform(recent_data)
    prediction = model.predict(np.array([scaled_data]))[0][0]
    return scaler.inverse_transform([[prediction]])[0][0]

async def trade():
    """Automated trading loop."""
    global trading_active
    trading_active = True
    assets = ["BTC", "ETH"]
    queue = asyncio.Queue()
    asyncio.create_task(fetch_market_data_ws(assets, queue))

    while trading_active:
        asset, price = await queue.get()
        data_buffer.append(price)
        if len(data_buffer) >= LOOKBACK:
            prediction = predict_price()
            if prediction:
                logging.info(f"{asset} - Prediction: {prediction}")
                trade_size = 0.01
                await execute_trade(asset, "buy" if prediction > price else "sell", trade_size)

async def fetch_market_data_ws(assets, queue):
    """Handles real-time WebSocket price streaming with robust reconnection."""
    reconnect_attempts = 0
    while trading_active:
        try:
            async with websockets.connect("wss://ws-feed.pro.coinbase.com") as ws:
                request = json.dumps({"type": "subscribe", "channels": [{"name": "ticker", "product_ids": [f"{asset}-USD" for asset in assets]}]})
                await ws.send(request)
                while trading_active:
                    response = await ws.recv()
                    market_data = json.loads(response)
                    if "price" in market_data:
                        asset = market_data["product_id"].split("-")[0]
                        price = float(market_data["price"])
                        await queue.put((asset, price))
        except Exception as e:
            reconnect_attempts += 1
            if reconnect_attempts > 5:
                logging.error("Max WebSocket reconnect attempts reached. Stopping.")
                break
            logging.error(f"WebSocket error: {e}, retrying in 5s...")
            await asyncio.sleep(5)

def stop_trading():
    global trading_active
    trading_active = False
    logging.info("Trading stopped.")

# Initialize database and start trading
init_db()
