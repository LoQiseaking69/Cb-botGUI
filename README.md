# Automated Crypto Trading System

## Overview
This is an **AI-powered cryptocurrency trading bot** that integrates **Coinbase Pro API**, an **LSTM-based prediction model**, and **risk management strategies**. The system uses **real-time WebSocket data**, enforces **trade limits**, and provides a **GUI for manual monitoring and control**.

## Features
- **AI-Powered Trading** – Uses an LSTM model to predict future price movements.
- **Real-Time Data Streaming** – Fetches live market data via WebSockets.
- **Automated Trade Execution** – Places buy/sell orders based on predictions.
- **Risk Management** – Implements stop-loss, take-profit, and position limits.
- **GUI Dashboard** – Allows manual control and monitoring of trades.
- **Trade Logging** – Stores executed trades in an SQLite database.

## Installation

### Prerequisites
- Python 3.8+
- A **Coinbase Pro API key** (Requires API credentials for trading)
- Required Python libraries

### Setup Instructions

1. **Clone the Repository**
   ```bash
   git clone https://github.com/LoQiseaking69/Cb-botGUI.git
   cd Cb-botGUI
   ```

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set Up API Keys**  
   Create a `.env` file and add your **Coinbase API credentials**:
   ```plaintext
   COINBASE_API_KEY=your_api_key
   COINBASE_API_SECRET=your_api_secret
   COINBASE_API_PASSPHRASE=your_passphrase
   ```

4. **Initialize Database**  
   ```bash
   python trading_bot.py --init-db
   ```

5. **Start the Trading Bot**  
   ```bash
   python trading_gui.py
   ```

## Usage

- **Trading Starts Automatically** once enabled in the GUI.
- **Trades are logged** in `trades.db`.
- **Risk management rules** are enforced automatically.

## Risk Management
- **Max Position Size:** Prevents excessive trade sizes.
- **Max Concurrent Trades:** Limits the number of open trades.
- **Stop-Loss & Take-Profit:** Automatically exits trades to prevent excessive loss.

## File Structure

```
/trading-bot
│── trading_bot.py           # Main trading logic (API, AI model, execution)
│── risk_management.py       # Enforces risk limits and trade safety
│── trading_gui.py           # GUI for trade monitoring and control
│── trades.db                # SQLite database storing executed trades
│── .env                     # API credentials (ignored in version control)
│── requirements.txt         # Required dependencies
│── README.md                # This documentation
```

## License
This project is licensed under the **MIT License**.

## Disclaimer
⚠ **Use this software at your own risk.** Trading cryptocurrencies involves significant financial risk. Ensure you understand the risks before using this system in live trading.

