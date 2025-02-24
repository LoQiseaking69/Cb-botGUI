import asyncio
import threading
import sqlite3
import logging
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from trading_bot import get_account_balances, trade, stop_trading
from risk_management import enforce_risk_management, check_stop_loss_take_profit  # NEW IMPORT
from datetime import datetime

class TradingBotGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Trading Dashboard")
        self.root.geometry("1200x750")

        self.trading_active = False
        self.loop = asyncio.new_event_loop()
        threading.Thread(target=self.loop.run_forever, daemon=True).start()

        self.create_tabs()
        self.create_dashboard_tab()
        self.create_trading_tab()
        self.create_history_tab()
        self.update_balances()
        self.update_price_chart()
        self.update_open_positions()
        self.update_history()

    def start_trading(self):
        """Start trading process with risk management."""
        if not self.trading_active:
            self.trading_active = True
            asyncio.run_coroutine_threadsafe(self.execute_trades(), self.loop)
            self.start_button["state"] = tk.DISABLED
            self.stop_button["state"] = tk.NORMAL
            self.log_message("Trading Started.")
            messagebox.showinfo("Trading Bot", "Trading Started.")

    async def execute_trades(self):
        """Execute trades while ensuring risk parameters."""
        while self.trading_active:
            try:
                trade_data = await trade()  # Fetch trade decision
                if trade_data:
                    asset, price, trade_size, side = trade_data
                    if enforce_risk_management(asset, price, trade_size, side):
                        if check_stop_loss_take_profit(asset, price):
                            self.log_message(f"Closing position due to stop-loss/take-profit: {asset}")
                            stop_trading()  # Close trade if risk parameters breached
                            continue
                        self.log_message(f"Trade executed: {side} {trade_size} {asset} at {price}")
            except Exception as e:
                self.log_message(f"Trade execution error: {e}")
            await asyncio.sleep(2)  # Avoid overloading API

    def stop_trading(self):
        """Stop trading process."""
        if self.trading_active:
            self.trading_active = False
            stop_trading()
            self.start_button["state"] = tk.NORMAL
            self.stop_button["state"] = tk.DISABLED
            self.log_message("Trading Stopped.")
            messagebox.showinfo("Trading Bot", "Trading Stopped.")

    def update_balances(self):
        """Fetch and display account balances."""
        try:
            balances = get_account_balances()
            self.balance_text.config(state=tk.NORMAL)
            self.balance_text.delete("1.0", tk.END)
            for asset, balance in balances.items():
                self.balance_text.insert(tk.END, f"{asset}: {balance:.4f}\n")
            self.balance_text.config(state=tk.DISABLED)
        except Exception as e:
            self.log_message(f"Failed to fetch balances: {e}")
        self.root.after(15000, self.update_balances)

    def update_open_positions(self):
        """Fetch and display open positions."""
        for row in self.positions_table.get_children():
            self.positions_table.delete(row)

        with sqlite3.connect("trades.db") as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT asset, side, amount, price FROM trades WHERE profit_loss IS NULL ORDER BY timestamp DESC LIMIT 10")
            rows = cursor.fetchall()

        for row in rows:
            self.positions_table.insert("", "end", values=row)

        self.root.after(30000, self.update_open_positions)

if __name__ == "__main__":
    root = tk.Tk()
    app = TradingBotGUI(root)
    root.mainloop()
