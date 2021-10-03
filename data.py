import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path

from dotenv import load_dotenv
from binance.client import Client

DATA_PATH = Path('data')
LENGTH = '7 day'
INTERVAL = Client.KLINE_INTERVAL_1MINUTE
DATA_FILE = 'bnbusdt_' + INTERVAL

API = None # Client

def error(message: str):
  print(f'ERROR: {message}')
  exit(1)

def load_api_client() -> Client:
  global API

  if not API:
    print("Loading Binance API...")

    load_dotenv()

    api_key = os.getenv('API_KEY')
    api_secret = os.getenv('API_SECRET')

    if not api_key:
      error("Missing API_KEY (.env)")
    if not api_secret:
      error("Missing API_SECRET (.env)")

    API = Client(api_key, api_secret)

def get_data(symbol = 'BNBUSDT', length = LENGTH, interval = INTERVAL) -> pd.DataFrame:
  print(f"Fetching data from Binance ({symbol} {length} {interval})...")

  data = API.get_historical_klines(symbol, interval, f'{length} UTC') # list of lists
  data = np.array(data) # convert to numpy 2D array (rows, columns)
  data = data[:, :-1] # skip last 'ignore' column on numpy array

  columns = [
    'open_time', 'open', 'high', 'low', 'close', 'volume',
    'close_time', 'quote_asset_volume', 'number_of_trades',
    'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume'
  ]

  return pd.DataFrame(data, columns=columns)

def get_data_file_path(filename: str) -> Path:
  return DATA_PATH / f'{filename}.csv'

def exists_data(filename: str) -> bool:
  return get_data_file_path(filename).exists()

def save_data(filename: str, data: pd.DataFrame):
  path = get_data_file_path(filename)
  os.makedirs(DATA_PATH, exist_ok=True)
  data.to_csv(path, index=False, header=True)
  print(f"Saved data: {path}")

def load_data(filename: str, columns: list[str] = None) -> pd.DataFrame:
  print("Loading data...")
  path = get_data_file_path(filename)
  data = pd.read_csv(path)
  data = pd.DataFrame(data, columns=columns)
  print(f"Loaded: {path}")
  return data

def plot_data(data: pd.DataFrame):
  TIMEZONE = 'Europe/Madrid'

  data['close_time'] = pd.DatetimeIndex(pd.to_datetime(data['close_time'], unit='ms')).tz_localize('UTC').tz_convert(TIMEZONE)

  plt.figure(figsize=(10,5))
  plt.title("BNB/USDT Training Data")
  plt.xlabel('Date')
  plt.ylabel('Price')
  plt.plot(data['close_time'], data['close'], color='green')
  plt.show()

if __name__ == "__main__":
  if not exists_data(DATA_FILE):
    load_api_client()
    save_data(DATA_FILE, get_data())

  data = load_data(DATA_FILE, ['close_time', 'close'])

  plot_data(data)
