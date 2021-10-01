import os
import csv

import pandas as pd
import matplotlib.pyplot as plt

from dotenv import load_dotenv
from binance.client import Client

DATA_PATH = './data/'

API = None # Client

def error(message: str):
  print(f'ERROR: {message}')
  exit(1)

def load_api_client() -> Client:
  global API

  if not API:
    load_dotenv()

    api_key = os.getenv('API_KEY')
    api_secret = os.getenv('API_SECRET')

    if not api_key:
      error("Missing API_KEY (.env)")
    if not api_secret:
      error("Missing API_SECRET (.env)")

    API = Client(api_key, api_secret)

def get_data(symbol = 'BNBUSDT', length = '7 day', interval = Client.KLINE_INTERVAL_1MINUTE):
  print(f"Fetching data from Binance ({symbol} {length} {interval})...")
  return API.get_historical_klines(symbol, interval, f'{length} UTC')

def ohlcv_csv_write(f, data):
  csv_writer = csv.writer(f)
  csv_writer.writerow([
    'open_time', 'open', 'high', 'low', 'close', 'volume',
    'close_time', 'quote_asset_volume', 'number_of_trades',
    'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume',
    'ignore'
  ])
  csv_writer.writerows(data)

def save_data(path, data, writer = ohlcv_csv_write):
  os.makedirs(DATA_PATH, exist_ok=True)

  with open(path, 'w') as f:
    writer(f, data)

  print(f"Saved data: {path}")

def load_data(path, columns = None):
  data = pd.read_csv(path)
  df = pd.DataFrame(data, columns=columns)
  print(f"Loaded: {path}")
  return df

if __name__ == "__main__":
  data_file = DATA_PATH + 'bnbusdt_training.csv'

  if not os.path.exists(data_file):
    load_api_client()
    save_data(data_file, get_data())

  df = load_data(data_file, ['close_time', 'close'])

  TIMEZONE = 'Europe/Madrid'

  df['close_time'] = pd.DatetimeIndex(pd.to_datetime(df['close_time'], unit='ms')).tz_localize('UTC').tz_convert(TIMEZONE)

  plt.title("BNBUSDT Training Data")
  plt.plot(df['close_time'], df['close'], color='green')
  plt.show()
