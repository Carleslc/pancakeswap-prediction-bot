import os
import json

import pandas as pd

from dotenv import load_dotenv
from pathlib import Path

from utils import error
from settings import API, SYMBOL, LENGTH, INTERVAL

DATA_FILE = f"{SYMBOL.replace('/', '').lower()}_{LENGTH.replace(' ', '_')}_{INTERVAL}"
DATA_PATH = Path('data')

API_CLIENT: API = None

def load_api_client(symbol: str = SYMBOL) -> API:
  global API_CLIENT

  if not API_CLIENT:
    print(f"Loading {API.__name__} API...")

    load_dotenv()

    api_key = os.getenv('BINANCE_API_KEY')
    api_secret = os.getenv('BINANCE_API_SECRET')

    if not api_key:
      error("Missing BINANCE_API_KEY (.env)")
    if not api_secret:
      error("Missing BINANCE_API_SECRET (.env)")

    API_CLIENT = API(api_key, api_secret, symbol)
  
  API_CLIENT.symbol = symbol
  
  return API_CLIENT

def get_binance_data(symbol = SYMBOL, length = LENGTH, interval = INTERVAL) -> pd.DataFrame:
  load_api_client(symbol)

  print(f"Fetching data from Binance ({API_CLIENT.symbol.upper()} {length} {interval})...")

  return API_CLIENT.get_historical_data(length, interval)

def get_current_price(symbol = SYMBOL, with_avg: bool = True):
  load_api_client(symbol)

  return API_CLIENT.get_current_price(with_avg)

def get_exchange_info():
  load_api_client()

  return API_CLIENT.get_exchange_info()

def get_data_file_path(filename: str, extension: str = 'csv') -> Path:
  return DATA_PATH / f'{filename}.{extension}'

def exists_data(filename: str = DATA_FILE, extension: str = 'csv') -> bool:
  return get_data_file_path(filename, extension).exists()

def _save_data(write, filename: str, extension: str = 'csv'):
  path = get_data_file_path(filename, extension)
  os.makedirs(DATA_PATH, exist_ok=True)
  write(path)
  print(f"Saved data: {path}")

def save_exchange_info(filename: str = 'exchange_info'):
  info = get_exchange_info()

  def write_exchange_info(path: Path):
    with open(path, 'w') as file:
      file.write(json.dumps(info, indent=2))
  
  _save_data(write_exchange_info, filename, 'json')

def save_data(data: pd.DataFrame, filename: str = DATA_FILE):
  def write_data_csv(path: Path):
    data.to_csv(path, index=False, header=True)
  
  _save_data(write_data_csv, filename, 'csv')

def load_data(filename: str = DATA_FILE, columns: list[str] = None, index_column: str = None) -> pd.DataFrame:
  print(f"Loading {filename}...")
  path = get_data_file_path(filename)
  data = pd.read_csv(path)
  if index_column is not None:
    data.set_index(index_column, inplace=True, drop=False, verify_integrity=True)
  data = pd.DataFrame(data, columns=columns)
  print(f"Loaded: {path}")
  return data

if __name__ == "__main__":
  if not exists_data():
    save_data(get_binance_data())

  data = load_data(columns=['close_time', 'close'])

  from visualization import plot_data

  plot_data(data, SYMBOL)
