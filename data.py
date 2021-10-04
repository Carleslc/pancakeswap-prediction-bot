import os

import pandas as pd

from dotenv import load_dotenv
from pathlib import Path

from utils import error
from settings import API, SYMBOL, LENGTH, INTERVAL

DATA_FILE = f"{SYMBOL.replace('/', '').lower()}_{LENGTH.replace(' ', '_')}_{INTERVAL}"
DATA_PATH = Path('data')

API_CLIENT: API = None

def load_api_client() -> API:
  global API_CLIENT

  if not API_CLIENT:
    print(f"Loading {type(API).__name__} API...")

    load_dotenv()

    api_key = os.getenv('API_KEY')
    api_secret = os.getenv('API_SECRET')

    if not api_key:
      error("Missing API_KEY (.env)")
    if not api_secret:
      error("Missing API_SECRET (.env)")

    API_CLIENT = API(api_key, api_secret)
  
  return API_CLIENT

def get_data(symbol = SYMBOL, length = LENGTH, interval = INTERVAL) -> pd.DataFrame:
  load_api_client()

  print(f"Fetching data from Binance ({symbol} {length} {interval})...")

  return API_CLIENT.get_historical_data(symbol, length, interval)

def get_data_file_path(filename: str) -> Path:
  return DATA_PATH / f'{filename}.csv'

def exists_data(filename: str = DATA_FILE) -> bool:
  return get_data_file_path(filename).exists()

def save_data(data: pd.DataFrame, filename: str = DATA_FILE):
  path = get_data_file_path(filename)
  os.makedirs(DATA_PATH, exist_ok=True)
  data.to_csv(path, index=False, header=True)
  print(f"Saved data: {path}")

def load_data(filename: str = DATA_FILE, columns: list[str] = None) -> pd.DataFrame:
  print("Loading data...")
  path = get_data_file_path(filename)
  data = pd.read_csv(path)
  data = pd.DataFrame(data, columns=columns)
  print(f"Loaded: {path}")
  return data

if __name__ == "__main__":
  if not exists_data():
    save_data(get_data())

  data = load_data(DATA_FILE, ['close_time', 'close'])

  from visualization import plot_data

  plot_data(data, SYMBOL)
