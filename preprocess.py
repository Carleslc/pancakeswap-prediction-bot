import numpy as np
import pandas as pd

from utils import error

from dataset import Dataset
from settings import LOOKAHEAD
from visualization import display_entries, preview_dataset, plot_correlation_matrix

from datetime import datetime

# Columns to use for training
FEATURE_COLUMNS = ['close', 'volume']

# History to save for each row
LOOKBEHIND_ROWS = [30, 5, 1]
LOOKBEHIND = max(LOOKBEHIND_ROWS, default=0)

def build_dataset(data: pd.DataFrame) -> Dataset:
  """
  Convert historical data to a dataset with labels
  X: ohlcv relevant data with rows at time t
  Y: 1 if price at t+LOOKAHEAD is higher (bull bet) or 0 if price at t+LOOKAHEAD is lower (bear bet)
  """
  if len(data) < LOOKBEHIND + LOOKAHEAD:
    error("Not enough data")
  
  Y = []

  close_prices = data['close']

  for t in range(LOOKBEHIND, len(data) - LOOKAHEAD):
    Y.append(1 if close_prices[t + LOOKAHEAD] > close_prices[t] else 0)

  dataset = Dataset(data, Y)

  add_time(dataset)
  add_price_diff(dataset)
  add_historical_data(dataset)

  dataset.X = dataset.X[FEATURE_COLUMNS + dataset.new_columns].iloc[LOOKBEHIND:-LOOKAHEAD]
  
  return dataset

def add_time(dataset: Dataset):
  days = []
  weekdays = []
  day_minutes = []

  for time in dataset['close_time']:
    time = datetime.fromtimestamp(time/1000)
    days.append(time.day)
    weekdays.append(time.weekday())
    day_minutes.append(60*time.hour + time.minute)

  dataset.add_column('day', days)
  dataset.add_column('weekday', weekdays)
  dataset.add_column('day_minute', day_minutes)

def add_price_diff(dataset: Dataset):
  close_price = dataset['close']
  open_price = dataset['open']
  high = dataset['high']
  low = dataset['low']

  price_diff = close_price - open_price
  price_range = high - low

  high_shadow = []
  low_shadow = []

  for t in range(len(dataset)):
    high_shadow.append(high[t] - max(open_price[t], close_price[t]))
    low_shadow.append(min(open_price[t], close_price[t]) - low[t])

  dataset.add_column('price_diff', price_diff)
  dataset.add_column('price_range', price_range)
  dataset.add_column('high_shadow', high_shadow)
  dataset.add_column('low_shadow', low_shadow)

def add_historical_data(dataset: Dataset, columns: list[str] = ['close', 'volume', 'price_diff', 'price_range']):
  if not LOOKBEHIND:
    return

  def add_column_lookbehind(column: str, label: str):
    X_lookbehind = []

    values = dataset[column]

    for t in range(len(dataset)):
      t_lookbehind = []

      for r in LOOKBEHIND_ROWS:
        t_lookbehind.append(values[t - r] if t >= r else np.nan)
      
      X_lookbehind.append(t_lookbehind)
    
    lookbehind_columns = list(map(lambda i: f'{label}-{i}', LOOKBEHIND_ROWS))

    dataset.add_columns(lookbehind_columns, X_lookbehind, prepend=True)

  for column in columns:
    add_column_lookbehind(column, column)

def get_dataset(data: pd.DataFrame, preview: bool = True, best_features: bool = True, correlation_matrix: bool = True) -> Dataset:
  print("Preprocessing data...")

  dataset = build_dataset(data)

  print(f"Features: {', '.join(dataset.features)}")

  if preview:
    preview_dataset(data, dataset, plot=False, preview_bars=None)

  if best_features:
    dataset.best_features()

  if correlation_matrix:
    plot_correlation_matrix(dataset)

  display_entries('Total', dataset.Y)

  dataset.train_test_split(normalize=True)

  return dataset
