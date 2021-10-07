import numpy as np
import pandas as pd

from numba import jit
from ta import add_all_ta_features
from ta.volatility import BollingerBands
from ta.momentum import rsi
from ta.others import daily_return

from dataset import Dataset
from settings import LOOKAHEAD
from visualization import display_entries, preview_dataset, plot_correlation_matrix

from utils import error, datetime_from_ms

# Columns to use for training
FEATURE_COLUMNS = ['high', 'low', 'close', 'volume']

# History to save for each row
RSI_W = 14
BB_W = 20
LOOKBEHIND_ROWS = list(range(2 * LOOKAHEAD, 0, -1))
LOOKBEHIND_ROWS_DIFF = [2 * LOOKAHEAD, LOOKAHEAD]
LOOKBEHIND = max(max(LOOKBEHIND_ROWS), max(LOOKBEHIND_ROWS_DIFF)) + max(RSI_W, BB_W)

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
  add_technical_indicators(dataset)
  add_historical_data(dataset, ['high', 'low', 'close', 'volume', 'rsi', 'bb_p', 'bb_w', 'vwap_diff'], LOOKBEHIND_ROWS_DIFF, diff=True)
  add_historical_data(dataset, ['change'], LOOKBEHIND_ROWS)

  dataset.X = dataset.X[FEATURE_COLUMNS + dataset.new_columns].iloc[LOOKBEHIND:-LOOKAHEAD]
  
  return dataset

def dataset_days(dataset: Dataset) -> int:
  open_time = dataset['open_time'].values
  datetime_diff = datetime_from_ms(open_time[-1]) - datetime_from_ms(open_time[0])
  return datetime_diff.days + 1

def add_time(dataset: Dataset):
  days = []
  weekdays = []
  day_minutes = []

  for time in dataset['close_time']:
    time = datetime_from_ms(time)
    days.append(time.day)
    weekdays.append(time.weekday())
    day_minutes.append(60*time.hour + time.minute)

  total_days = dataset_days(dataset)

  if total_days >= 7:
    dataset['day'] = days
    dataset['weekday'] = weekdays
  
  dataset['day_minute'] = day_minutes

def add_price_diff(dataset: Dataset):
  close_price = dataset['close']
  open_price = dataset['open']
  high = dataset['high']
  low = dataset['low']

  dataset['change'] = daily_return(close_price)
  price_diff = close_price - open_price
  # dataset['price_diff'] = price_diff

  hlc3 = (high + low + close_price) / 3
  # dataset['hlc3'] = (high + low + close_price) / 3
  # dataset['ohlc4'] = (open_price + high + low + close_price) / 4
  dataset['vwap_diff'] = vwap(hlc3.values, dataset['volume'].values) - hlc3

  high_shadow = []
  low_shadow = []

  for t in range(len(dataset)):
    high_shadow.append(high[t] - max(open_price[t], close_price[t]))
    low_shadow.append(min(open_price[t], close_price[t]) - low[t])
  
  height = high - low
  # dataset['height'] = height
  dataset['body_%'] = np.abs(price_diff / height)

  # dataset['high_shadow'] = high_shadow
  dataset['high_shadow_%'] = np.abs(high_shadow / height)
  # dataset['low_shadow'] = low_shadow
  dataset['low_shadow_%'] = np.abs(low_shadow / height)

@jit
def vwap(typical_price: np.ndarray, volume: np.ndarray):
  return np.cumsum(volume * typical_price) / np.cumsum(volume)

def add_technical_indicators(dataset: Dataset, all: bool = False):
  if all:
    before_ta_features = len(dataset.features)
    add_all_ta_features(dataset.X, open='open', close='close', high='high', low='low', volume='volume', fillna=True)
    dataset.new_columns.extend(dataset.features[before_ta_features + 1:])
  else:
    close = dataset['close']
    
    dataset['rsi'] = rsi(close, window=RSI_W)

    # https://technical-analysis-library-in-python.readthedocs.io/en/latest/ta.html?highlight=bollinger#ta.volatility.BollingerBands
    bb = BollingerBands(close, window=BB_W, window_dev=2)
    dataset['bb_m'] = bb.bollinger_mavg()
    # dataset['bb_h'] = bb.bollinger_hband()
    # dataset['bb_l'] = bb.bollinger_lband()
    dataset['bb_w'] = bb.bollinger_wband() # BandWidth
    dataset['bb_p'] = bb.bollinger_pband() # %B

def add_historical_data(dataset: Dataset, columns: list[str], lookbehind: list[int] = LOOKBEHIND_ROWS, diff: bool = False):
  if not len(lookbehind):
    return

  def add_column_lookbehind(column: str, label: str):
    X_lookbehind = []

    values = dataset[column]

    for t in range(len(dataset)):
      t_lookbehind = []

      for r in lookbehind:
        if t - r < 0:
          value = np.nan
        else:
          value = values[t - r]
          if diff:
            value = values[t] / value - 1 if value != 0 else values[t]
        
        t_lookbehind.append(value)
      
      X_lookbehind.append(t_lookbehind)
    
    lookbehind_columns = list(map(lambda i: f'{label}-{i}', lookbehind))

    dataset[lookbehind_columns] = X_lookbehind

  for column in columns:
    add_column_lookbehind(column, column)

def get_dataset(data: pd.DataFrame, preview: bool = True, best_features: bool = True, correlation_matrix: bool = True) -> Dataset:
  print("Preprocessing data...")

  dataset = build_dataset(data)

  print(f"Features: {', '.join(dataset.features)}")

  if preview:
    preview_dataset(data, dataset, plot=False, preview_bars=None)
  
  dataset.train_test_split()

  if best_features:
    dataset.best_features()

  if correlation_matrix and len(dataset.features) <= 30:
    plot_correlation_matrix(dataset)

  display_entries('Total', dataset.Y)

  return dataset
