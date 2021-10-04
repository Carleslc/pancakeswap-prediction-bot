import numpy as np
import pandas as pd

from utils import error
from dataset import Dataset
from settings import LOOKBEHIND, LOOKAHEAD, FEATURE_COLUMNS
from visualization import display_entries, preview_dataset, plot_correlation_matrix

def _build_dataset(data: pd.DataFrame, columns: list[str] = None) -> Dataset:
  """
  Convert historical data to a dataset with labels
  X: ohlcv relevant data with rows at time t
  Y: 1 if price at t+LOOKAHEAD is higher (bull bet) or 0 if price at t+LOOKAHEAD is lower (bear bet)
  """
  X_data = data if columns is None else data[columns]

  X = X_data.iloc[LOOKBEHIND:-LOOKAHEAD]
  Y = []

  if X.empty:
    error("Not enough data")
  
  close_prices = data['close'].values

  X_lookbehind = []
  
  for t in range(LOOKBEHIND, len(X) + LOOKBEHIND):
    Y.append(1 if close_prices[t + LOOKAHEAD] > close_prices[t] else 0)
    X_lookbehind.append(close_prices[t - LOOKBEHIND:t])

  lookbehind_columns = list(map(lambda i: f't-{i}', range(LOOKBEHIND, 0, -1)))

  dataset = Dataset(X, Y)
  dataset.add_columns(lookbehind_columns, X_lookbehind, prepend=True)
  
  return dataset

def get_dataset(data: pd.DataFrame, preview: bool = True, best_features: bool = True, correlation_matrix: bool = True) -> Dataset:
  print("Preprocessing data...")

  dataset = _build_dataset(data, columns=FEATURE_COLUMNS)

  print(f"Features: {', '.join(dataset.features)}")

  if preview:
    preview_dataset(data, dataset)

  if best_features:
    dataset.best_features()

  if correlation_matrix:
    plot_correlation_matrix(dataset)

  display_entries('Total', dataset.Y)

  dataset.train_test_split()

  return dataset
