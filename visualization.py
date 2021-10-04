from typing import Union, TYPE_CHECKING
from numpy.typing import ArrayLike

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from dataset import Dataset
from settings import LOOKAHEAD, TIMEZONE

if TYPE_CHECKING:
  from classifier import Classifier

def ms_to_datetime(data: pd.DataFrame, column: str, timezone = TIMEZONE):
  return pd.DatetimeIndex(pd.to_datetime(data[column], unit='ms')).tz_localize('UTC').tz_convert(timezone)

def plot_data(data: pd.DataFrame, symbol: str):
  data['close_time'] = ms_to_datetime(data, 'close_time')

  plt.figure(figsize=(10,5))
  plt.title(f"{symbol} Training Data")
  plt.xlabel('Date')
  plt.ylabel('Price')
  plt.plot(data['close_time'], data['close'], color='green')
  plt.show()

def get_Y_preview(data: pd.DataFrame, Y: np.ndarray, preview_bars: int = 10):
  preview_bars = min(preview_bars, len(Y) // LOOKAHEAD) if len(Y) >= LOOKAHEAD else 1
  start_index = preview_bars * LOOKAHEAD
  Y_preview = Y[-start_index::LOOKAHEAD]
  Y_dates = ms_to_datetime(data, 'open_time')[-LOOKAHEAD-start_index:-LOOKAHEAD:LOOKAHEAD]

  return Y_preview, Y_dates, start_index

def preview_dataset(data: pd.DataFrame, dataset: Dataset, preview_bars: int = 10):
  Y_preview, Y_dates, start_index = get_Y_preview(data, dataset.Y, preview_bars)
  X_preview = pd.DataFrame(dataset.X.iloc[-start_index::LOOKAHEAD])

  X_preview['open_time'] = Y_dates
  X_preview['Y'] = Y_preview
  
  print('Preview dataset')
  print(X_preview)

  plt.scatter(Y_dates, Y_preview, c=Y_preview, cmap='RdYlGn')
  plt.title("Preview dataset: BULL / BEAR")
  plt.show()

def preview_prediction(data: pd.DataFrame, classifier: 'Classifier', Y_title: str, Y: np.ndarray, Y_pred_title: str, Y_pred: np.ndarray, preview_bars: int = 10):
  Y_preview, Y_dates, start_index = get_Y_preview(data, Y, preview_bars)
  Y_pred_preview = Y_pred[-start_index::LOOKAHEAD]

  for i in range(len(Y_preview)):
    print(f"{Y_dates[i]} -> {'BUY' if Y_pred_preview[i] else 'SELL'} ({'success' if Y_preview[i] == Y_pred_preview[i] else 'fail'})")

  line1 = plt.scatter(Y_dates, Y_preview, label=Y_title)
  line2 = plt.scatter(Y_dates, Y_pred_preview, label=Y_pred_title)
  plt.legend(handles=[line1, line2])
  plt.title(f"Preview {classifier}")
  plt.show()

def plot_correlation_matrix(dataset: Dataset):
  sns.heatmap(dataset.correlation_matrix(), annot=True, cmap="RdYlGn")
  plt.show()

def plot_balances(balances: dict['Classifier', list[float]], start_balance: float):
  plt.title("Performance")
  plt.xlabel('#')
  plt.ylabel('Balance')
  plt.axhline(y=0, color='red', linestyle='-')
  plt.axhline(y=start_balance, color='gray', linestyle='--')

  balance_lines = []
  
  for classifier, balance_history in balances.items():
    classifier_balances, = plt.plot(balance_history, label=classifier.name)
    balance_lines.append(classifier_balances)
  
  plt.legend(handles=balance_lines)
  plt.show()

def plot_barh(df: pd.DataFrame, title: str = '', **kwargs):
  ax = df.plot.barh(**kwargs)
  ax.invert_yaxis()
  plt.title(title)
  plt.show()

def display_entries(label: str, Y: np.ndarray):
  if Y is not None:
    total = len(Y)
    bulls = sum(Y)
    bears = total - bulls
    print(f"Entries ({label}): {total}")
    print(f"BULL: {bulls} ({((bulls / total)*100):.2f}%) | BEAR: {bears} ({((bears / total)*100):.2f}%)")
