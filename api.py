import numpy as np
import pandas as pd

from binance.client import Client

class Binance:

  INTERVAL_1_MINUTE = Client.KLINE_INTERVAL_1MINUTE
  INTERVAL_5_MINUTE = Client.KLINE_INTERVAL_5MINUTE

  def __init__(self, api_key: str, api_secret: str):
    self.client = Client(api_key, api_secret)

  def get_historical_data(self, symbol: str, length: str = '7 day', interval: str = INTERVAL_1_MINUTE) -> pd.DataFrame:
    symbol = symbol.replace('/', '')

    data = self.client.get_historical_klines(symbol, interval, f'{length} UTC') # list of lists
    data = np.array(data) # convert to numpy 2D array (rows, columns)
    data = data[:, :-1] # skip last 'ignore' column on numpy array

    columns = [
      'open_time', 'open', 'high', 'low', 'close', 'volume',
      'close_time', 'quote_asset_volume', 'number_of_trades',
      'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume'
    ]

    return pd.DataFrame(data, columns=columns)
