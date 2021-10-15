import numpy as np
import pandas as pd

from binance.client import Client

class Binance:

  INTERVAL_1_MINUTE = Client.KLINE_INTERVAL_1MINUTE
  INTERVAL_5_MINUTE = Client.KLINE_INTERVAL_5MINUTE

  def __init__(self, api_key: str, api_secret: str, symbol: str = 'BTCUSDT'):
    self.client = Client(api_key, api_secret)
    self.symbol = symbol
  
  @property
  def symbol(self) -> str:
    return self._symbol
  
  @symbol.setter
  def symbol(self, symbol: str):
    self._symbol = symbol.replace('/', '')

  def get_historical_data(self, length: str = '7 day', interval: str = INTERVAL_1_MINUTE) -> pd.DataFrame:
    data = self.client.get_historical_klines(self.symbol, interval, f'{length} UTC') # list of lists
    data = np.array(data) # convert to numpy 2D array (rows, columns)
    data = data[:, :-1] # skip last 'ignore' column on numpy array

    columns = [
      'open_time', 'open', 'high', 'low', 'close', 'volume',
      'close_time', 'quote_asset_volume', 'number_of_trades',
      'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume'
    ]

    return pd.DataFrame(data, columns=columns)

  def get_current_price(self, with_avg: bool = True):
    ticker = self.client.get_symbol_ticker(symbol=self.symbol)

    ticker['price'] = float(ticker['price'])

    if with_avg:
      ticker['avg'] = self.client.get_avg_price(symbol=self.symbol)
      ticker['avg']['price'] = float(ticker['avg']['price'])

    return ticker

  # https://api.binance.com/api/v3/exchangeInfo?symbol=BNBUSDT
  def get_exchange_info(self):
    return self.client.get_exchange_info()
