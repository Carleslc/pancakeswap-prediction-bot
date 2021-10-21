import statistics

import numpy as np
import pandas as pd

from math import ceil
from settings import SYMBOL, TRANSACTION_FEE
from data import get_binance_data, save_data, load_data, get_current_price, DATA_FILE
from preprocess import prepare
from strategies import bet_min_prob_greedy as get_bet
from visualization import ms_to_datetime
from utils import try_float, is_nan, error
from classifier import Classifier

DECIMALS = 4
ADJUST_EXPECTED = 0.2
BET_ON_EXPECTED = False
MIN_PROBABILITY = 0.6

PREVIEW_COLUMNS = ['close_time', 'open', 'close', 'high', 'low', 'change', 'volume', 'body_%', 'high_shadow_%', 'low_shadow_%', 'rsi', 'bb_m', 'bb_p', 'ma_diff']

DATA_FILE_LAST = f"{SYMBOL.replace('/', '').lower()}_last"

# TODO: Sync blocks 5m
# TODO: Training with rounds info
# TODO: Auto bet

def show_current_price():
  current = get_current_price()
  print(f"\n{current['symbol']} {round(current['price'], DECIMALS)} | Average {round(current['avg']['price'], DECIMALS)} ({current['avg']['mins']}m)\n")

def get_last_data(preview: bool = True) -> pd.DataFrame:
  save_data(get_binance_data(length='1 day'), DATA_FILE_LAST)

  data = load_data(DATA_FILE_LAST)
  dataset = prepare(data)

  if preview:
    data['close_time'] = ms_to_datetime(data, 'close_time')
    print(data[PREVIEW_COLUMNS].tail())

  return dataset.X_test.tail(1)

def load_model(name: str) -> Classifier:
  return Classifier.load(name, DATA_FILE)

def predict(model: Classifier, data: pd.DataFrame) -> int:
  prediction = model.predict(data)[0]
  probability = model.probabilities(data)[0][prediction]
  return prediction, probability

def pretty_prediction(prediction: int) -> str:
  return 'UP' if prediction else 'DOWN'

def load_models(models: list[str]) -> list[Classifier]:
  loaded_models = []

  for model in models:
    loaded_model = load_model(model)

    if loaded_model is None:
      error(f'{model} not found')
    else:
      if hasattr(loaded_model, '_score'):
        print(f'{model} Score: {loaded_model._score:.4f}')
      loaded_models.append(loaded_model)
  
  return loaded_models

if __name__ == "__main__":
  models = load_models(['RSI', 'MA', 'Stack', 'Agg'])

  print()

  try:
    while True:
      data = get_last_data()

      show_current_price()

      predictions = []

      for model in models:
        prediction, probability = predict(model, data)
        predictions.append((prediction, probability))
        print(f'{pretty_prediction(prediction)} {(probability*100):.2f}\t{model}')

      unwrap_predictions = [prediction for prediction, _ in predictions]
      consensus_prediction = statistics.mode(unwrap_predictions)
      repetitions = np.bincount(unwrap_predictions)

      do_bet = repetitions[consensus_prediction] > ceil(len(predictions) / 2)

      if do_bet:
        consensus_probabilities = [probability for prediction, probability in predictions if prediction == consensus_prediction]
        prediction = pretty_prediction(consensus_prediction)
        probability = statistics.mean(consensus_probabilities)

        print(f'Mean {prediction} probability: {(probability*100):.2f}%')

        do_bet = probability > MIN_PROBABILITY

        if do_bet:
          payout = try_float(input(f"\n{prediction} payout: "))

          if not is_nan(payout):
            if BET_ON_EXPECTED:
              payout = payout + min(max(2 - payout, -ADJUST_EXPECTED), ADJUST_EXPECTED)
              print(f"Expected: {round(payout, 2)}")
            
            bet = get_bet(payout, probability, min_probability=MIN_PROBABILITY)

            do_bet = bet > 2*TRANSACTION_FEE

            if do_bet:
              print(f'BET {round(bet, DECIMALS)} {prediction}')
      
      if not do_bet:
        print('DO NOT BET')
      
      print()
      input("Next round? ")

      print()
  except KeyboardInterrupt:
    pass
