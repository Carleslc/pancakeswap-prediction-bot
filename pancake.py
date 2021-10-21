from typing import Iterable

import pandas as pd

from datetime import datetime

from bsc import BinanceSmartChain, bsc_client, from_wei, same_address
from visualization import pretty
from data import exists_data, save_data, load_data

PREDICTION_DATA_FILE = 'pancake_prediction'

# https://pancakeswap.finance/prediction

PancakePredictionV2 = '0x18b2a687610328590bc8f2e5fedde3b582a49cda'
# PancakePredictionV2 https://bscscan.com/address/0x18b2a687610328590bc8f2e5fedde3b582a49cda#contracts
# currentEpoch
# minBetAmount
# intervalSeconds
# treasuryFee
# paused
# rounds

PancakeOracle_BNB_USD = '0xd276fcf34d54a926773c399ebaa772c12ec394ac'
# Oracle PancakeSwap BNB / USD EACAggregatorProxy https://bscscan.com/address/0xd276fcf34d54a926773c399ebaa772c12ec394ac#contracts
# latestAnswer
# latestTimestamp
# proposedGetRoundData

def get_rounds_data(rounds: Iterable[int], bsc: BinanceSmartChain = None) -> pd.DataFrame:
  rounds_data = []

  if bsc is None or not same_address(bsc.address, PancakePredictionV2):
    bsc = bsc_client(PancakePredictionV2)

  with bsc as pancake_prediction:
    print(f"Fetching data from {pancake_prediction.name}...")

    for epoch in rounds:
      round_data = pancake_prediction.rounds(epoch)
      lockPrice = round_data[4]
      closePrice = round_data[5]
      totalAmount = round_data[8]
      bullAmount = round_data[9]
      bearAmount = round_data[10]
      upPayout = totalAmount / bullAmount if bullAmount != 0 else 1.0
      downPayout = totalAmount / bearAmount if bearAmount != 0 else 1.0
      change = closePrice - lockPrice
      result = 1 if closePrice > lockPrice else (-1 if closePrice < lockPrice else 0)
      round_data.append(upPayout)
      round_data.append(downPayout)
      round_data.append(change)
      round_data.append(result)
      rounds_data.append(round_data)
      print(f'#{epoch}')

  rounds_data = pd.DataFrame(rounds_data, columns=['epoch', 'startTimestamp', 'lockTimestamp', 'closeTimestamp', 'lockPrice', 'closePrice', 'lockOracleId', 'closeOracleId', 'totalAmount', 'bullAmount', 'bearAmount', 'rewardBaseCalAmount', 'rewardAmount', 'oracleCalled', 'upPayout', 'downPayout', 'change', 'result'])
  rounds_data.set_index('epoch', inplace=True, drop=False, verify_integrity=True)

  return rounds_data

def update_rounds_data(current_epoch: int, bsc: BinanceSmartChain = None) -> pd.DataFrame:
  if exists_data(PREDICTION_DATA_FILE):
    saved_rounds_data = load_data(PREDICTION_DATA_FILE, index_column='epoch')

    last_saved_epoch = saved_rounds_data['epoch'].iloc[-1]
    update_from_epoch = max(1, last_saved_epoch - 2)

    if current_epoch >= update_from_epoch:
      new_rounds_data = get_rounds_data(range(update_from_epoch, current_epoch + 1), bsc)
      saved_rounds_data.drop(range(update_from_epoch, last_saved_epoch + 1), inplace=True)
      rounds_data = saved_rounds_data.append(new_rounds_data, verify_integrity=True)
    else:
      rounds_data = saved_rounds_data
  else:
    rounds_data = get_rounds_data(range(1, current_epoch + 1), bsc)
  
  save_data(rounds_data, PREDICTION_DATA_FILE)

  return load_data(PREDICTION_DATA_FILE, index_column='epoch')

if __name__ == "__main__":
  with bsc_client(PancakeOracle_BNB_USD) as pancake_oracle:
    print(f'\nChainlink Oracle ({pancake_oracle.name})', pancake_oracle.address)
    print(f'\n{pancake_oracle.description()}')
    decimals = pancake_oracle.decimals()
    latest_answer = pancake_oracle.latestAnswer()
    latest_datetime = datetime.fromtimestamp(pancake_oracle.latestTimestamp())
    print(f'\nLatest: {from_wei(latest_answer, decimals)} USD ({latest_datetime})')
  
  with bsc_client(PancakePredictionV2) as pancake_prediction:
    print(f'\n{pancake_prediction.name}', pancake_prediction.address)
    print(f'\nBalance: {pancake_prediction.get_bnb_balance()} BNB')
    print('\n' + '\n'.join(pancake_prediction.list_functions()))
    print('\nPaused:', pancake_prediction.paused())
    next_round: int = pancake_prediction.currentEpoch()
    current_round = next_round - 1
    print('\nRunning Round')
    print(pretty(pancake_prediction.call(pancake_prediction.functions.rounds(current_round))))
    print('\nNext Round')
    print(pretty(pancake_prediction.call(pancake_prediction.functions.rounds(next_round))))

    print('\nFetching Rounds...')
    rounds_data = update_rounds_data(next_round, pancake_prediction)
    print()
    print(rounds_data)
