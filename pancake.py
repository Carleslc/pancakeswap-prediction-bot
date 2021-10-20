from datetime import datetime

from bsc import bsc_client, from_wei
from visualization import pretty

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

if __name__ == "__main__":
  with bsc_client(PancakePredictionV2) as pancake_prediction:
    print(f'\n{pancake_prediction.name}', pancake_prediction.address)
    print(f'\nProxy? {pancake_prediction.is_proxy}')
    print(f'\nBalance: {pancake_prediction.get_bnb_balance()} BNB')
    print('\n' + '\n'.join(pancake_prediction.list_functions()))
    print('\nPaused:', pancake_prediction.paused())
    next_round = pancake_prediction.currentEpoch()
    current_round = next_round - 1
    print('\nRunning Round')
    print(pretty(pancake_prediction.call(pancake_prediction.functions.rounds(current_round))))
    print('\nNext Round')
    print(pretty(pancake_prediction.call(pancake_prediction.functions.rounds(next_round))))
  
  with bsc_client(PancakeOracle_BNB_USD) as pancake_oracle:
    print(f'\n{pancake_oracle.name}', pancake_oracle.address)
    print(f'\nProxy? {pancake_oracle.is_proxy}')
    balance = pancake_oracle.get_bnb_balance()
    if balance.is_zero():
      balance = 0
    print(f'\nBalance: {balance} BNB')
    print('\n' + '\n'.join(pancake_oracle.list_functions()))
    print(f'\n{pancake_oracle.description()}')
    decimals = pancake_oracle.decimals()
    latest_answer = pancake_oracle.latestAnswer()
    latest_datetime = datetime.fromtimestamp(pancake_oracle.latestTimestamp())
    print(f'\nLatest: {from_wei(latest_answer, decimals)} BNB ({latest_datetime})')
