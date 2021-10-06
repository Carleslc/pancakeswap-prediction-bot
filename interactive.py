from utils import try_float, is_nan
from settings import get_bet_greedy

DECIMALS = 4
ADJUST_EXPECTED = 0.2
BET_ON_EXPECTED = True

# TODO: Get current data and sync blocks 5m

if __name__ == "__main__":
  print("Input the payout to get the bet amount")

  try:
    while True:
      payout = try_float(input())
      
      if BET_ON_EXPECTED:
        payout = payout + min(max(2 - payout, -ADJUST_EXPECTED), ADJUST_EXPECTED)
        print(f"Expected: {round(payout, 2)}")

      if not is_nan(payout):
        bet = get_bet_greedy(payout)
        print(f'BET {round(bet, DECIMALS)}')
      else:
        print('NaN')
  except KeyboardInterrupt:
    pass
