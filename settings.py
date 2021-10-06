from api import Binance
from utils import truncated_normal_generator

API = Binance

TIMEZONE = 'Europe/Madrid'

SYMBOL = 'BNB/USDT'
LENGTH = '7 day'
INTERVAL = API.INTERVAL_1_MINUTE

# Number of bars ahead to predict (5 minutes)
LOOKAHEAD = 5 if INTERVAL == API.INTERVAL_1_MINUTE else 1

# Balance simulation
START_BALANCE = 0.15
MAX_CONSECUTIVE_LOSSES = 10
BET = START_BALANCE / MAX_CONSECUTIVE_LOSSES
PRIZE_FEE = 0.03
TRANSACTION_FEE = 0.001

random_payout = truncated_normal_generator(mean=1.98, sd=0.4, lower=1.1, upper=10)

def get_bet_greedy(payout: float = 2, probability: float = 1) -> float:
  if payout <= 1/probability:
    return 0
  return max(BET / 2, min(BET * 2, BET * (1 + (payout - 2)))) # Greedy if payout > 2

def get_bet_same(payout: float = 2, probability: float = 1) -> float:
  return BET if payout > 1/probability else 0

get_bet = get_bet_same
