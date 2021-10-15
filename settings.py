from numpy.random import RandomState

from api import Binance
from utils import truncated_normal_generator

API = Binance

TIMEZONE = 'Europe/Madrid'

SYMBOL = 'BNB/USDT'
LENGTH = '7 day'
INTERVAL = API.INTERVAL_1_MINUTE

# Number of bars ahead to predict (5 minutes + 1 setting bets)
LOOKAHEAD = 6 if INTERVAL == API.INTERVAL_1_MINUTE else 1

# Balance simulation
START_BALANCE = 0.15
MAX_CONSECUTIVE_LOSSES = 10
BET = START_BALANCE / MAX_CONSECUTIVE_LOSSES
PRIZE_FEE = 0.03
TRANSACTION_FEE = 0.001

SEED = 42
RANDOM_STATE = RandomState(SEED)

random_payout = truncated_normal_generator(mean=1.98, sd=0.4, lower=1.1, upper=10, random_state=RANDOM_STATE)
