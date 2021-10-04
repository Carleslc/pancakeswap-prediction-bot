from api import Binance
from utils import truncated_normal_generator

API = Binance

TIMEZONE = 'Europe/Madrid'

SYMBOL = 'BNB/USDT'
LENGTH = '7 day'
INTERVAL = API.INTERVAL_1_MINUTE

# Number of bars ahead to predict (5 minutes)
LOOKAHEAD = 5 if INTERVAL == API.INTERVAL_1_MINUTE else 1

# Number of bars to save for each row
LOOKBEHIND = 20

# Columns to use for training
FEATURE_COLUMNS = ['close', 'volume', 'close_time']

# Balance simulation
START_BALANCE = 1
BET = 0.05
PRIZE_FEE = 0.03
TRANSACTION_FEE = 0.001

random_payout = truncated_normal_generator(mean=1.98, sd=0.4, lower=1.1, upper=10)
