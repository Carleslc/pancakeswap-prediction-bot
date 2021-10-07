import math

from datetime import datetime
from scipy.stats import truncnorm

def error(message: str):
  print(f'ERROR: {message}')
  exit(1)

def truncated_normal_generator(mean = 0, sd = 1, lower = -1, upper = 1, random_state = None):
  generator = truncnorm((lower - mean) / sd, (upper - mean) / sd, loc=mean, scale=sd)
  def get(size = None) -> float:
    return generator.rvs(size, random_state=random_state)
  return get

def datetime_from_ms(ms: int) -> datetime:
  return datetime.fromtimestamp(ms/1000)

def try_float(f: str) -> float:
  try:
    if ',' in f:
      f = f.replace(',', '.')
    return float(f)
  except ValueError:
    return math.nan

def is_nan(f: float) -> bool:
  return math.isnan(f)
