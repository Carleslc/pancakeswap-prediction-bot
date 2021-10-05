import math

from scipy.stats import truncnorm

def error(message: str):
  print(f'ERROR: {message}')
  exit(1)

def truncated_normal_generator(mean = 0, sd = 1, lower = -1, upper = 1):
  generator = truncnorm((lower - mean) / sd, (upper - mean) / sd, loc=mean, scale=sd)
  def get(size = None) -> float:
    return generator.rvs(size)
  return get

def try_float(f: str) -> float:
  try:
    if ',' in f:
      f = f.replace(',', '.')
    return float(f)
  except ValueError:
    return math.nan

def is_nan(f: float) -> bool:
  return math.isnan(f)
