from scipy.stats import truncnorm

def error(message: str):
  print(f'ERROR: {message}')
  exit(1)

def truncated_normal_generator(mean = 0, sd = 1, lower = -1, upper = 1):
  generator = truncnorm((lower - mean) / sd, (upper - mean) / sd, loc=mean, scale=sd)
  def get(size = None) -> float:
    return generator.rvs(size)
  return get
