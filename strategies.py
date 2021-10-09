import numpy as np

import statistics

from settings import BET
from dataset import Dataset
from classifier import Classifier, RandomClassifier

# Bet strategies

def bet_same_always(*_) -> float:
  return BET

def bet_same(payout: float = 2, probability: float = 1, _ = None) -> float:
  return BET if payout > 1/probability else 0

def bet_greedy(payout: float = 2, probability: float = 1, _ = None) -> float:
  if payout <= 1/probability:
    return 0
  return max(BET / 2, min(BET * 2, BET * (1 + (payout - 2)))) # Greedy if payout > 2

def bet_min_prob(payout: float, probability: float, classifier: Classifier, min_probability: float = 0.6) -> float:
    return bet_same(payout) if isinstance(classifier, RandomClassifier) or probability >= min_probability else 0
  
def bet_min_prob_greedy(payout: float, probability: float, classifier: Classifier, min_probability: float = 0.6) -> float:
  return bet_greedy(payout) if isinstance(classifier, RandomClassifier) or probability >= min_probability else 0

# Custom classifiers

class RSIClassifier(Classifier):
  
  OVERBOUGHT = 70
  OVERSOLD = 30

  def __init__(self, dataset: Dataset = None, rsi_column: str = 'rsi'):
    super().__init__(dataset)
    self.rsi_column = rsi_column
  
  def fit(self, _, Y_train):
    self.classes = np.unique(Y_train)
    self.most_common_class = statistics.mode(Y_train)

  def probabilities(self, X) -> np.ndarray:
    rsi = self.get_column(X, self.rsi_column)

    down_probability = rsi / 100
    # down_probability[(rsi < RSIClassifier.OVERBOUGHT) & (rsi > RSIClassifier.OVERSOLD)] = 0.5
    up_probability = 1 - down_probability
    
    return np.array([down_probability, up_probability]).T

  def predict(self, X) -> np.ndarray:
    predicted_probabilities = self.probabilities(X)
    tie_mask = np.all(predicted_probabilities == 0.5, axis=1)
    max_indices = np.argmax(predicted_probabilities, axis=1)
    max_indices[tie_mask] = self.most_common_class
    return self.classes[max_indices]
