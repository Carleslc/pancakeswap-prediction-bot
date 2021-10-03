import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from scipy.stats import truncnorm

from data import *

def get_dataset(data: pd.DataFrame, columns: list[str] = None, lookahead: int = 5) -> tuple[pd.DataFrame, np.ndarray]:
  """
  Convert historical data to a dataset with labels
  X: ohlcv relevant data with rows at time t
  Y: 1 if price at t+lookahead is higher (bull bet) or 0 if price at t+lookahead is lower (bear bet)
  """
  if columns is None:
    columns = ['close', 'volume']
  elif 'close' not in columns:
    columns.append('close')
  
  X = data[columns].iloc[:-lookahead]
  Y = []
  
  close_prices = data['close'].values

  for t in range(len(X)):
    Y.append(1 if close_prices[t + lookahead] > close_prices[t] else 0)

  Y = np.array(Y)
  
  return X, Y

def preprocess(data: pd.DataFrame):
  print("Preprocessing data...")

  X, Y = get_dataset(data, columns=['open', 'high', 'low', 'close', 'volume'])

  print(f"Features: {', '.join(X.columns.values)}")

  scaler = StandardScaler() # normalize data
  scaler.fit(X)

  split_at = int(len(X)*0.8) # 80% training, 20% test

  X_train, X_test = scaler.transform(X[:split_at]), scaler.transform(X[split_at:])
  Y_train, Y_test = Y[:split_at], Y[split_at:]

  return X_train, Y_train, X_test, Y_test

def train(classifier, X_train, Y_train, X_test, Y_test):
  print(f"Training model with {type(classifier).__name__}...")

  classifier.fit(X_train, Y_train)

  print("Model trained")

  Y_train_pred = classifier.predict(X_train)
  Y_test_pred = classifier.predict(X_test)

  print(f"Accuracy (Training): {accuracy_score(Y_train, Y_train_pred)}")
  print(f"Accuracy (Test): {accuracy_score(Y_test, Y_test_pred)}")

  return Y_test_pred

def random_prediction(Y_test):
  Y_test_pred_random = np.random.randint(0, 2, size=len(Y_test)) # random values within [0, 2) aka [0,1]

  print(f"Accuracy (Random): {accuracy_score(Y_test, Y_test_pred_random)}")

  return Y_test_pred_random

def truncated_normal_generator(mean = 0, sd = 1, lower = -1, upper = 1):
  generator = truncnorm((lower - mean) / sd, (upper - mean) / sd, loc=mean, scale=sd)
  def get(size = None) -> float:
    return generator.rvs(size)
  return get

def compare_to_random(classifier, Y_test, Y_test_pred, Y_test_pred_random):
  classifier = type(classifier).__name__

  start_balance = 1
  balance = start_balance
  balance_random = start_balance
  bet = 0.05
  transaction_fee = 0.001
  prize_fee = 0.03

  balance_history = [balance]
  balance_random_history = [balance_random]

  randn_payout = truncated_normal_generator(mean=1.97, sd=0.4, lower=1.1, upper=10)

  for i in range(len(Y_test)):
    payout = randn_payout() * (1 - prize_fee) # winner multiplier

    balance -= bet + transaction_fee
    balance_random -= bet + transaction_fee

    # actual classifier
    if Y_test_pred[i] == Y_test[i]:
      balance += payout*bet
    
    # random classifier
    if Y_test_pred_random[i] == Y_test[i]:
      balance_random += payout*bet
    
    balance_history.append(balance)
    balance_random_history.append(balance_random)
  
  print(f"{classifier} final balance: {balance:.3f} ({((balance - start_balance)*100):.2f}%)")
  print(f"Random final balance: {balance_random:.3f} ({((balance_random - start_balance)*100):.2f}%)")
  
  classifier_line, = plt.plot(balance_history, label=classifier)
  random_line, = plt.plot(balance_random_history, label='Random')
  plt.xlabel('#')
  plt.ylabel('Balance')
  plt.axhline(y=0, color='red', linestyle='-')
  plt.axhline(y=start_balance, color='gray', linestyle='--')
  plt.legend(handles=[classifier_line, random_line])
  plt.title(f"{classifier} VS Random")
  plt.show()

if __name__ == "__main__":
  if not exists_data(DATA_FILE):
    load_api_client()
    save_data(DATA_FILE, get_data())

  data = load_data(DATA_FILE)

  X_train, Y_train, X_test, Y_test = preprocess(data)

  Y_test_pred_random = random_prediction(Y_test)

  # Classifiers
  max_depth = 20
  n_estimators = 500
  RANDOM_FOREST = RandomForestClassifier(max_depth=max_depth, n_estimators=n_estimators)
  ADA_BOOST = AdaBoostClassifier(DecisionTreeClassifier(max_depth=max_depth), n_estimators=n_estimators)
  GRADIENT_BOOST = GradientBoostingClassifier(max_depth=max_depth, n_estimators=n_estimators)
  # SVM = SVC()

  for classifier in [RANDOM_FOREST]:
    Y_test_pred = train(classifier, X_train, Y_train, X_test, Y_test)

    compare_to_random(classifier, Y_test, Y_test_pred, Y_test_pred_random)
