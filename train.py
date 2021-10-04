import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.model_selection import GridSearchCV, ShuffleSplit
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.metrics import accuracy_score

from scipy.stats import truncnorm

from data import *

# Number of bars ahead to predict (5 minutes)
LOOKAHEAD = 5 if INTERVAL == '1m' else 1

# Number of bars to save for each row
LOOKBEHIND = 20

FEATURE_COLUMNS = ['close', 'volume', 'close_time']

def get_dataset(data: pd.DataFrame, columns: list[str] = None) -> tuple[pd.DataFrame, np.ndarray]:
  """
  Convert historical data to a dataset with labels
  X: ohlcv relevant data with rows at time t
  Y: 1 if price at t+LOOKAHEAD is higher (bull bet) or 0 if price at t+LOOKAHEAD is lower (bear bet)
  """
  X_data = data if columns is None else data[columns]

  X = X_data.iloc[LOOKBEHIND:-LOOKAHEAD]
  Y = []

  if X.empty:
    error("Not enough data")
  
  close_prices = data['close'].values

  X_lookbehind = []
  
  for t in range(LOOKBEHIND, len(X) + LOOKBEHIND):
    Y.append(1 if close_prices[t + LOOKAHEAD] > close_prices[t] else 0)
    X_lookbehind.append(close_prices[t - LOOKBEHIND:t])

  lookbehind_columns = list(map(lambda i: f't-{i}', range(LOOKBEHIND, 0, -1)))
  X_lookbehind = pd.DataFrame(np.array(X_lookbehind), columns=lookbehind_columns, index=X.index)
  
  X = pd.concat([X_lookbehind, X], axis=1)
  Y = np.array(Y)
  
  return X, Y

def get_Y_preview(data: pd.DataFrame, Y: np.ndarray, preview_bars: int = 10):
  preview_bars = min(preview_bars, len(Y) // LOOKAHEAD) if len(Y) >= LOOKAHEAD else 1
  start_index = preview_bars * LOOKAHEAD
  Y_preview = Y[-start_index::LOOKAHEAD]
  Y_dates = ms_to_datetime(data, 'open_time')[-LOOKAHEAD-start_index:-LOOKAHEAD:LOOKAHEAD]

  return Y_preview, Y_dates, start_index

def preview_dataset(data: pd.DataFrame, X: pd.DataFrame, Y: np.ndarray, preview_bars: int = 10):
  Y_preview, Y_dates, start_index = get_Y_preview(data, Y, preview_bars)
  X_preview = pd.DataFrame(X.iloc[-start_index::LOOKAHEAD])

  X_preview['open_time'] = Y_dates
  X_preview['Y'] = Y_preview
  
  print('Preview dataset')
  print(X_preview)

  plt.scatter(Y_dates, Y_preview, c=Y_preview, cmap='RdYlGn')
  plt.title("Preview dataset: BULL / BEAR")
  plt.show()

def display_entries(label: str, Y: np.ndarray):
  total = len(Y)
  bulls = sum(Y)
  bears = total - bulls
  print(f"Entries ({label}): {total}")
  print(f"BULL: {bulls} ({((bulls / total)*100):.2f}%) | BEAR: {bears} ({((bears / total)*100):.2f}%)")

def preprocess(data: pd.DataFrame):
  global FEATURE_COLUMNS
  print("Preprocessing data...")

  X, Y = get_dataset(data, columns=FEATURE_COLUMNS)

  # preview_dataset(data, X, Y)

  top_features = min(10, len(X.columns))
  kbest = SelectKBest(score_func=chi2, k=top_features)
  features_fit = kbest.fit(X, Y)
  feature_scores = pd.concat([pd.DataFrame(X.columns), pd.DataFrame(features_fit.scores_)], axis=1)
  feature_scores.columns = ['Column', 'Score']
  feature_scores = feature_scores.nlargest(top_features, 'Score')
  if not feature_scores.empty:
    print("Most relevant features")
    print(feature_scores)

  if len(X.columns) < 10:
    labeled_data = pd.concat([X, pd.DataFrame(Y, columns=['Y'], index=X.index)], axis=1)
    correlation = labeled_data.corr()
    sns.heatmap(correlation, annot=True, cmap="RdYlGn")
    plt.show()

  display_entries('Total', Y)

  FEATURE_COLUMNS = X.columns.values
  print(f"Features: {', '.join(FEATURE_COLUMNS)}")

  split_at = int(len(X)*0.8) # 80% training, 20% test

  scaler = StandardScaler() # normalize data
  scaler.fit(X)

  X_train, X_test = scaler.transform(X[:split_at]), scaler.transform(X[split_at:])
  Y_train, Y_test = Y[:split_at], Y[split_at:]

  return X_train, Y_train, X_test, Y_test

def train(classifier, X_train, Y_train, X_test, Y_test, search = False):
  print(f"Training model with {type(classifier).__name__}...")

  if search:
    search_params = {
      'n_estimators': [100, 500],
      'min_weight_fraction_leaf': [0, 0.05, 0.2],
      'max_depth': [1, 2, 20, None]
    }

    # cv = [(slice(None), slice(None))] # no fold
    cv = ShuffleSplit(test_size=0.2, n_splits=1) # 1 fold
    classifier = GridSearchCV(classifier, search_params, cv=cv, n_jobs=-1, verbose=3)

  classifier.fit(X_train, Y_train)

  if search:
    print(classifier.best_params_)
    # print(classifier.best_estimator_)

  print("Model trained")

  if hasattr(classifier, 'feature_importances_'):
    feature_importances = pd.Series(classifier.feature_importances_, index=FEATURE_COLUMNS).sort_values(ascending=False)
    print(feature_importances)
    # feature_importances.plot(kind='barh')
    # plt.title("Feature importances")
    # plt.show()

  Y_train_pred = classifier.predict(X_train)

  accuracy('Training', Y_train, Y_train_pred)

  Y_test_pred = classifier.predict(X_test)

  accuracy('Test', Y_test, Y_test_pred)
  
  display_entries('Test', Y_test_pred)

  return Y_test_pred

def accuracy(title, Y, Y_pred) -> float:
  score = accuracy_score(Y, Y_pred)
  print(f"Accuracy ({title}): {score:.3f}")
  return score

def preview_prediction(data, Y_title, Y, Y_pred_title, Y_pred, preview_bars: int = 10):
  Y_preview, Y_dates, start_index = get_Y_preview(data, Y, preview_bars)
  Y_pred_preview = Y_pred[-start_index::LOOKAHEAD]

  for i in range(len(Y_preview)):
    print(f"{Y_dates[i]} -> {'BUY' if Y_pred_preview[i] else 'SELL'} ({'success' if Y_preview[i] == Y_pred_preview[i] else 'fail'})")

  line1 = plt.scatter(Y_dates, Y_preview, label=Y_title)
  line2 = plt.scatter(Y_dates, Y_pred_preview, label=Y_pred_title)
  plt.legend(handles=[line1, line2])
  plt.title(f"{Y_title} VS {Y_pred_title}")
  plt.show()

def random_prediction(Y_test):
  Y_test_pred_random = np.random.randint(0, 2, size=len(Y_test)) # random values within [0, 2) aka [0,1]

  accuracy('Random', Y_test, Y_test_pred_random)

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
  RANDOM_FOREST = RandomForestClassifier(n_estimators=500)
  EXTRA_RANDOM_FOREST = ExtraTreesClassifier(n_estimators=500)
  GRADIENT_BOOST = GradientBoostingClassifier(n_estimators=100, max_depth=20)
  classifiers = [('Random Forest', RANDOM_FOREST), ('Gradient Boosting', GRADIENT_BOOST), ('Extra', EXTRA_RANDOM_FOREST)]
  MIXED = VotingClassifier(classifiers, voting='hard', n_jobs=-1, verbose=True)

  for classifier in [MIXED]:
    Y_test_pred = train(classifier, X_train, Y_train, X_test, Y_test)

    # preview_prediction(data, 'Y_test', Y_test, 'Y_test_pred', Y_test_pred)

    compare_to_random(classifier, Y_test, Y_test_pred, Y_test_pred_random)
