from typing import Union, Callable

import numpy as np

from sklearn.base import ClassifierMixin
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.svm import SVC

from data import exists_data, get_data, save_data, load_data
from classifier import Dataset, Classifier, RandomClassifier, SklearnClassifier, GridSearchClassifier, StackClassifier, VoteClassifier, wrap_classifiers
from visualization import display_entries, preview_prediction, plot_balances
from strategies import RSIClassifier, bet_same, bet_same_always, bet_greedy, bet_min_prob, bet_min_prob_greedy
from settings import RANDOM_STATE, START_BALANCE, TRANSACTION_FEE, PRIZE_FEE, random_payout

import preprocess

def search(classifier: Union[ClassifierMixin, SklearnClassifier], params: dict[str, list[float]]) -> Classifier:
  if isinstance(classifier, (ClassifierMixin, SklearnClassifier)):
    search_params = params or {
      'n_estimators': [100, 500],
      'min_weight_fraction_leaf': [0, 0.05, 0.2],
      'max_depth': [2, 5, 10, 20, None]
    }
    return GridSearchClassifier(classifier, search_params, cv=1, verbose=True)
  return classifier

def compare_classifiers(dataset: Dataset, classifiers: list[Classifier], bet_strategy: Callable[[float, float, Classifier], float] = bet_same_always, payouts: list[float] = None, verbose: bool = False):
  title = f"Classifiers Performance ({bet_strategy.__name__})"

  print(f'\n{title}')

  balances = dict(map(lambda classifier: (classifier, [START_BALANCE]), classifiers))

  for classifier in classifiers:
    classifier.Y_prob = classifier.probabilities(dataset.X_test)
    classifier.Y_success_probs = []
    classifier.Y_fail_probs = []
    classifier.bets = 0
    classifier.win_bets = 0

  Y_test = dataset.Y_test
  X_test = dataset.X_test

  # for each round
  for r in range(len(Y_test)):
    payout = payouts[r] if payouts is not None else random_payout() # winner multiplier

    if verbose:
      print(f"#{r + 1} x{payout:.2f}")
      print(X_test.iloc[r])

    for classifier in classifiers:
      balance_history = balances[classifier]
      balance = balance_history[-1]

      prediction = classifier.Y_pred[r]
      probability = classifier.Y_prob[r][prediction]

      bet = bet_strategy(payout, probability, classifier)

      win = prediction == Y_test[r]

      if bet > 0:
        classifier.bets += 1

        balance -= bet + TRANSACTION_FEE

        if win:
          classifier.win_bets += 1
          prize = bet * (payout - 1)
          balance += bet + (prize * (1 - PRIZE_FEE))
        
        if verbose:
          print(f"{'UP' if prediction else 'DOWN'} {probability:.3f} {'SUCCESS' if win else 'FAIL'} ({classifier})")
      
        if win:
          classifier.Y_success_probs.append(probability)
        else:
          classifier.Y_fail_probs.append(probability)

      balance_history.append(balance)
  
  # final results
  for classifier in classifiers:
    final_balance = balances[classifier][-1]
    score = classifier.win_bets/classifier.bets if classifier.bets > 0 else 0.5
    print(f"\n{classifier} score: {score:.3f}")
    print(f"{classifier} final balance: {final_balance:.3f} ({((final_balance/START_BALANCE - 1)*100):.2f}%)")
    if len(classifier.Y_success_probs):
      print(f"{classifier} success mean probability: {np.mean(classifier.Y_success_probs):.3f}")
    if len(classifier.Y_fail_probs):
      print(f"{classifier} fail mean probability: {np.mean(classifier.Y_fail_probs):.3f}")

  plot_balances(title, balances, START_BALANCE)

def compare_classifiers_bet_strategies(dataset: Dataset, classifiers: list[Classifier], bet_strategies: list[Callable[[float, float, Classifier], float]], payouts: list[float] = None, verbose: bool = False):
  for bet_strategy in bet_strategies:
    compare_classifiers(dataset, classifiers, bet_strategy, payouts=payouts, verbose=verbose)

def train_log(classifier: Classifier):
  print(f"\nTraining {classifier.name}...")
  classifier.train()
  print(f"{classifier.name} trained")

if __name__ == "__main__":
  # Load data
  if not exists_data():
    save_data(get_data())

  data = load_data()

  PREVIEW_DATA = True
  PREVIEW_DATA_PLOT = False
  KBEST_FEATURES = True
  CORRELATION_MATRIX = False
  FEATURE_IMPORTANCES = False
  PREVIEW_PREDICTIONS = False

  BET_STRATEGIES = [bet_same_always, bet_min_prob] # bet_same, bet_min_prob, bet_greedy, bet_min_prob_greedy

  # Prepare data for training
  dataset = preprocess.get_dataset(data, preview=PREVIEW_DATA, preview_plot=PREVIEW_DATA_PLOT, best_features=KBEST_FEATURES, correlation_matrix=CORRELATION_MATRIX)

  # Classifiers
  RANDOM = RandomClassifier(dataset, random_state=RANDOM_STATE)
  RANDOM_FOREST = RandomForestClassifier(n_estimators=500, criterion='entropy', random_state=RANDOM_STATE)
  EXTRA_RANDOM_FOREST = ExtraTreesClassifier(n_estimators=500, random_state=RANDOM_STATE)
  GRADIENT_BOOST = GradientBoostingClassifier(n_estimators=500, n_iter_no_change=50, random_state=RANDOM_STATE)
  SVM = SVC(probability=True, random_state=RANDOM_STATE)
  RSI = RSIClassifier(dataset)
  # TODO: XGBoost

  mix_classifiers = wrap_classifiers(dataset, [EXTRA_RANDOM_FOREST, GRADIENT_BOOST])
  STACK = StackClassifier(dataset, mix_classifiers, passthrough=True, verbose=True) # *

  # Training
  RANDOM.train()
  
  for classifier in [RSI, STACK]:
    train_log(classifier)
  
  VOTE = VoteClassifier(dataset, [*STACK.estimators_, RSI], 'VotingClassifier', voting='soft', weights=[1, 0.75, 0.25], prefit=True, verbose=True) # *
  VOTE_STACKING = VoteClassifier(dataset, [STACK, VOTE, RSI], 'VoteStacking', prefit=True)
  STACKING_VOTE = StackClassifier(dataset, [STACK, VOTE], 'StackingVote', passthrough=True, verbose=True) # *

  train_log(STACKING_VOTE)

  # Testing
  classifiers: list[Classifier] = [RANDOM, *VOTE.estimators_, STACK, VOTE, VOTE_STACKING, STACKING_VOTE]

  for classifier in classifiers:
    print(f'\n{classifier}')
    
    Y_test_pred = classifier.test()

    display_entries(classifier.name, Y_test_pred)

    if FEATURE_IMPORTANCES:
      classifier.feature_importances()

    if PREVIEW_PREDICTIONS:
      preview_prediction(data, classifier, 'Y_test', classifier.Y_test, 'Y_pred', classifier.Y_pred)

  # Balance Simulation

  payouts = random_payout(len(dataset.Y_test))

  compare_classifiers_bet_strategies(dataset, classifiers, BET_STRATEGIES, payouts, verbose=False)
