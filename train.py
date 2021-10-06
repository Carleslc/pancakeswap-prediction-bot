from typing import Union, Callable

import numpy as np

from sklearn.base import ClassifierMixin
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.svm import SVC

from data import exists_data, get_data, save_data, load_data
from classifier import Dataset, Classifier, RandomClassifier, SklearnClassifier, GridSearchClassifier, MixedClassifier
from visualization import display_entries, preview_prediction, plot_balances
from settings import START_BALANCE, TRANSACTION_FEE, PRIZE_FEE, random_payout, bet_same, bet_same_always, bet_greedy

import preprocess

def search(classifier: Union[ClassifierMixin, SklearnClassifier]) -> Classifier:
  if isinstance(classifier, (ClassifierMixin, SklearnClassifier)):
    search_params = {
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

  # for each round
  for r in range(len(Y_test)):
    payout = payouts[r] if payouts is not None else random_payout() # winner multiplier

    if verbose:
      print(f"#{r + 1} x{payout:.2f}")

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
    print(f"{classifier} success mean probability: {np.mean(classifier.Y_success_probs):.3f}")
    print(f"{classifier} fail mean probability: {np.mean(classifier.Y_fail_probs):.3f}")

  plot_balances(title, balances, START_BALANCE)

if __name__ == "__main__":
  # Load data
  if not exists_data():
    save_data(get_data())

  data = load_data()

  # Prepare data for training
  dataset = preprocess.get_dataset(data, preview=True, correlation_matrix=False)

  # Classifiers
  RANDOM = RandomClassifier()
  RANDOM_FOREST = RandomForestClassifier(n_estimators=500, criterion='entropy')
  EXTRA_RANDOM_FOREST = ExtraTreesClassifier(n_estimators=500)
  GRADIENT_BOOST = GradientBoostingClassifier(n_estimators=500, n_iter_no_change=10)
  SVM = SVC(probability=True)
  # TODO: XGBoost

  MIXED = MixedClassifier([
      RANDOM_FOREST,
      EXTRA_RANDOM_FOREST,
      GRADIENT_BOOST,
      SVM
    ], 'VotingClassifier', voting='soft', weights=[1, 2, 0.75, 0.75], verbose=True)

  # Training
  for classifier in [RANDOM, MIXED]:
    classifier.train(dataset)
  
  classifiers: list[Classifier] = [RANDOM, *MIXED.classifiers, MIXED]

  for classifier in classifiers:
    print(f'\n{classifier}')
    
    Y_test_pred = classifier.test(dataset)

    display_entries(classifier.name, Y_test_pred)

    classifier.feature_importances()

    # if classifier is not RANDOM:
    #   preview_prediction(data, classifier, 'Y_test', classifier.dataset.Y_test, 'Y_pred', classifier.Y_pred)

  payouts = random_payout(len(dataset.Y_test))

  compare_classifiers(dataset, classifiers, payouts=payouts, verbose=False)

  def bet_min_prob(payout: float, probability: float, classifier: Classifier, min_probability: float = 0.6) -> float:
    return bet_same(payout) if isinstance(classifier, RandomClassifier) or probability >= min_probability else 0
  
  compare_classifiers(dataset, classifiers, bet_min_prob, payouts)

  def bet_min_prob_greedy(payout: float, probability: float, classifier: Classifier, min_probability: float = 0.6) -> float:
    return bet_greedy(payout) if isinstance(classifier, RandomClassifier) or probability >= min_probability else 0

  compare_classifiers(dataset, classifiers, bet_min_prob_greedy, payouts)

  compare_classifiers(dataset, classifiers, bet_same, payouts)
