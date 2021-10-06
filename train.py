from typing import Union

import numpy as np

from sklearn.base import ClassifierMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.svm import SVC

from data import exists_data, get_data, save_data, load_data
from classifier import Dataset, Classifier, RandomClassifier, SklearnClassifier, GridSearchClassifier
from visualization import display_entries, preview_prediction, plot_balances
from settings import START_BALANCE, TRANSACTION_FEE, PRIZE_FEE, random_payout, get_bet

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

def compare_classifiers(dataset: Dataset, classifiers: list[Classifier], min_probability: float = 0.6, debug: bool = False):
  print()

  def debug(s: str):
    if debug:
      print(s)
  
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
    payout = random_payout() # winner multiplier

    debug(f"#{r} x{payout:.2f}")

    for classifier in classifiers:
      balance_history = balances[classifier]
      balance = balance_history[-1]

      prediction = classifier.Y_pred[r]
      probability = classifier.Y_prob[r][prediction]

      bet = get_bet(payout) if isinstance(classifier, RandomClassifier) or probability >= min_probability else 0

      win = prediction == Y_test[r]

      if bet > 0:
        classifier.bets += 1

        balance -= bet + TRANSACTION_FEE

        if win:
          classifier.win_bets += 1
          prize = bet * (payout - 1)
          balance += bet + (prize * (1 - PRIZE_FEE))
        
        debug(f"{'UP' if prediction else 'DOWN'} {probability:.3f} {'SUCCESS' if win else 'FAIL'} ({classifier})")
      
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

  plot_balances(balances, START_BALANCE)

def prefitted(voting_classifier: VotingClassifier, dataset: Dataset) -> VotingClassifier:
  voting_classifier.estimators_ = list(map(lambda prefitted_classifier: prefitted_classifier[1], voting_classifier.estimators))
  voting_classifier.le_ = LabelEncoder().fit(dataset.Y)
  voting_classifier.classes_ = voting_classifier.le_.classes_
  return voting_classifier

if __name__ == "__main__":
  # Load data
  if not exists_data():
    save_data(get_data())

  data = load_data()

  # Prepare data for training
  dataset = preprocess.get_dataset(data, preview=True, correlation_matrix=False)

  # Classifiers
  RANDOM = RandomClassifier()
  SVM = SVC(probability=True)
  RANDOM_FOREST = RandomForestClassifier(n_estimators=500, criterion='entropy')
  EXTRA_RANDOM_FOREST = ExtraTreesClassifier(n_estimators=500)
  GRADIENT_BOOST = GradientBoostingClassifier(n_estimators=100, max_depth=10)
  # TODO: XGBoost

  mix_classifiers = [RANDOM_FOREST, GRADIENT_BOOST, EXTRA_RANDOM_FOREST, SVM]
  mix_classifiers_tuples = list(map(lambda classifier: (type(classifier).__name__, classifier), mix_classifiers))
  MIXED = VotingClassifier(mix_classifiers_tuples, voting='soft', n_jobs=-1, verbose=True, weights=[1, 0.75, 1.5, 0.75])

  classifiers: list[Classifier] = [RANDOM, *map(SklearnClassifier, mix_classifiers)]

  # Training
  for classifier in classifiers:
    classifier.train(dataset)

  classifiers.append(SklearnClassifier(prefitted(MIXED, dataset)))

  for classifier in classifiers:
    print()
    print(classifier)
    
    Y_test_pred = classifier.test(dataset)

    display_entries(classifier.name, Y_test_pred)

    classifier.feature_importances()

    # if classifier is not RANDOM:
    #   preview_prediction(data, classifier, 'Y_test', classifier.dataset.Y_test, 'Y_pred', classifier.Y_pred)

  compare_classifiers(dataset, classifiers, debug=True)
