from typing import Union

from sklearn.base import ClassifierMixin
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier, VotingClassifier

from data import exists_data, get_data, save_data, load_data
from classifier import Dataset, Classifier, RandomClassifier, SklearnClassifier, GridSearchClassifier
from visualization import display_entries, preview_prediction, plot_balances
from settings import START_BALANCE, BET, TRANSACTION_FEE, PRIZE_FEE, random_payout

import preprocess

def train(classifier: Classifier, dataset: Dataset):
  Y_test_pred = classifier.train(dataset)

  display_entries(classifier.name, Y_test_pred)

  # classifier.feature_importances()

  return Y_test_pred

def search(classifier: Union[ClassifierMixin, SklearnClassifier]) -> Classifier:
  if isinstance(classifier, (ClassifierMixin, SklearnClassifier)):
    search_params = {
      'n_estimators': [100, 500],
      'min_weight_fraction_leaf': [0, 0.05, 0.2],
      'max_depth': [1, 2, 20, None]
    }
    return GridSearchClassifier(classifier, search_params, cv=1, verbose=True)
  return classifier

def compare_classifiers(dataset: Dataset, classifiers: list[Classifier]):
  balances = dict(map(lambda classifier: (classifier, [START_BALANCE]), classifiers))

  Y_test = dataset.Y_test

  # for each round
  for r in range(len(Y_test)):
    payout = random_payout() # winner multiplier

    for classifier in classifiers:
      balance_history = balances[classifier]
      balance = balance_history[-1] - BET - TRANSACTION_FEE

      if classifier.Y_pred[r] == Y_test[r]:
        prize = BET * (payout - 1)
        balance += BET + (prize * (1 - PRIZE_FEE))
      
      balance_history.append(balance)
  
  # final results
  for classifier in classifiers:
    final_balance = balances[classifier][-1]
    print(f"{classifier} final balance: {final_balance:.3f} ({((final_balance - START_BALANCE)*100):.2f}%)")

  plot_balances(balances, START_BALANCE)

if __name__ == "__main__":
  # Load data
  if not exists_data():
    save_data(get_data())

  data = load_data()

  # Prepare data for training
  dataset = preprocess.get_dataset(data, preview=False, correlation_matrix=False)

  # Classifiers
  RANDOM = RandomClassifier()
  RANDOM_FOREST = RandomForestClassifier(n_estimators=500)
  EXTRA_RANDOM_FOREST = ExtraTreesClassifier(n_estimators=500)
  GRADIENT_BOOST = GradientBoostingClassifier(n_estimators=100, max_depth=20)
  mix_classifiers = [('RandomForestClassifier', RANDOM_FOREST), ('GradientBoostingClassifier', GRADIENT_BOOST), ('ExtraTreesClassifier', EXTRA_RANDOM_FOREST)]
  MIXED = VotingClassifier(mix_classifiers, voting='soft', n_jobs=-1, verbose=True)

  classifiers = [RANDOM, SklearnClassifier(RANDOM_FOREST), SklearnClassifier(MIXED)]

  # Training
  for classifier in classifiers:
    train(classifier, dataset)

    # if classifier is not RANDOM:
    #   preview_prediction(data, classifier, 'Y_test', classifier.dataset.Y_test, 'Y_pred', classifier.Y_pred)

  compare_classifiers(dataset, classifiers)
