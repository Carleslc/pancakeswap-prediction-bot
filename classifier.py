from typing import Union

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV, ShuffleSplit
from sklearn.ensemble import ExtraTreesClassifier, VotingClassifier, StackingClassifier
from sklearn.metrics import accuracy_score

from dataset import Dataset
from visualization import plot_barh
from settings import RANDOM_STATE

class Classifier(BaseEstimator, ClassifierMixin, ABC):

  def __init__(self, dataset: Dataset, name: str = None):
    self.dataset = dataset
    self.name = name or type(self).__name__
  
  def train(self):
    self.fit(self.dataset.X_train, self.dataset.Y_train)
  
  def test(self) -> np.ndarray:
    self.Y_pred = self.predict(self.dataset.X_train)
    self.accuracy(self.dataset.Y_train, 'Training')

    Y_test_pred = None

    if self.dataset.X_test is not None:
      Y_test_pred = self.predict(self.dataset.X_test)
      self.Y_pred = Y_test_pred
    
      if self.dataset.Y_test is not None:
        self.accuracy(self.dataset.Y_test, 'Test')

    return Y_test_pred
  
  @abstractmethod
  def fit(self, X_train, Y_train):
    ...
  
  @abstractmethod
  def predict(self, X) -> np.ndarray:
    ...
  
  @abstractmethod
  def probabilities(self, X) -> np.ndarray:
    ...
  
  def predict_proba(self, X) -> np.ndarray:
    return self.probabilities(X)
  
  def accuracy(self, Y, title: str = None) -> float:
    score = accuracy_score(Y, self.Y_pred)
    print(f"Accuracy ({title or self.name}): {score:.3f}")
    return score
  
  def feature_importances(self) -> pd.DataFrame:
    return pd.DataFrame([None] * len(self.dataset.features), columns=['Importance'], index=self.dataset.features)
  
  def get_params(self, deep=True):
    return {
      'dataset': self.dataset
    }
  
  def __str__(self):
    return self.name
  
  def get_column(self, X: Union[pd.DataFrame, np.ndarray], column_name: str):
    return X[column_name] if type(X) == pd.DataFrame else X.T[self.dataset.features.get_loc(column_name)]

  @staticmethod
  def get_values(X):
    return X.values if type(X) == pd.DataFrame else X

class SklearnClassifier(Classifier):

  def __init__(self, dataset: Dataset, classifier: ClassifierMixin, name: str = None):
    super().__init__(dataset, name or type(classifier).__name__)
    self.classifier = classifier
  
  def fit(self, X_train, Y_train):
    self.Y_train = Y_train
    SklearnClassifier.set_classes(self, Y_train)
    self.classifier.fit(self.get_values(X_train), Y_train)
  
  def predict(self, X) -> np.ndarray:
    return self.classifier.predict(self.get_values(X))
  
  def probabilities(self, X) -> np.ndarray:
    return self.classifier.predict_proba(self.get_values(X))
  
  def feature_importances(self, plot: bool = True, limit: int = 30) -> pd.DataFrame:
    if hasattr(self.classifier, 'feature_importances_'):
      feature_importances = pd.DataFrame(self.classifier.feature_importances_, columns=['Importance'], index=self.dataset.features).sort_values(by='Importance', ascending=False)
      
      feature_importances_limited = feature_importances

      if len(feature_importances_limited) > limit:
        feature_importances_limited = feature_importances_limited.iloc[:limit]

      print(feature_importances_limited)

      if plot:
        plot_barh(feature_importances_limited, title=f"Feature importances ({self.name})")
      
      return feature_importances
    return super().feature_importances()
  
  def get_params(self, deep=True):
    params = super().get_params(deep)
    params['classifier'] = self.classifier
    return params
  
  @staticmethod
  def set_classes(classifier, Y_train):
    classifier.le_ = LabelEncoder().fit(Y_train)
    classifier.classes_ = classifier.le_.classes_

def wrap_classifiers(dataset: Dataset, classifiers: list[Union[ClassifierMixin, SklearnClassifier]]) -> list[SklearnClassifier]:
  sklearn_classifiers = []
  for classifier in classifiers:
    if not isinstance(classifier, SklearnClassifier):
      classifier = SklearnClassifier(dataset, classifier)
    sklearn_classifiers.append(classifier)
  return sklearn_classifiers

def unwrap(classifier: Union[ClassifierMixin, SklearnClassifier]) -> ClassifierMixin:
  while isinstance(classifier, SklearnClassifier):
    classifier = classifier.classifier
  return classifier

def allow_no_cv(cv):
  if not cv or cv == 1:
    return ShuffleSplit(test_size=0.2, n_splits=1, random_state=RANDOM_STATE)
  return cv

class GridSearchClassifier(SklearnClassifier):

  def __init__(self, dataset: Dataset, classifier: ClassifierMixin, params: dict[str, list[float]], cv = 5, parallel: bool = True, verbose: bool = False):
    self.params = params
    self.verbose = verbose

    grid_classifier = GridSearchCV(classifier, params, cv=allow_no_cv(cv), n_jobs=-1 if parallel else None, verbose=3 if verbose else 0)

    super().__init__(dataset, grid_classifier, f'{self.__class__.__name__} ({classifier.__class__.__name__})', verbose)
  
  def fit(self, X_train, Y_train):
    super().fit(X_train, Y_train)

    if self.verbose:
      print(self.classifier.best_params_)
      print(self.classifier.best_estimator_)
  
  def get_params(self, deep=True):
    params = super().get_params(deep)
    params['params'] = self.params
    return params

class AbstractEnsembleClassifier(SklearnClassifier):

  def __init__(self, dataset: Dataset, classifiers: list[Union[ClassifierMixin, SklearnClassifier]], name: str = None, prefit: bool = False, parallel: bool = True, verbose: bool = False):
    self.classifiers = classifiers
    self.estimators_ = wrap_classifiers(dataset, classifiers)

    self.verbose = verbose
    self._parallel = parallel

    estimators = list(map(lambda classifier: (classifier.name, classifier), self.estimators_))
    ensemble_classifier = self.get_ensemble_classifier(estimators)

    if prefit:
      prefit_classifier = classifiers[0] # must be a trained Classifier
      ensemble_classifier.estimators_ = classifiers
      SklearnClassifier.set_classes(ensemble_classifier, prefit_classifier.Y_train)

    super().__init__(dataset, ensemble_classifier, name)
  
  @abstractmethod
  def get_ensemble_classifier(self, estimators) -> ClassifierMixin:
    ...
  
  def fit(self, X_train, Y_train):
    super().fit(X_train, Y_train)
    self.estimators_ = wrap_classifiers(self.dataset, self.classifier.estimators_)
  
  def get_params(self, deep=True):
    return {
      'dataset': self.dataset,
      'classifiers': self.classifiers
    }

class VoteClassifier(AbstractEnsembleClassifier):

  def __init__(self, dataset: Dataset, classifiers: list[Union[ClassifierMixin, SklearnClassifier]], name: str = None, voting: str = 'hard', weights: list[float] = None, prefit: bool = False, parallel: bool = True, verbose: bool = False):
    self.voting = voting
    self.weights = weights

    super().__init__(dataset, classifiers, name, prefit, parallel, verbose)
  
  def get_ensemble_classifier(self, estimators) -> ClassifierMixin:
    return VotingClassifier(estimators, voting=self.voting, weights=self.weights, verbose=self.verbose, n_jobs=-1 if self._parallel else None)
  
  def probabilities(self, X) -> np.ndarray:
    if self.voting == 'hard':
      # TODO: Replace with the mean of probabilities for classifiers matching the prediction
      predictions = np.array([classifier.predict(X) for classifier in self.classifiers]).T
      class_counts = np.apply_along_axis(lambda x_predictions: np.bincount(x_predictions, minlength=len(self.classifier.classes_)), axis=1, arr=predictions)
      probs = class_counts / np.full(class_counts.shape, len(self.classifiers))
      return probs
    return super().probabilities(X)
  
  def get_params(self, deep=True):
    params = super().get_params(deep)
    params.update({
      'voting': self.voting,
      'weights': self.weights
    })
    return params

class StackClassifier(AbstractEnsembleClassifier):

  def __init__(self, dataset: Dataset, classifiers: list[Union[ClassifierMixin, SklearnClassifier]], name: str = None, passthrough: bool = False, final_estimator: ClassifierMixin = None, cv = 2, parallel: bool = True, verbose: bool = False):
    self.cv = cv
    self.passthrough = passthrough
    self.final_estimator = final_estimator or ExtraTreesClassifier(n_estimators=500, random_state=RANDOM_STATE)

    super().__init__(dataset, classifiers, name, parallel=parallel, verbose=verbose)
  
  def get_ensemble_classifier(self, estimators) -> ClassifierMixin:
      return StackingClassifier(estimators, final_estimator=self.final_estimator, cv=self.cv, passthrough=self.passthrough, verbose=2 if self.verbose else 0, n_jobs=-1 if self._parallel else None)
  
  def get_params(self, deep=True):
    params = super().get_params(deep)
    params.update({
      'passthrough': self.passthrough,
      'cv': self.cv
    })
    return params

class RandomClassifier(Classifier):

  def __init__(self, dataset: Dataset, random_state: np.random.RandomState = None):
    super().__init__(dataset)
    self.random = random_state or np.random.RandomState()
  
  def fit(self, _, Y_train):
    self.classes = np.unique(Y_train)
  
  def predict(self, X) -> np.ndarray:
    return self.random.choice(self.classes, size=len(X))

  def probabilities(self, X) -> np.ndarray:
    return np.array([[1/len(self.classes)] * len(self.classes)] * len(X))
