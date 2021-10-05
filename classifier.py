from typing import Union

from abc import ABC, abstractmethod

from sklearn.base import ClassifierMixin
from sklearn.model_selection import GridSearchCV, ShuffleSplit
from sklearn.metrics import accuracy_score

from dataset import Dataset
from visualization import plot_barh

import numpy as np
import pandas as pd

class Classifier(ABC):

  def __init__(self, name: str):
    self.name = name

  def train(self, dataset: Dataset) -> np.ndarray:
    print(f"\nTraining {self.name}...")

    self.dataset = dataset

    self._fit()

    print(f"{self.name} trained")

    self.predict(dataset.X_train)
    self.accuracy(dataset.Y_train, 'Training')

    Y_test_pred = None

    if dataset.X_test is not None:
      Y_test_pred = self.predict(dataset.X_test)
    
      if dataset.Y_test is not None:
        self.accuracy(dataset.Y_test, 'Test')

    return Y_test_pred
  
  @abstractmethod
  def _fit(self):
    ...
  
  @abstractmethod
  def _predict(self, X) -> np.ndarray:
    ...
  
  def predict(self, X) -> np.ndarray:
    self.Y_pred = self._predict(X)
    return self.Y_pred
  
  def accuracy(self, Y, title: str = None) -> float:
    score = accuracy_score(Y, self.Y_pred)
    print(f"Accuracy ({title or self.name}): {score:.3f}")
    return score
  
  def feature_importances(self):
    self.feature_importances = np.zeros(len(self.dataset.features))
  
  def __str__(self):
    return self.name

class SklearnClassifier(Classifier):

  def __init__(self, classifier: ClassifierMixin, name: str = None):
    self.classifier = classifier
    super().__init__(name or type(classifier).__name__)
  
  def _fit(self):
    self.classifier.fit(self.dataset.X_train, self.dataset.Y_train)
  
  def _predict(self, X) -> np.ndarray:
    return self.classifier.predict(X)
  
  def feature_importances(self, plot: bool = True, limit: int = 30):
    if hasattr(self.classifier, 'feature_importances_'):
      self.feature_importances = pd.DataFrame(self.classifier.feature_importances_, columns=['Importance'], index=self.dataset.features).sort_values(by='Importance', ascending=False)
      
      if len(self.feature_importances) <= limit:
        print(self.feature_importances)

        if plot:
          plot_barh(self.feature_importances, title=f"Feature importances ({self.name})")

class GridSearchClassifier(SklearnClassifier):

  def __init__(self, classifier: Union[ClassifierMixin, SklearnClassifier], params: dict[str, list[float]], cv = 5, parallel: bool = True, verbose: bool = False):
    if cv is None:
      cv = [(slice(None), slice(None))] # no fold
    elif cv == 1:
      cv = ShuffleSplit(test_size=0.2, n_splits=1) # 1 fold
    
    self.params = params
    self.verbose = verbose

    if isinstance(classifier, SklearnClassifier):
      classifier = classifier.classifier
    elif not isinstance(classifier, ClassifierMixin):
      raise TypeError(f"Expected SklearnClassifier but found {type(classifier).__name__}")

    grid_classifier = GridSearchCV(classifier, params, cv=cv, n_jobs=-1 if parallel else None, verbose=3 if verbose else 0)

    super().__init__(grid_classifier, f'{self.__class__.__name__} ({classifier.__class__.__name__})')
  
  def _fit(self):
    super()._fit()

    if self.verbose:
      print(self.classifier.best_estimator_)
      # print(self.classifier.best_params_)

class RandomClassifier(Classifier):

  def __init__(self):
    super().__init__('RandomClassifier')
  
  def _fit(self):
    pass
  
  def _predict(self, X) -> np.ndarray:
    return np.random.randint(0, 2, size=len(X)) # random values within [0, 2) aka [0,1]