from typing import Union
from collections.abc import Iterable
from numpy.typing import ArrayLike

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif # alternative: chi2

class Dataset:

  def __init__(self, X: pd.DataFrame, Y: ArrayLike = None):
    self.X = X
    self.Y = np.array(Y) if Y is not None else None
    self.new_columns = []
  
  @property
  def features(self):
    return self.X.columns
  
  @property
  def has_labels(self):
    return self.Y is not None and len(self.Y) > 0
  
  @property
  def labeled_data(self):
    if not self.has_labels:
      return self.X
    return pd.concat([self.X, pd.DataFrame(self.Y, columns=['Y'], index=self.X.index)], axis=1)
  
  def add_column(self, name: str, values: ArrayLike):
    self.X[name] = values
    self.new_columns.append(name)
  
  def add_columns(self, names: list[str], values: Iterable[ArrayLike], prepend: bool = False):
    new_columns = pd.DataFrame(np.array(values), columns=names, index=self.X.index)
    new_X = [new_columns, self.X] if prepend else [self.X, new_columns]
    self.X = pd.concat(new_X, axis=1)
    self.new_columns.extend(names)
  
  def train_test_split(self, train_percentage: float = 0.8, normalize: Union[bool, Iterable[str]] = True):
    split_at = int(len(self.X)*train_percentage)

    self.X_train = self.X[:split_at].copy(deep=False)
    self.X_test = self.X[split_at:].copy(deep=False)

    self.Y_train = self.Y[:split_at] if self.has_labels else None
    self.Y_test = self.Y[split_at:] if self.has_labels else None

    if (isinstance(normalize, Iterable) and len(normalize)) or bool(normalize):
      scaler = StandardScaler()

      train = len(self.X_train) > 0
      test = len(self.X_test) > 0

      scaler_data = self.X_train if train else self.X_test

      if len(scaler_data):
        if isinstance(normalize, Iterable):
          scaler.fit(scaler_data[normalize])
          if train:
            self.X_train[normalize] = scaler.transform(self.X_train[normalize])
          if test:
            self.X_test[normalize] = scaler.transform(self.X_test[normalize])
        else:
          scaler.fit(scaler_data)
          if train:
            self.X_train[:] = scaler.transform(self.X_train)
          if test:
            self.X_test[:] = scaler.transform(self.X_test)
  
  def best_features(self, top_features: int = 10) -> pd.DataFrame:
    top_features = min(10, len(self.features))
    
    kbest = SelectKBest(score_func=f_classif, k=top_features)
    features_fit = kbest.fit(self.X_train, self.Y_train)
    feature_scores = pd.concat([pd.DataFrame(self.features), pd.DataFrame(features_fit.scores_)], axis=1)
    feature_scores.columns = ['Feature', 'Score']
    feature_scores = feature_scores.nlargest(top_features, 'Score')

    if not feature_scores.empty:
      print("Most relevant features")
      print(feature_scores)
    
    return feature_scores

  def correlation_matrix(self) -> pd.DataFrame:
    return self.labeled_data.corr()

  def __str__(self) -> str:
    return str(self.labeled_data)

  def __len__(self) -> int:
    return len(self.X)
  
  def __getitem__(self, key: str):
    return self.X[key]

  def __setitem__(self, key: Union[str, list[str]], value: Union[ArrayLike, Iterable[ArrayLike]]):
    if isinstance(key, list):
      self.add_columns(key, value)
    else:
      self.add_column(key, value)
