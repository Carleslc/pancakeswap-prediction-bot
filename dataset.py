from numpy.typing import ArrayLike

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

class Dataset:

  def __init__(self, X: pd.DataFrame, Y: ArrayLike = None):
    self.X = X
    self.Y = np.array(Y) if Y is not None else None
  
  @property
  def features(self):
    return self.X.columns.values
  
  @property
  def labeled_data(self):
    if self.Y is None:
      return self.X
    return pd.concat([self.X, pd.DataFrame(self.Y, columns=['Y'], index=self.X.index)], axis=1)
  
  def add_column(self, name: str, values: ArrayLike):
    self.X[name] = values
  
  def add_columns(self, names: list[str], values: list[ArrayLike], prepend: bool = False):
    new_columns = pd.DataFrame(np.array(values), columns=names, index=self.X.index)
    new_X = [new_columns, self.X] if prepend else [self.X, new_columns]
    self.X = pd.concat(new_X, axis=1)
  
  def train_test_split(self, train_percentage: float = 0.8, normalize: bool = True):
    split_at = int(len(self.X)*train_percentage)

    self.X_train = self.X[:split_at]
    self.X_test = self.X[split_at:]

    self.Y_train = self.Y[:split_at] if self.Y is not None else None
    self.Y_test = self.Y[split_at:] if self.Y is not None else None

    if normalize:
      scaler = StandardScaler()
      scaler.fit(self.X)
      self.X_train = scaler.transform(self.X_train)
      self.X_test = scaler.transform(self.X_test)
  
  def best_features(self, top_features: int = 10) -> pd.DataFrame:
    top_features = min(10, len(self.features))
    
    kbest = SelectKBest(score_func=chi2, k=top_features)
    features_fit = kbest.fit(self.X, self.Y)
    feature_scores = pd.concat([pd.DataFrame(self.features), pd.DataFrame(features_fit.scores_)], axis=1)
    feature_scores.columns = ['Feature', 'Score']
    feature_scores = feature_scores.nlargest(top_features, 'Score')

    if not feature_scores.empty:
      print("Most relevant features")
      print(feature_scores)
    
    return feature_scores

  def correlation_matrix(self) -> pd.DataFrame:
    return self.labeled_data.corr()
