import numpy as np
import pandas as pd

from typing import Optional, Union, Tuple

def fit(
    x_train: Union[pd.DataFrame, np.ndarray],
    y_train: Union[pd.Series, np.ndarray, list],
    half_life: Optional[int] = None,
    ):
    model = None # here is where you create and train the tcn model
    return model

def predict(model, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
    proba = None # Here is where you use the created and trained TCN model to predict 
    return np.asarray(proba, dtype=float)