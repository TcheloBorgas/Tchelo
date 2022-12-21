import pandas as pd
import numpy as np
import joblib

clf2 = joblib.load('modelo')

FEATURES = ['sepal length (cm)',
 'sepal width (cm)',
 'petal length (cm)',
 'petal width (cm)']

FLORES = np.array(['setosa', 'versicolor', 'virginica'])
def iris(sepal_length,	sepal_width, petal_length,petal_width):
  
  x = np.array([sepal_length,	sepal_width, petal_length,petal_width	])
  x = pd.DataFrame(np.array(x).reshape(1,-1), columns=FEATURES)
  # print(type(x))
  flor =  clf2.predict(x)[0]
  return FLORES[flor]
