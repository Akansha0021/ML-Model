# -*- coding: utf-8 -*-
"""
Created on Tue Aug 20 22:32:16 2024

@author: Lenovo
"""

import numpy as np
import pickle

# loading the saved model
loaded_model = pickle.load(open('C:/Users/Lenovo/Desktop/Twitter SE Deployment/trained_model.sav', 'rb'))
         

X_new = X_test[200]
print(Y_test[200])

prediction = loaded_model.predict(X_new)
print(prediction)

if (prediction[0]==0):
  print('Negative Tweet')
else:
  print('Positive Tweet')                      
                                