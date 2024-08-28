# -*- coding: utf-8 -*-
"""
Created on Wed Aug 21 00:04:42 2024

@author: Lenovo
"""

import numpy as np
import pickle
import streamlit

# loading the saved model
loaded_model = pickle.load(open('C:/Users/Lenovo/Desktop/Twitter SE Deployment/trained_model.sav', 'rb'))
