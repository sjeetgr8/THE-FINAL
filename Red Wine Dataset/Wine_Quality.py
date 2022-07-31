# -*- coding: utf-8 -*-
"""
Created on Fri Jul 29 18:02:44 2022

@author: Satya
"""

import pandas as pd
import numpy as np
from sklearn import linear_model
import os

os.chdir("E:/Google Drive/Github/THE FINAL/Red Wine Dataset")

df=pd.read_csv('winequality-red.csv')
df

