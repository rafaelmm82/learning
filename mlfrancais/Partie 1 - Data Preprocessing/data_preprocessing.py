#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 17 18:03:24 2019

@author: @rafaelmm82
github: @rafaelmm82

Les codes dans cettes scripts font partie de mon apprentissage du français.
J'ai décidé de commencer un cours  de ma compétence technique pour faciliter
l'assimilation du vocabulaire et de la structure de la langue française.


Partie 1 - Data-preprocessing

"""

# Importer les librairies
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.impute import SimpleImputer

# Importer le dataset
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, -1].values

# Gérer les données manquantes
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])

# Gérer les variables catégoriques



