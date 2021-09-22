'''
cours en france de machine learning
Github: @rafaelmm82
'''

# Importer les librairies
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# La gestion des données
# ----------------------

# Importer le dataset
filename = 'mlfrancais/Partie 1 - Data Preprocessing/Data.csv'
dataset = pd.read_csv(filename, header=0)
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values


# Gérer les données manquantes
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer.fit(X[:, 1:])
X[:, 1:] = imputer.transform(X[:, 1:])

# on peut fait de comparaison du description de donnéés e le resultat avec le imputer
# dataset.describe()
# print(imputer.transform(X[:, 1:]))


# Gérer les variable catéoriques
from sklearn.preprocessing import LabelEncoder, LabelBinarizer

binarizer = LabelBinarizer()
new_columns = binarizer.fit_transform(X[:, 0])
X = np.column_stack((new_columns, X[:,1:]))

label_enc = LabelEncoder()
y = label_enc.fit_transform(y)


# Diviser le dataset entre le Training set et le Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


# Feature Scaling
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

