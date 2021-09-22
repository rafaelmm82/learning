'''
cours en france de machine learning
Github: @rafaelmm82
'''

# Importer les librairies
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# ----------------------
# La gestion des données
# ----------------------

# Importer le dataset
filename = 'mlfrancais/Partie 2 - Regression/Salary_Data.csv'
dataset = pd.read_csv(filename, header=0)
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Il n'y a pas des données manquantes
# Il n'y a pas des variable catéoriques

# Diviser le dataset entre le Training set et le Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/3, random_state=0)

# Pas besoin Feature Scaling

# ----------------------
# L'aprentissage
# --------------

# Construction du modèle
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Faire de nouvelles prédictions
y_pred = regressor.predict(X_test)
regressor.predict(np.array([[15]]))

# Visualiser les résultats
plt.scatter(X_test, y_test, color='red')
plt.plot(X_train, regressor.predict(X_train), color='blue')
plt.title("Salaire vs Experience")
plt.xlabel("Experience")
plt.ylabel("Salaire")
plt.show()
