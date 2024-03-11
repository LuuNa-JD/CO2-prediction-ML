import matplotlib.pyplot as plt
import pandas as pd
import pylab as pl
import numpy as np

# télécharge le jeu de données
df = pd.read_csv("FuelConsumptionCo2.csv")

# affiche les 5 premières lignes du jeu de données
df.head()


# résumé des données
df.describe()

# sélectionne certaines caractéristiques pour explorer plus avant
cdf = df[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB','CO2EMISSIONS']]
cdf.head(9)

# trace les histogrammes des variables
viz = cdf[['CYLINDERS','ENGINESIZE','CO2EMISSIONS','FUELCONSUMPTION_COMB']]
viz.hist()
plt.show()

# trouve la relation entre la consommation de carburant et les emissions de CO2 en utilisant la fonction scatter
# plt.scatter(cdf.FUELCONSUMPTION_COMB, cdf.CO2EMISSIONS,  color='blue')
# plt.xlabel("FUELCONSUMPTION_COMB")
# plt.ylabel("Emission")
# # plt.show()

# trouve la relation entre la taille du moteur et les emissions de CO2 en utilisant la fonction scatter
plt.scatter(cdf.ENGINESIZE, cdf.CO2EMISSIONS,  color='blue')
plt.xlabel("Engine size")
plt.ylabel("Emission")
plt.show()

# trouve la relation entre les cylindres et les emissions de CO2 en utilisant la fonction scatter
# plt.scatter(cdf.CYLINDERS, cdf.CO2EMISSIONS, color="red")
# plt.xlabel("cylindres")
# plt.ylabel("Emission")
# plt.show()

# Crée un jeu de données d'entrainement et de test

# On divise notre jeu de données en deux parties, une pour l'entrainement et l'autre pour le test
# 80% des données pour l'entrainement et 20% pour le test
msk = np.random.rand(len(df)) < 0.8
train = cdf[msk]
test = cdf[~msk]

# Modèle de régression simple

# On utilise la régression linéaire pour prédire les émissions de CO2 en fonction de la taille du moteur
from sklearn import linear_model
regr = linear_model.LinearRegression()
train_x = np.asanyarray(train[['ENGINESIZE']])
train_y = np.asanyarray(train[['CO2EMISSIONS']])
regr.fit (train_x, train_y)
# Les coefficients
print ('Coefficients: ', regr.coef_)
print ('Intercept: ',regr.intercept_)

# On trace la droite de régression
plt.scatter(train.ENGINESIZE, train.CO2EMISSIONS,  color='blue')
plt.plot(train_x, regr.coef_[0][0]*train_x + regr.intercept_[0], '-r')
plt.xlabel("Engine size")
plt.ylabel("Emission")
plt.show()

# Evaluation

# On évalue le modèle en utilisant le jeu de données de test
from sklearn.metrics import r2_score

test_x = np.asanyarray(test[['ENGINESIZE']])
test_y = np.asanyarray(test[['CO2EMISSIONS']])
test_y_ = regr.predict(test_x)

# On évalue le modèle en utilisant différentes métriques
# La moyenne de l'erreur absolue -> doit être proche de 0
print("Mean absolute error: %.2f" % np.mean(np.absolute(test_y_ - test_y)))
# La somme résiduelle des carrés (MSE) -> doit être proche de 0
print("Residual sum of squares (MSE): %.2f" % np.mean((test_y_ - test_y) ** 2))
# Le score R2 -> doit être proche de 1
print("R2-score: %.2f" % r2_score(test_y , test_y_) )

# On utilise la régression linéaire pour prédire les émissions de CO2 en fonction de la consommation de carburant

# train_x = train[["FUELCONSUMPTION_COMB"]]
# train_y = train[["CO2EMISSIONS"]]


# regr = linear_model.LinearRegression()

# regr.fit(train_x, train_y)

# predictions = regr.predict(test_x)

# print("Mean absolute error: %.2f" % np.mean(np.absolute(predictions - test_y)))
# print("Residual sum of squares (MSE): %.2f" % np.mean((predictions - test_y) ** 2))
# print("R2-score: %.2f" % r2_score(predictions , test_y) )

# sauvegarder le modèle

import joblib

joblib.dump(regr, 'model.pkl')

# charger le modèle

regr = joblib.load('model.pkl')
