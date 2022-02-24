import os 
import pandas as pd
import sklearn.model_selection as ms
import sklearn.linear_model as lm
import matplotlib.pyplot as plt 
import numpy as np

dataset = pd.read_csv("teste_ia/Regressao linear/dataset_salario.csv",sep=';')
dataset = dataset.dropna()
y = dataset.iloc[:, 0].values
X = dataset.iloc[:, 1].values

Xm = np.array([X])
X = Xm.T

regressor = lm.LinearRegression()
regressor.fit(X,y)

y_pred = regressor.predict(X)

plt.scatter(X, y, color = 'red')
plt.plot(X, regressor.predict(X), color = 'blue')
plt.title('Treino')
plt.show()

plt.scatter(X, y, color = 'red')
plt.plot(X, regressor.predict(X), color = 'blue')
plt.title('Teste')
plt.show()

print(regressor.predict([[10]]))
