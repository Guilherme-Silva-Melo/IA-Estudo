from keras.models import Sequential
from keras.layers import Dense 
import numpy as np

x = np.array([[0.0,0.0],[0.2,0.1],[0.25,0.25],[0.125,0.125]])
y = np.array([[0.0],[0.3],[0.5],[0.25]])
z = np.array([[0.3,0.7],[0.2,0.9],[0.1,0.4]])

model = Sequential()

model.add(Dense(3,input_dim=2,activation='tanh'))
model.add(Dense(1,activation='sigmoid'))

model.compile(optimizer='adam',loss ='mse', metrics=['acc'])
model.fit(x,y,epochs=1)

print(model.predict(z))