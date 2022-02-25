import tensorflow as tf
from keras.datasets import mnist
from keras.models import Sequential 
from keras.layers import Dense 
from numpy import reshape

(x,y),(_,_) = mnist.load_data()

x = x.reshape(60000,784)

x = x/255.0

y = tf.keras.utils.to_categorical(y,10)

model = Sequential()

model.add(Dense(64,input_shape=(784,),activation='relu'))
model.add(Dense(64,activation='relu'))
model.add(Dense(10,activation='softmax'))

model.compile(optimizer='adam',loss='categorical_crossentropy',metrics='acc')
model.fit(x,y, batch_size=128, epochs = 10)

print(model.predict(x))
