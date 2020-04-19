from keras.models import Sequential  
from keras.layers import Dense
import numpy as np

def generateData():
    i = 1
    X =[]
    Y = []
    while i < 100:
        X.append(i)
        Y.append(i - 3 )
        i = i + 1
    return np.array( X), np.array(Y)

model = Sequential()
model.add(Dense(1,input_dim = 1))
model.compile(loss='mean_absolute_error', optimizer ='adam')
model.summary()

X,Y=generateData()

model.fit(X[0:70],Y[0:70], epochs=200, batch_size=1)

scores= model.evaluate(x=X[70:],y=Y[70:])

print(scores)

t = np.array([2020,4040])
print(model.predict(t))