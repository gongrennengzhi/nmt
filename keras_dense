from keras.models import Sequential  
from keras.layers import Dense
import numpy as np
import random
def generateData():
    i = 1
    X =[]
    Y = []
    while i < 1000:
        t1 = random.randint(0,1000)
        t2 = random.randint(0,1000)
        X.append([t1, t2])
        Y.append( t1 + t2)
        i = i + 1
    return np.array( X), np.array(Y)

model = Sequential()
model.add(Dense(1,input_dim = 2))
model.compile(loss='mean_absolute_error', optimizer ='adam')
model.summary()

X,Y=generateData()

model.fit(X[0:700],Y[0:700], epochs=500, batch_size=1)

scores= model.evaluate(x=X[70:],y=Y[70:])

print(scores)

t = np.array([[2020,4040]])
print(model.predict(t))

