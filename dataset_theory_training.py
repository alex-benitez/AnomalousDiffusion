'''
Trying to train a network on balanced numbers of trajectories
'''
from andi_datasets.datasets_challenge import challenge_theory_dataset
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import BatchNormalization
from keras.layers import Dropout
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from tensorflow.keras.optimizers import Adam
import random
from playsound import playsound
import time
import Convtools as ct
import seaborn as sn
from andi_datasets.datasets_theory import datasets_theory

N = 100000
min_T = 320
size = 10
start = time.time()

trajectories = [[] for i in range(5)]
for i in range(5):
    print('aaaaaaa')
    for big in range(N):
        if i==0 or i==1:
            exponent = 0.5
        else:
            exponent = 1.5
        trajectories[i].append(datasets_theory().create_dataset(T=min_T,N_models=1,models=i,exponents=[exponent])[0])
    
# alpha = [[0]*i + [1] + [0]*(4-i) for i in range(5)]  Brilliantly condensed code which will never see the light of day again

print(len(trajectories))

print(len(trajectories[3]))


for i in range(len(trajectories)):
    
    for j in range(len(trajectories[i])):
        trajectories[i],alpha = ct.normalize_data(trajectories[i],[i for x in range(len(trajectories))])
        del trajectories[i][j][320]
        del trajectories[i][j][320]

model = Sequential()
model.add(Conv1D(filters=16, kernel_size=size, activation='relu', input_shape=(320,1)))
model.add(BatchNormalization())
model.add(MaxPooling1D(pool_size=4))
model.add(Conv1D(filters=32, kernel_size=4, activation='relu'))#, input_shape=(304, 16)))
model.add(BatchNormalization())
model.add(MaxPooling1D(pool_size=2))
model.add(Conv1D(filters=64, kernel_size=4, activation='relu'))#, input_shape=(304, 16)))
model.add(BatchNormalization())
model.add(MaxPooling1D(pool_size=2))
model.add(Dense(50, activation='relu'))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(5,activation='softmax'))
model.summary()
model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')


'''
Fittin the model time, how does this work? Time to figure it out 

'''
# model.fit(X2[0], alpha, epochs=30)



for i in range(5):
    for pos,trajectory in enumerate(trajectories[i]):
        trajectory = np.array(trajectory)
        trajectories[i][pos] = np.reshape(trajectory,(1,320,1))

accurate = 0
yhat = [[] for i in range(5)]
models = ['attm', 'ctrw', 'fbm', 'lw', 'sbm']

for i in range(5):
    for j in range(len(trajectories[i])):
        yhat[i].append(model.predict(trajectories[i][j],verbose=0)[0])
    
# print(yhat)
heatmap_list = [[0 for i in range(5)] for j in range(5)]

for pos,model in enumerate(yhat):
    for traj in model:
        heatmap_list[pos][np.argmax(traj)] += 1
    

sn.heatmap(heatmap_list,cmap='Blues',xticklabels=models,yticklabels=models)
# print(yhat)



accurate = 0
for i in range(5):
    yhat = model.predict(trajectories[i],verbose=0)
    for i in range(len(yhat)):
        if np.argmax(yhat[i]) == i:
            accurate += 1
    

print('The model had a {}% accuracy'.format(100*accurate/len(yhat)))