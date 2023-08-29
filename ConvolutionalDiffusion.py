'''
Gonna have to do some magic here, just find a way to apply your convolutional network to these timeseries my guy,
lets start by thinking, what exactly is a CNN. You split it into different squares and each one is assigned to a neuron,
then afterwards you do that again with the remaining neurons, were gonna need a lot of padding for this one, I think
'''
from andi_datasets.datasets_challenge import challenge_theory_dataset
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import BatchNormalization
from keras.layers import Dropout
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
import random
from playsound import playsound
import time
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


N = 5000
min_T = 320
size = 10
start = time.time()
X1, Y1, X2, Y2, X3, Y3 = challenge_theory_dataset(N=N,tasks=2,dimensions=1,min_T=min_T,max_T=min_T+1,save_dataset=True,path_datasets='./datasets')


'''
The structure is X2 = [ [ [],[]...[] ], [],[] ], so to get the first trajectory for X2, t = X2[0][0]
For Y2 the structure is the same, except that it only stores one number per trajectory
So now we can generate our model as such:
'''


for pos, traj in enumerate(X2[0]):
    if traj[3] > 10000:
        del X2[0][pos]
        del Y2[0][pos]
    avg = sum(X2[0][pos])/len(X2[0][pos])
    sd = (sum([((x - avg) ** 2) for x in X2[0][pos]]) / len(X2[0][pos]))**0.5
    X2[0][pos] = [(i-avg)/sd for i in X2[0][pos]]
    plt.plot(X2[0][pos])
    
  
print('Trajectories generated and culled\n')
print(time.time()-start)
alpha = []

for pos, val in enumerate(Y2[0]):
    temp = [0,0,0,0,0]
    temp[int(val)] += 1
    alpha.append(temp)


        
print(time.time()-start)


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
model.fit(X2[0], alpha, epochs=30)


X1, Y1, test_data, test_check, X3, Y3 = challenge_theory_dataset(N=500,tasks=2,dimensions=1,min_T=min_T,max_T=min_T+1,load_dataset=False,path_datasets='./datasets')



for pos, traj in enumerate(test_data[0]):
    if traj[3] > 10000:
        del test_data[0][pos]
        del test_check[0][pos]
    avg = sum(test_data[0][pos])/len(test_data[0][pos])
    sd = (sum([((x - avg) ** 2) for x in test_data[0][pos]]) / len(test_data[0][pos]))**0.5
    test_data[0][pos] = [(i-avg)/sd for i in test_data[0][pos]]
print('Trajectories generated and culled\n')



accurate = 0
yhat = model.predict(test_data[0],verbose=0)
for i in range(len(yhat)):
    if np.argmax(yhat[i]) == test_check[0][i]:
        accurate += 1
    
try:
    model.save('/home/alex/Desktop/KURF/Scripts/Models/Convolutional2')
except:
    model.save('./')

print('The model had a {}% accuracy'.format(100*accurate/len(yhat)))
print('The total time taken was {} seconds'.format(time.time()-start))







playsound('bad-to-the-bone.mp3')




