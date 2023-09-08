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
# from playsound import playsound
import time
import Convtools as ct
import seaborn as sn
from andi_datasets.datasets_theory import datasets_theory

N = 20
min_T = 320
size = 2
start = time.time()




trajectories = []
alpha_vals = []
for i in range(size):
    traj_type = random.randint(0,4)
    print(traj_type)
    if traj_type==0 or traj_type==1 or traj_type==4:
        alpha = random.random()
    if traj_type==3:
        alpha = random.random() + 1        
    else:
        alpha = random.random()*2
    print(alpha)
    trajectories.append(datasets_theory().create_dataset(T=min_T,N_models=1,models=traj_type,exponents=[alpha])[0])
print(trajectories,alpha)

        
    
# # alpha = [[0]*i + [1] + [0]*(4-i) for i in range(5)]  Brilliantly condensed code which will never see the light of day again (literally just makes an I matrix)

# print(len(trajectories))

# print(len(trajectories[3]))


# for i in range(len(trajectories)):
    
#     for j in range(len(trajectories[i])):
#         trajectories[i],alpha = ct.normalize_data(trajectories[i],[i for x in range(len(trajectories))])
#         del trajectories[i][j][min_T]
#         del trajectories[i][j][min_T] # For some reason trajectories generate with extra numbers at the end, just clip them
        


# model = Sequential()
# model.add(Conv1D(filters=16, kernel_size=size, activation='relu', input_shape=(min_T,1)))
# model.add(BatchNormalization())
# model.add(MaxPooling1D(pool_size=4))
# model.add(Conv1D(filters=32, kernel_size=4, activation='relu'))#, input_shape=(304, 16)))
# model.add(BatchNormalization())
# model.add(MaxPooling1D(pool_size=2))
# model.add(Conv1D(filters=64, kernel_size=4, activation='relu'))#, input_shape=(304, 16)))
# model.add(BatchNormalization())
# model.add(MaxPooling1D(pool_size=2))
# model.add(Dense(50, activation='relu'))
# model.add(Dropout(0.25))
# model.add(Flatten())
# model.add(Dense(5,activation='softmax'))
# model.summary()
# model.compile(optimizer=Adam(learning_rate=0.001), loss='mse') # Random model tbh, not much thought put into it


# '''
# Fittin the model time, how does this work? Time to figure it out lord have mercy

# Okay so we have our trajectories = [ [ ], [ ], [ ], [ ], [ ] ]

# '''
# # model.fit(X2[0], alpha, epochs=30)




# for i in range(5):
#     for pos,trajectory in enumerate(trajectories[i]):
#         trajectory = np.array(trajectory)
#         trajectories[i][pos] = np.reshape(trajectory,(1,320,1))

# accurate = 0
# yhat = [[] for i in range(5)]
# models = ['attm', 'ctrw', 'fbm', 'lw', 'sbm']

# for i in range(5):
#     for j in range(len(trajectories[i])):
#         yhat[i].append(model.predict(trajectories[i][j],verbose=0)[0])
    
# # print(yhat)
# heatmap_list = [[0 for i in range(5)] for j in range(5)]

# for pos,traj_type in enumerate(yhat):
#     for traj in traj_type:
#         heatmap_list[pos][np.argmax(traj)] += 1
    

# sn.heatmap(heatmap_list,cmap='Blues',xticklabels=models,yticklabels=models)
# # print(yhat)

# try:
#     model.save('/home/alex/Desktop/KURF/Scripts/Models/Convolutional-DatasetTheory')
# except:
#     model.save('./')

# accurate = 0
# for i in range(5):
#     yhat = model.predict(trajectories[i],verbose=0)
#     for i in range(len(yhat)):
#         if np.argmax(yhat[i]) == i:
#             accurate += 1
    

# print('The model had a {}% accuracy'.format(100*accurate/len(yhat)))