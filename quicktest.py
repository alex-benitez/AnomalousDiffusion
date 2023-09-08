'''
Gonna have to do some magic here, just find a way to apply your convolutional network to these timeseries my guy,
lets start by thinking, what exactly is a CNN. You split it into different squares and each one is assigned to a neuron,
then afterwards you do that again with the remaining neurons, were gonna need a lot of padding for this one, I think
'''
from andi_datasets.datasets_challenge import challenge_theory_dataset
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import keras
import random
from playsound import playsound
import time
import Convtools as ct
import seaborn as sn
from andi_datasets.datasets_theory import datasets_theory

N = 100
min_T = 320
size = 10
start = time.time()

trajectories = [[] for i in range(5)]
for i in range(5):
    print('aaaaaaa')
    for big in range(N):
        if i==0 or i==1:
            exponent = 0.75
        else:
            exponent = 1.9
        trajectories[i].append(datasets_theory().create_dataset(T=min_T,N_models=1,models=i,exponents=[exponent])[0])
    
# alpha = [[0]*i + [1] + [0]*(4-i) for i in range(5)]  Brilliantly condensed code which will never see the light of day again

print(len(trajectories))

print(len(trajectories[3]))


for i in range(len(trajectories)):
    
    for j in range(len(trajectories[i])):
        trajectories[i],alpha = ct.normalize_data(trajectories[i],[i for x in range(len(trajectories))])
        del trajectories[i][j][320]
        del trajectories[i][j][320]
        plt.plot(trajectories[i][j])

    
model = keras.models.load_model('/home/alex/Desktop/KURF/Scripts/Models/Convolutional_Final')
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




