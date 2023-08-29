'''
Going to write some simple functions that will aid my code later, autism rocks
'''

from andi_datasets.datasets_challenge import challenge_theory_dataset
import pandas as pd
import matplotlib.pyplot as plt
import json
import numpy as np
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.saving import save_model
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

N = 15000
min_T = 320
size = 10

def generate_norm_data(N=1000,min_T=320):
    X1, Y1, X2, Y2, X3, Y3 = challenge_theory_dataset(N=N,tasks=2,dimensions=1,min_T=min_T,max_T=min_T+1)
    for pos, traj in enumerate(X2[0]):
        if traj[3] > 10000:
            del X2[0][pos]
            del Y2[0][pos]
            
        avg = sum(X2[0][pos])/len(X2[0][pos])
        sd = (sum([((x - avg) ** 2) for x in X2[0][pos]]) / len(X2[0][pos]))**0.5
        X2[0][pos] = [(i-avg)/sd for i in X2[0][pos]]
        
    alpha = []
    for pos, val in enumerate(Y2[0]):
        temp = [0,0,0,0,0]
        temp[int(val)] += 1
        alpha.append(temp)
    return [X2[0],alpha]

# data,alpha = generate_norm_data(1000,320)


'''

We have to make some sort of obscure code in order to define layers for the network so for example:
    1 = Conv
    2 = Pooling
    3 = Batch Norm
    4 = Dense
We can then run a script that generates all of the combinations, we can even do it through if statements like a monkey
Then we hook that code up into parallel somehow, last thing, why am I talking in the third person
'''

    

def Convolute(dataset,size):
    Convoluted_data = np.array([])
    for pos, traj in enumerate(dataset):
        Convoluted_data.append([])
        for conv in range(min_T-size):
            Convoluted_data = np.append(Convoluted_data[pos], traj[conv:conv+size])
            
            
            
            
            
def Generate_NN(layer_stack,ks):
    
    for stack in layer_stack:
        try:
            model = Sequential()
            for layer in stack:
                if layer == 1:
                    model.add(Conv1D(filters=16, kernel_size=ks, activation='relu', input_shape=(min_T-size, size)))
                elif layer ==2:
                    model.add(MaxPooling1D(pool_size=2))
                elif layer == 3:
                    model.add(BatchNormalization())
                elif layer == 4:
                    model.add(Dense(50))
        
            model.add(Dense(50))
            model.add(Dropout(0.25))
            model.add(Flatten())
            model.add(Dense(5,activation='softmax'))
            model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
            model.summary()
            
            model_name = [str(lay) for lay in stack]
            model_name.append('-{}'.format(ks))
            
            model.save('/home/alex/Desktop/KURF/Scripts/Models/All_The_Nets/{}'.format(''.join(model_name)))
            print(''.join(model_name))
        except:
            continue
        # return model

'''
Now I need to figure out how to generate the layer stack, ideally it should have a second function that defines the filters and the kernel size, or the pool size etc

Ok how do I do this, so I need to iterate over all the possible combinations of Convolutional networks, pooling layers and batch normalizations, so it would initially be:
    1st pass:
        Conv1-Flatten-Dense then run through kernel size 2-4-6-10-16-32
    2nd pass:
        Conv1-Pool-Norm-Conv1-Pool-Norm-Dense \
        Conv1-Norm-Pool-Conv1-Norm-Pool-Dense \ \
                                                    Run through 2-4-6-10-16-32
        Conv1-Norm-Pool-Conv1-Pool-Norm-Dense \ \
        Conv1-Pool-Norm-Conv1-Norm-Pool-Dense \

'''

def Generate_layer_stack():
    '''
    Really dirty code that works surprisingly well, biggest potential source of error is user error
    '''
    total = []
    stack1 = []
    for i in range(2):
        layer_stack = [1]
        layer_stack.append(int(2+0.5-0.5*(-1)**i))
        layer_stack.append(int(2+0.5+0.5*(-1)**i))
        stack1.append(layer_stack)
        total.append(layer_stack)
    stack2 = []
    for series in stack1:
            temp = [j for j in series]
            temp.append(1)
            temp.append(2)
            temp.append(3)
            stack2.append(temp)
            total.append(temp)
            temp = [j for j in series]
            temp.append(1)
            temp.append(3)
            temp.append(2)
            stack2.append(temp)
            total.append(temp)
    stack3 = []
    for series in stack2:
            temp = [j for j in series]
            temp.append(1)
            temp.append(2)
            temp.append(3)
            stack3.append(temp)
            total.append(temp)
            temp = [j for j in series]
            temp.append(1)
            temp.append(3)
            temp.append(2)
            stack3.append(temp)
            total.append(temp)
    stack4 = []
    for series in stack3:
            temp = [j for j in series]
            temp.append(1)
            temp.append(2)
            temp.append(3)
            stack4.append(temp)
            total.append(temp)
            temp = [j for j in series]
            temp.append(1)
            temp.append(3)
            temp.append(2)
            stack4.append(temp)
            total.append(temp)
            
            

  
    return total

stax = Generate_layer_stack()
kernel_size = [2,4,6,10,16,32]

for ks in kernel_size:
    Generate_NN(stax,ks)
    print(ks)
        