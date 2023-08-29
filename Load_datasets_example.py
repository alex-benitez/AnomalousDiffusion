'''
This is a little test to store and retrieve networks
'''

'''
Gonna have to do some magic here, just find a way to apply your convolutional network to these timeseries my guy,
lets start by thinking, what exactly is a CNN. You split it into different squares and each one is assigned to a neuron,
then afterwards you do that again with the remaining neurons, were gonna need a lot of padding for this one, I think
'''
from andi_datasets.datasets_challenge import challenge_theory_dataset

import numpy as np
from tensorflow import keras
import time
min_T = 320
start = time.time()
model = keras.models.load_model('/home/alex/Desktop/KURF/Scripts/Models/Convolutional')
size = 10
model.summary()
X1, Y1, test_data, test_check, X3, Y3 = challenge_theory_dataset(N=100,tasks=2,dimensions=1,min_T=min_T,max_T=min_T+1,load_dataset=False,path_datasets='./datasets')



for pos, traj in enumerate(test_data[0]):
    if traj[3] > 10000:
        del test_data[0][pos]
        del test_check[0][pos]
    avg = sum(test_data[0][pos])/len(test_data[0][pos])
    sd = (sum([((x - avg) ** 2) for x in test_data[0][pos]]) / len(test_data[0][pos]))**0.5
    test_data[0][pos] = [(i-avg)/sd for i in test_data[0][pos]]
print('Trajectories generated and culled\n')




Convoluted_test= []
'''
This would be higly parallelizable if you manage for it to generate the convolutional trajectories in blocks, trainign the actual model is harder to parallelize 
'''

for pos, traj in enumerate(test_data[0]):
    Convoluted_test.append([])
    for conv in range(min_T-size):
        Convoluted_test[pos].append(traj[conv:conv+size])
    



accurate = 0
yhat = model.predict(Convoluted_test,verbose=0)
for i in range(len(yhat)):
    if np.argmax(yhat[i]) == test_check[0][i]:
        accurate += 1
    
    
    
print('The model had a {}% accuracy'.format(100*accurate/len(yhat)))
print('The total time taken was {} seconds'.format(time.time()-start))









