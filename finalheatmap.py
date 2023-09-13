'''
Operation 'Divide and Conquer':
    First we must divide, I will send an infiltrator to access the code in the files, mainly the dataset_theory file, 
    he is our best man, and I trust he will gather the neccessary info, he will extract the functions that generate the 
    datasets and send them back to HQ.
    
    On our side, we will build a function out of their function that generates trajectories equally of each type, 
    this will allow us to calculate heatmaps and finally finish that damn report, onwards!
'''
'''
Commence phase 1: The infiltrator perpetrator
'''
from andi_datasets.datasets_challenge import challenge_theory_dataset
import matplotlib.pyplot as plt
import keras
import seaborn as sn
import numpy as np

# challenge_theory_dataset(N=10,tasks=2,dimensions=1,min_T=1,max_T='ooglyboogly',path_datasets='./datasets')
'''
Phase 1 completed, onto Phase 2: Generate and Collate
'''
N = 1000
min_T = 320





X1, Y1, trajectories, num_model, X3, Y3 = challenge_theory_dataset(N=N,tasks=2,dimensions=1,min_T=min_T,max_T=min_T+1,save_dataset=True,path_datasets='./datasets')
num_model = num_model[0]
trajectories = trajectories[0]

'''
Some trajectories look a bit off so Im gonna write a program to detect them
'''

for i in range(5):
    highest = 0
    poshigh = 0
    old = 0
    for pos, trajectory in enumerate(trajectories):
        traj = [abs(i) for i in trajectory]
        highest = sum(traj)
        if highest > old:
            old = highest
            poshigh = pos
    print('The biggest traj was at position {}, with model type {}'.format(poshigh,num_model[poshigh]))
    del trajectories[poshigh]
    del num_model[poshigh]



for pos, traj in enumerate(trajectories):
    if traj[3] > 10000:
        del trajectories[pos]
        del num_model[pos]
    avg = sum(trajectories[pos])/len(trajectories[pos])
    sd = (sum([((x - avg) ** 2) for x in trajectories[pos]]) / len(trajectories[pos]))**0.5
    trajectories[pos] = [(i-avg)/sd for i in trajectories[pos]]
    # plt.plot(trajectories[pos])




'''
What we need to do is generate double the needed trajectories, and then just group them into categories in order to do the heatmaps,
we also need to plot them to make sure theyre the right ones
'''
classified_models = [[],[],[],[],[]]
for position,trajectory in enumerate(trajectories):
    classified_models[int(num_model[position])].append(np.reshape(trajectory,(1,320,1)))
        
'''
Now we have a list of trajectories that are categorized into models
'''

model = keras.models.load_model('/home/alex/Desktop/KURF/Scripts/Models/Convolutional_Final')

    
accurate = 0
yhat = [[] for i in range(5)]
models = ['attm','sbm','ctrw','fbm','lw']

for i in range(5):
    for j in range(len(classified_models[i])):
        yhat[i].append(model.predict(classified_models[i][j],verbose=0)[0])
    
# print(yhat)
heatmap_list = [[0 for i in range(5)] for j in range(5)]
accurate = 0
for pos,model in enumerate(yhat):
    for traj in model:
        heatmap_list[pos][np.argmax(traj)] += 1
        if pos == np.argmax(traj):
            accurate += 1
    
print(accurate)
print(len(trajectories))
sn.heatmap(heatmap_list,cmap='Blues',xticklabels=models,yticklabels=models,annot=True,fmt='g')
# print(yhat)

plt.savefig('./heatmaplotsofvalues.png', dpi=300)

print('The model had a {}% accuracy'.format(100*accurate/len(trajectories)))















