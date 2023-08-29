'''
Going to all of a sudden do a 180 and use a completely different library, this is the testing area
'''


import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras as keras
import innvestigate as inn
from andi_datasets.datasets_challenge import challenge_theory_dataset
import numpy as np
import os
import seaborn as sn

tf.compat.v1.disable_eager_execution()

min_T = 320

model = keras.models.load_model('/home/alex/Desktop/KURF/Scripts/Models/Convolutional_Final')



X1, Y1, test_data, test_check, X3, Y3 = challenge_theory_dataset(N=1,tasks=2,dimensions=1,min_T=min_T,max_T=min_T+1,load_dataset=False,path_datasets='./datasets')

library = test_data[0][0]



for pos, traj in enumerate(test_data[0]):
    if traj[3] > 10000:
        del test_data[0][pos]
        del test_check[0][pos]
    avg = sum(test_data[0][pos])/len(test_data[0][pos])
    sd = (sum([((x - avg) ** 2) for x in test_data[0][pos]]) / len(test_data[0][pos]))**0.5
    test_data[0][pos] = [(i-avg)/sd for i in test_data[0][pos]]
print('Trajectories generated and culled\n')



data = np.array(test_data[0])
data = np.resize(data,(1,320,1)) # Resize the data otherwise it wont work, nothign changes in terms of the structure though
accurate = 0
yhat = model.predict(data,verbose=0)
for i in range(len(yhat)):
    if np.argmax(yhat[i]) == test_check[0][i]:
        accurate += 1
    

print('The model had a {}% accuracy'.format(100*accurate/len(yhat)))

# Stripping the softmax activation from the model
model_wo_sm = inn.model_wo_softmax(model)

# Creating an analyzer
analyzer = inn.create_analyzer(
    "lrp.epsilon", model_wo_sm, neuron_selection_mode="max_activation", **{"epsilon": 1}
)


# gradient_analyzer = inn.create_analyzer("gradient", model)

# Applying the analyzer
analysis = analyzer.analyze(data)
# Applying the analyzer



'''
My objective now is to find a way to make the plot look good, how about plotting only the most important 
points with respect to the original trajectory, what Ill do is exactly that with subplots, I should probably also 
plot the timesteps as discrete jumps of 1

First Im going to take the displacements and turn them into a graph of position, additionally to make it prettier, Im going to
also add x-values for the displacement and make them equal to 1, therefore later its easier

Watch out for the fact that the initial value will be 0 in order to haev rpettier graphs, so all of the values later on will be useless
'''

# Going to add the function that turns the displacement into position, remember the starting 0

position = [0]
for pos,delta in enumerate(library):
    position.append(position[pos]+delta)
    
    
x_values = np.arange(0,len(position))

# Now that the values are generated, we have to find which are the n bigggest values out of analysis

temp_list = [[0,0] for i in range(25)]
sig_val = 25
check = 1000

for i in range(25):
    biggest = 0
    bigpos = 0
    for pos,heat in enumerate(analysis[0]):
        if biggest < heat and heat < check:
            biggest = heat
            bigpos = pos
            
    check = biggest
        
    temp_list[i][0] = float(biggest)
    temp_list[i][1] = bigpos

print('temp_list')
print(temp_list)

# After finding the 25 values and their indices, we need to find the corresponding parts of the trajectory and 
# figure out a smart way to plot these parts

fig, axs = plt.subplots(2,1,sharex=True)
# plt.plot(position)
# axs[0].plot(position)


axs[0].plot(position,lw=0.5)
axs[1].plot(library,lw=0.5)


for i in range(sig_val):
    pos = temp_list[i][1]
    if pos == len(position):
        curr_val = [position[pos],position[pos+1]]
        x_val = [pos,pos+1]
    elif pos == 0:
        curr_val = [position[pos+1],position[pos+2]]
        x_val = [pos+1,pos+2]
    else:
        curr_val = [position[pos],position[pos+1]]
        x_val = [pos,pos+1]
    
    axs[0].plot(x_val,curr_val,c='red')
axs[0].set_ylim([min(position),max(position)])


for i in range(sig_val):
    pos = temp_list[i][1]
    if pos == len(library):
        curr_val = [library[pos-1],library[pos]]
        x_val = [pos-1,pos]
    elif pos == 0:
        curr_val = [library[pos],library[pos+1]]
        x_val = [pos,pos+1]
    else:
        curr_val = [library[pos-1],library[pos]]
        x_val = [pos-1,pos]
    
    color = ((sig_val-i)/sig_val,0,0)
    axs[1].plot(x_val,curr_val,c=color,alpha=(sig_val-i)/sig_val)

axs[1].set_ylim([min(library),max(library)])

plt.subplots_adjust(hspace=0.5)
axs[0].title.set_text('Position of the particle over time')
axs[1].title.set_text('Displacement of the particle over time')


files = os.listdir('/home/alex/Desktop/KURF/Scripts/Plots/')
filename= 0
for i in files:
    if int(i[0]) > filename:
        filename = int(i[0])
        print(filename)
    

plt.savefig('/home/alex/Desktop/KURF/Scripts/Plots/{}.png'.format(filename+1),dpi=300)
print('\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n DONE \n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n')



















