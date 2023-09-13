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
from playsound import playsound

tf.compat.v1.disable_eager_execution()

min_T = 320

model = keras.models.load_model('/home/alex/Desktop/KURF/Scripts/Models/Convolutional_Final')



X1, Y1, test_data, test_check, X3, Y3 = challenge_theory_dataset(N=100,tasks=2,dimensions=1,min_T=min_T,max_T=min_T+1,load_dataset=False,path_datasets='./datasets')


for pos, traj in enumerate(test_data[0]):
    if traj[3] > 10000:
        del test_data[0][pos]
        del test_check[0][pos]
    avg = sum(test_data[0][pos])/len(test_data[0][pos])
    sd = (sum([((x - avg) ** 2) for x in test_data[0][pos]]) / len(test_data[0][pos]))**0.5
    test_data[0][pos] = [(i-avg)/sd for i in test_data[0][pos]]
print('Trajectories generated and culled\n')
library = test_data[0]
traj_type = test_check[0]


Plottable = [[] for i in range(5)]
forlater = [0 for i in range(5)]

for pos,trajectory in enumerate(library):
    if Plottable[int(traj_type[pos])] == []:
        Plottable[int(traj_type[pos])] = np.resize(trajectory,(1,320,1))
        forlater[int(traj_type[pos])] = pos
accurate = 0


for i in range(5):
    yhat = model.predict(Plottable[i],verbose=0)
    print(yhat)
    if np.argmax(yhat) == i:
        accurate += 1
    

print('The model had a {}% accuracy'.format(accurate*100/5))

# Stripping the softmax activation from the model
model_wo_sm = inn.model_wo_softmax(model)

# Creating an analyzer
analyzer = inn.create_analyzer(
    "lrp.epsilon", model_wo_sm, neuron_selection_mode="max_activation", **{"epsilon": 1}
)


# gradient_analyzer = inn.create_analyzer("gradient", model)

# # Applying the analyzer
analysis = []
for i in range(5):
    analysis.append(analyzer.analyze(Plottable[i])[0])
# Applying the analyzer



'''
My objective now is to find a way to make the plot look good, how about plotting only the most important 
points with respect to the original trajectory, what Ill do is exactly that with subplots, I should probably also 
plot the timesteps as discrete jumps of 1

First Im going to take the displacements and turn them into a graph of position, additionally to make it prettier, Im going to
also add x-values for the displacement and make them equal to 1, therefore later its easier

Watch out for the fact that the initial value will be 0 in order to haev rpettier graphs, so all of the values later on will be useless
'''


trajectories = [library[i] for i in forlater]
# Going to add the function that turns the displacement into position, remember the starting 0
fig, axs = plt.subplots(5,1,sharex=True)
fig.tight_layout(pad=1)
# plt.figure(figsize=(7,12))
x_values = np.arange(0,min_T)
models = ['attm','sbm','ctrw','fbm','lw']
for i in range(5):
    axs[i].plot(x_values,trajectories[i],lw=0.5)
    axs[i].set_title('Relevant parts of trajectories described by {}'.format(models[i]))
    



short = analysis[0]

for model in range(5):
    big_list = [10000]
    bigpos = [0]
    biggest = 0
    position = 0
    for iteration in range(25):
        for pos,significant in enumerate(analysis[model]):
            if significant > biggest and significant<big_list[-1]:
                biggest = significant
                position = pos
        big_list.append(biggest)
        bigpos.append(position)
        biggest = 0
    for i in range(25):
        pos = int(bigpos[i+1])

        if pos == len(trajectories[model]):
            curr_val = [trajectories[model][pos-1],trajectories[model][pos]]
            x_val = [pos-1,pos]
        else:
            curr_val = [trajectories[model][pos],trajectories[model][pos+1]]
            x_val = [pos,pos+1]
        
        axs[model].plot(x_val,curr_val,c='red')
    

plt.savefig('/home/alex/Desktop/KURF/Scripts/Plots/Fivegraphs2.png',dpi=300)    


# # Now that the values are generated, we have to find which are the n bigggest values out of analysis

# temp_list = [[0,0] for i in range(25)]
# sig_val = 25
# check = 1000

# for i in range(25):
#     biggest = 0
#     bigpos = 0
#     for pos,heat in enumerate(analysis[0]):
#         if biggest < heat and heat < check:
#             biggest = heat
#             bigpos = pos
            
#     check = biggest
        
#     temp_list[i][0] = float(biggest)
#     temp_list[i][1] = bigpos



# # After finding the 25 values and their indices, we need to find the corresponding parts of the trajectory and 
# # figure out a smart way to plot these parts

# # plt.plot(position)
# # axs[0].plot(position)


# axs[0].plot(position,lw=0.5)
# axs[1].plot(library,lw=0.5)


# for i in range(sig_val):
#     pos = temp_list[i][1]
#     if pos == len(position):
#         curr_val = [position[pos],position[pos+1]]
#         x_val = [pos,pos+1]
#     elif pos == 0:
#         curr_val = [position[pos+1],position[pos+2]]
#         x_val = [pos+1,pos+2]
#     else:
#         curr_val = [position[pos],position[pos+1]]
#         x_val = [pos,pos+1]
    
#     axs[0].plot(x_val,curr_val,c='red')
# axs[0].set_ylim([min(position),max(position)])


# for i in range(sig_val):
#     pos = temp_list[i][1]
#     if pos == len(library):
#         curr_val = [library[pos-1],library[pos]]
#         x_val = [pos-1,pos]
#     elif pos == 0:
#         curr_val = [library[pos],library[pos+1]]
#         x_val = [pos,pos+1]
#     else:
#         curr_val = [library[pos-1],library[pos]]
#         x_val = [pos-1,pos]
    
#     color = ((sig_val-i)/sig_val,0,0)
#     axs[1].plot(x_val,curr_val,c=color,alpha=(sig_val-i)/sig_val)

# axs[1].set_ylim([min(library),max(library)])

# plt.subplots_adjust(hspace=0.5)
# axs[0].title.set_text('Position of the particle over time')
# axs[1].title.set_text('Displacement of the particle over time')


# files = os.listdir('/home/alex/Desktop/KURF/Scripts/Plots/')
# filename= 0
# for i in files:
#     if int(i[0]) > filename:
#         filename = int(i[0])
#         print(filename)
    
# '''
# Going to cannibalize this function in order to make it into a subplot generating machine
# Lets do it 
# '''


# plt.savefig('/home/alex/Desktop/KURF/Scripts/Plots/{}.png'.format(filename+1),dpi=300)
playsound('bad-to-the-bone.mp3')




















