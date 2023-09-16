'''
After a lot of dilly-dallying, it is finally time to bite the bullet and write the coolest piece of code you've written yet, it's gonna be an amazing clockwork-like system
if you're scrolling through my github and happen to see this, hire me, I promise I can be trusted with production code ;)
Before we start, a quick joke: 
    My ex wife still misses me, but her aim is getting better
'''

from andi_datasets.datasets_challenge import challenge_theory_dataset
import matplotlib.pyplot as plt
import numpy as np
import keras.models
from multiprocessing import Pool

def generate_trajectories(N=60000,min_T=320):
    '''
    Parameters
    ----------
    N : Number of trajectories.
    min_T : Size of the trajectories
    Returns
    -------
    Gives you exactly N trajectories, after deleting the weird ones, and normalizing and reshaping the data so it can be fed into a NN

    '''
    
    X1, Y1, X2, Y2, X3, Y3 = challenge_theory_dataset(N=N,tasks=2,dimensions=1,min_T=min_T,max_T=min_T+1)
    trajectories = X2[0]
    types = Y2[0]
    for pos,traj in enumerate(trajectories):

        if traj[4] > 1000: # The program that generates the trajectories sometimes has an overflow or some weird bug, and prints out trajectories where the particle moves at about 10e8 times the speed of light

            X1, Y1, temptraj, temppos, X3, Y3 = challenge_theory_dataset(N=1,tasks=2,dimensions=1,min_T=min_T,max_T=min_T+1)    
            trajectories[pos] = temptraj[0][0]
            types[pos] = temppos[0][0]
            print(temptraj[0])
            if trajectories[pos][4] > 1001:
                # If this happens, I don't know if you're really unlucky or you've angered the gods of fortune but honestly congratulations
                print('Millions to one')
                    
                X1, Y1, temptraj, temppos, X3, Y3 = challenge_theory_dataset(N=1,tasks=2,dimensions=1,min_T=min_T,max_T=min_T+1)    
                trajectories[pos] = temptraj[0][0]
                types[pos] = temppos[0][0]
                
                if trajectories[pos][4] > 1000:
                    # It would take my computer 775 years to generate enough trajectories to have a 50% chance of this happenning, so I think we're in the clear now
                    print('How did you manage')
                    X1, Y1, temptraj, temppos, X3, Y3 = challenge_theory_dataset(N=1,tasks=2,dimensions=1,min_T=min_T,max_T=min_T+1)    
                    trajectories[pos] = temptraj[0][0]
                    types[pos] = temppos[0][0]
                    
                    
        avg = sum(trajectories[pos])/len(trajectories[pos])
        sd = (sum([((x - avg) ** 2) for x in trajectories[pos]]) / len(trajectories[pos]))**0.5

        trajectories[pos] = [(i-avg)/sd for i in trajectories[pos]] # By subtracting the average and dividing by the standard deviation, the network learns faster
        trajectories[pos] = np.reshape(trajectories[pos],(1,min_T,1)) # Reshape the trajectories so they can be fed into the networks
    
    # Now we need to extract exactly 2000 of each type of trajectory to validate the networks
    
    validation_data = [[] for i in range(5)]
    
    total = 0
    counter = 0
    while counter < 10000:
        current_type = int(types[total])
        if len(validation_data[current_type]) < 2000:
            validation_data[current_type].append(trajectories.pop(current_type))
            del types[current_type]
            counter += 1
            total += 1
        else:
            total += 1
        
    
    return [trajectories,types,validation_data]
        

# a,b,c = generate_trajectories()
# print(len(a))
# print(len(b))
# for listo in c:
#     print(len(listo))
# for traj in a:
#     plt.plot(np.reshape(traj,(320)))



'''
Next step is to extract the networks I generated earlier and train them, since the function to generate is pretty clunky, I have a text file with all of the names for the networks, so I'll read from that file and train the networks
'''


textfile = open('./readme.txt','r')
list_of_nets = textfile.readlines()
for pos,element in enumerate(list_of_nets):
    list_of_nets[pos] = element[:-1]
print(list_of_nets,len(list_of_nets))
models = ['attm','sbm','ctrw','fbm','lw']

def train_the_network(network_id):

    model = keras.models.load_model('/home/alex/Desktop/KURF/Scripts/Models/All_The_Nets/{}'.format(network_id))
    trajectories,types,validation_data = generate_trajectories()
    print('Starting the training of network {}'.format(network_id))
    model.fit(trajectories, types, epochs=30)
    model.save('/home/alex/Desktop/KURF/Scripts/Models/All_The_Nets/{}'.format(network_id))
    accuracy = [0 for i in range(5)]
    
    for pos,listoftraj in enumerate(validation_data):
        for traj in listoftraj:
            prediction = model.predict(traj,verbose=0)[0]
            if int(prediction) == pos:
                accuracy[pos] += 1
    resultsfile = open('./Results/{}-results'.format(network_id),'w')
    resultsfile.write('Total accuracy: {}%'.format(round(sum(accuracy)/10000,2)))
    for pos,component in enumerate(accuracy):
        resultsfile.write(' {} accuracy: {}%'.format(models[pos],round(sum(component)/2000,2)))
    
    print('All done!')

if __name__ == "__main__":
    with Pool() as pool:
      result = pool.map(train_the_network, list_of_nets)
    print("Program finished!")

    
    



