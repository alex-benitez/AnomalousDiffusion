from andi_datasets.datasets_challenge import challenge_theory_dataset
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import numpy as np
import time
from playsound import playsound



traj = 1000
X2, Y1, X1, Y2, X3, Y3 = challenge_theory_dataset(N = traj,tasks=2,dimensions=[1],min_T=99,max_T=200)

'''
Now we need to have the X positions and the Y positions, this will be easy if we just:
'''



#ax = plt.axes(projection='3d')
biggest = 0

    
for i in range(traj):
    xpos = []
    current = X1[0][i]
    for itera,val in enumerate(X1[0][i]):
        xpos.append(sum(X1[0][i][:itera]))
    
    # ypos = []
    # for itera,val in enumerate(X1[1][i]):
    #     ypos.append(sum(X1[1][i][:itera]))
        
    # zpos = []
    # for itera,val in enumerate(X1[2][i]):
    #     zpos.append(sum(X1[2][i][:itera]))
        
        
    # lowest = min(len(xpos),len(ypos))#len(zpos)) 
    # xpos = xpos[:lowest]
    # ypos = ypos[:lowest]
    # zpos = zpos[:lowest]    
    #ax.plot3D(xpos,ypos)#,zpos)
    print(i)
    if abs(biggest) < abs(xpos[-1]):
        biggest = xpos[-1]
        bigtrajx = xpos
        # bigtrajy = ypos
        bigindex = i
    plt.plot(xpos)


print(bigtrajx,bigindex,len(bigtrajx))    
print(Y2[0][bigindex])

plt.savefig('spidah.pdf',dpi=300)
print(biggest)
    

















playsound("bad-to-the-bone.mp3")
