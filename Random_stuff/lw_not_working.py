import numpy as np

def lw(T, alpha):
            ''' Creates a 1D Levy walk trajectory '''             
            if alpha < 1:
                raise ValueError('Levy walks only allow for anomalous exponents > 1.')
            # Define exponents for the distribution of flight times                            
            if alpha == 2:
                sigma = np.random.rand()
            else:
                sigma = 3-alpha
            dt = (1-np.random.rand(T))**(-1/sigma)
            dt[dt > T] = T+1
            # Define the velocity
            v = 10*np.random.rand()                        
            # Generate the trajectory
            positions = np.empty(0)
            for t in dt:
                positions = np.append(positions, v*np.ones(int(t))*(2*np.random.randint(0,2)-1))
                if len(positions) > T:
                    break 
            return [np.cumsum(positions[:int(T)]) - positions[0],dt]

biggest = [0,0]
biggums = 0
alpha = np.linspace(1,2,100)
for alpha_val in alpha:
    for i in range(100):
        traj,dt =lw(184,alpha_val)
        xpos = []
        for itera,val in enumerate(traj):
            xpos.append(sum(traj[:itera]))
        if abs(xpos[-1]) > abs(biggest[0]):
            biggest[1] = i
            biggest[0] = xpos[-1]
            bigdt = dt
    print(biggest[0])
    if abs(biggest[0]) > abs(biggums):
        biggums = biggest[0]
        
print(biggums)
        
