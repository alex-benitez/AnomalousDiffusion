'''
This file is supposed to be for generation of data, in order to later train different types of networks,
as such it should be easily adjustable
'''
import numpy as np
from andi_datasets.datasets_challenge import challenge_theory_dataset
from playsound import playsound


 # Variable declaration:
     
No_Trajectories = 500000

tasks = 2     
dimensions = [1]

Smallest_Traject = 400
Maximum_Traject = 550


save_data = True   
dataset_path = './datasets' 

     
 # Main function for generating datasets 
X1, Y1, X2, Y2, X3, Y3 = challenge_theory_dataset(N = No_Trajectories, max_T=Maximum_Traject, min_T = Smallest_Traject, dimensions= dimensions, tasks=tasks,save_dataset=save_data,path_datasets=dataset_path)
playsound('bad-to-the-bone.mp3')
