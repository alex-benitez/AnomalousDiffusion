'''
This is a test for the heatmaps, lets get this motherfucking bread

'''
import matplotlib.pyplot as plt
import numpy
import time
import numpy as np
import model_io
import data_io
import render
from tensorflow import keras
from andi_datasets.datasets_challenge import challenge_theory_dataset

'''
Have to go fetch a network, then run the program and check the heatmap, good luck man
'''
min_T = 320
nn = keras.models.load_model('/home/alex/Desktop/KURF/Scripts/Models/Convolutional') 
X1, Y1, test_data, test_check, X3, Y3 = challenge_theory_dataset(N=100,tasks=2,dimensions=1,min_T=min_T,max_T=min_T+1,load_dataset=False,path_datasets='./datasets')


for pos, traj in enumerate(test_data[0]):
    if traj[3] > 10000:
        del test_data[0][pos]
        del test_check[0][pos]
    avg = sum(test_data[0][pos])/len(test_data[0][pos])
    sd = (sum([((x - avg) ** 2) for x in test_data[0][pos]]) / len(test_data[0][pos]))**0.5
    test_data[0][pos] = [(i-avg)/sd for i in test_data[0][pos]]


X = test_data[0]
Y = test_check[0]
temp = [[0,0,0,0,0] for i in range(Y.size[0])]
for traj,pos in enumerate(Y):
    temp[pos][traj] += 1

Y = temp
print(Y)

for i in I[:10]:
    x = X[i:i+1,...]

    #forward pass and prediction
    ypred = nn.forward(x)
    print('True Class:     ', np.argmax(Y[i]))
    print('Predicted Class:', np.argmax(ypred),'\n')

    #prepare initial relevance to reflect the model's dominant prediction (ie depopulate non-dominant output neurons)
    mask = np.zeros_like(ypred)
    mask[:,np.argmax(ypred)] = 1
    Rinit = ypred*mask


    #compute first layer relevance according to prediction
    #R = nn.lrp(Rinit)                   #as Eq(56) from DOI: 10.1371/journal.pone.0130140
    R = nn.lrp(Rinit,'epsilon',1.)    #as Eq(58) from DOI: 10.1371/journal.pone.0130140
    #R = nn.lrp(Rinit,'alphabeta',2)    #as Eq(60) from DOI: 10.1371/journal.pone.0130140

    #R = nn.lrp(ypred*Y[na,i],'epsilon',1.) #compute first layer relevance according to the true class label


    '''
    
    #compute first layer relvance for an arbitrarily selected class
    for yselect in range(10):
        yselect = (np.arange(Y.shape[1])[na,:] == yselect)*1.
        R = nn.lrp(ypred*yselect,'epsilon',0.1)
    '''

    '''
    # you may also specify different decompositions for each layer, e.g. as below:
    # first, set all layers (by calling set_lrp_parameters on the container module
    # of class Sequential) to perform alpha-beta decomposition with alpha = 1.
    # this causes the resulting relevance map to display excitation potential for the prediction
    #
    nn.set_lrp_parameters('alpha',1.)
    #
    # set the first layer (a convolutional layer) decomposition variant to 'w^2'. This may be especially
    # usefill if input values are ranged [0 V], with 0 being a frequent occurrence, but one still wishes to know about
    # the relevance feedback propagated to the pixels below the filter
    # the result with display relevance in important areas despite zero input activation energy.
    #
    nn.modules[0].set_lrp_parameters('ww') # also try 'flat'
    # compute the relevance map
    R = nn.lrp(Rinit)
    '''



    #sum over the third (color channel) axis. not necessary here, but for color images it would be.
    R = R.sum(axis=3)
    #same for input. create brightness image in [0,1].
    xs = ((x+1.)/2.).sum(axis=3)

    if not np == numpy: # np=cupy
        xs = np.asnumpy(xs)
        R = np.asnumpy(R)

    #render input and heatmap as rgb images
    digit = render.digit_to_rgb(xs, scaling = 3)
    hm = render.hm_to_rgb(R, X = xs, scaling = 3, sigma = 2)
    digit_hm = render.save_image([digit,hm],'../heatmap.png')
    data_io.write(R,'../heatmap.npy')

    #display the image as written to file
    plt.imshow(digit_hm, interpolation = 'none')
    plt.axis('off')
    plt.show()