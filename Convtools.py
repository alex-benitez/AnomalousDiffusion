'''
Packing everything into functions for later use
'''

def create_T2dataset(N,min_T,save_location=False):
    '''
    Generates trajectories in 1D for the second task
    '''
    from andi_datasets.datasets_challenge import challenge_theory_dataset
    
    if save_location != False:
        X1, Y1, actual_data, actual_type, X3, Y3 = challenge_theory_dataset(N=100,tasks=2,dimensions=1,min_T=min_T,max_T=min_T+1,save_trajectories=save_location)
    else:
        X1, Y1, actual_data, actual_type, X3, Y3 = challenge_theory_dataset(N=100,tasks=2,dimensions=1,min_T=min_T,max_T=min_T+1)

    return [actual_data[0], actual_type[0]]


def normalize_data(dataset,alpha,plot=False):
    
    for pos, traj in enumerate(dataset):
        if traj[3] > 10000:
            del dataset[pos]
            del alpha[pos]
            
        avg = sum(dataset[pos])/len(dataset[pos])
        sd = (sum([((x - avg) ** 2) for x in dataset[pos]]) / len(dataset[pos]))**0.5
        dataset[pos] = [(i-avg)/sd for i in dataset[pos]]
        if plot:
            import matplotlib.pyplot as plt
            plt.plot(dataset[pos])
        

    alpha_binarized = []

    for pos, val in enumerate(alpha):
        temp = [0,0,0,0,0]
        temp[int(val)] += 1
        alpha_binarized.append(temp)

            
    return [dataset,alpha_binarized]

def test_model(model,number_of_tests,min_T=320):
    import numpy.argmax as arg
    from andi_datasets.datasets_challenge import challenge_theory_dataset
    X1, Y1, test_data, test_check, X3, Y3 = challenge_theory_dataset(N=number_of_tests,tasks=2,dimensions=1,min_T=min_T,max_T=min_T+1)
    test_data,test_check = normalize_data(test_data,test_check)
    accurate = 0
    yhat = model.predict(test_data,verbose=0)
    for i in range(len(yhat)):
        if arg(yhat[i]) == test_check[0][i]:
            accurate += 1
        
    try:
        model.save('/home/alex/Desktop/KURF/Scripts/Models/Convtools')
    except:
        model.save('./')

    print('The model had a {}% accuracy'.format(100*accurate/len(yhat)))
    


