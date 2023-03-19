import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import pprint as pp 
import pickle

# Setup / load data as an np array
# data = pd.read_csv('./data/train.csv')
data = pd.read_csv('./data/test.csv')
data = np.array(data)
 
# Get data shape
data_rows, number_cols = data.shape

# shuffle data
np.random.shuffle(data)


data_test = data.T
data_test_labels = data_test[0] # first row is now the data label
data_test_inputs = data_test[1:number_cols]
data_test_inputs = data_test_inputs / 255 # normalize to 0-1 


# setup a smaller dev set, 1000 rows should do it.
# we will also transposne the data, currently each row is one record, first column is label, 2-number_cols is a pixel per col. We want each input as a column 
# data_dev = data[0:1000].T
# data_dev_labels = data_dev[0] # first row is now the data label
# data_dev_inputs = data_dev[1:number_cols]
# data_dev_inputs = data_dev_inputs / 255 # normalize to 0-1

# # training set, same steps as above
# data_train = data[1000:data_rows].T
# data_train_labels = data_train[0] # first row is now the data label
# data_train_inputs = data_train[1:number_cols]
# data_train_inputs = data_train_inputs / 255 # normalize to 0-1 

# # Get number of training rows
# _,training_rows = data_train_inputs.shape


def init_params(inputs,hiddens,outputs):
    # inputs,hiddens,outputs
    weights ={}

    # quick hack to take array hiddens 0 (future state will support multiple hiddens)
    weights['hidden_count'] = len(hiddens)
    
    for hidden_index, hidden in enumerate(hiddens):
        
        node_int = str(hidden_index+1)

        if hidden_index == 0:
            weights['I_H'+node_int+'_W'] = np.random.rand(hidden,inputs) - 0.5
            weights['I_H'+node_int+'_b'] = np.random.rand(hidden,1) - 0.5
        if hidden_index == len(hiddens)-1:
            weights['H'+node_int+'_O_W'] = np.random.rand(outputs,hidden) - 0.5
            weights['H'+node_int+'_O_b'] = np.random.rand(outputs,1) - 0.5
        else:
            next_node = str(hidden_index+2)
            hidden2 = hiddens[hidden_index+1]
            weights['H'+node_int+'_H'+next_node+'_W'] = np.random.rand(hidden2,hidden) - 0.5
            weights['H'+node_int+'_H'+next_node+'_b'] = np.random.rand(hidden2,1) - 0.5

    
    # W1 = np.random.rand(10,784) - 0.5
    
    # b1 = np.random.rand(10,1)- 0.5
    # W2 = np.random.rand(10,10)- 0.5
    # b2 = np.random.rand(10,1)- 0.5

    return weights
    
def ReLU(Z):
    # need to understand this math more. Why can't we return negative numbers?
    return np.maximum(Z,0)

def softmax(Z):
    return np.exp(Z) / sum(np.exp(Z))

def forward_prop(weights, inputs):
    
    aWeights = {}

    hiddenLayers = weights['hidden_count']



    # Input to first hidden // ReLU - should support at least sigmoid
    # dot product of input layer to hidden layer weights + bias
    aWeights['I_H1_W'] = weights['I_H1_W'].dot(inputs) + weights['I_H1_b']
    # ReLU of dot product (returns positive number, or 0)
    aWeights['A_I_H1_W'] = ReLU(aWeights['I_H1_W'])
    
    
    #last Hidden to output
    if hiddenLayers >1:
        for i in range(1,hiddenLayers):
            # Last Layer
            nextI = i +1
            prevI = i -1
            if i == 1:
                aWeights['H{}_H{}_W'.format(i,nextI)] = weights['H{}_H{}_W'.format(i,nextI)].dot(aWeights['A_I_H{}_W'.format(i)])+weights['H{}_H{}_b'.format(i,nextI)]
                aWeights['A_H{}_H{}_W'.format(i,nextI)] = ReLU(aWeights['H{}_H{}_W'.format(i,nextI)])
            else:
                
                aWeights['H{}_H{}_W'.format(i,nextI)] = weights['H{}_H{}_W'.format(i,nextI)].dot(aWeights['A_H{}_H{}_W'.format(prevI,i)])+weights['H{}_H{}_b'.format(i,nextI)]
                aWeights['A_H{}_H{}_W'.format(i,nextI)] = ReLU(aWeights['H{}_H{}_W'.format(i,nextI)])

        prevI = hiddenLayers -1
        
        aWeights['H{}_O_W'.format(hiddenLayers)]= weights['H{}_O_W'.format(hiddenLayers)].dot(aWeights['A_H{}_H{}_W'.format(prevI,hiddenLayers)]) + weights['H{}_O_b'.format(hiddenLayers)]
        # softmax those (to ensure sum of all outputs = 1)
        aWeights['OUTPUT'] = softmax(aWeights['H{}_O_W'.format(hiddenLayers)])           


    else:
        # refactor later, but if there is only one layer format is slightly different.
        aWeights['H1_O_W']= weights['H1_O_W'].dot(aWeights['A_I_H1_W']) + weights['H1_O_b']
        # softmax those (to ensure sum of all outputs = 1)
        aWeights['OUTPUT'] = softmax(aWeights['H1_O_W'])
    
    


    return aWeights

def one_hot(labels):
    # specific to this data set, return an array of label output with 0's except for position of label set to 1.
    one_h = np.zeros((labels.size,labels.max() +1))
    one_h[np.arange(labels.size),labels] =1
    return one_h.T

def ReLU_deriv(Z):
    return Z > 0

def backward_prop(aWeights, weights, inputs,labels):

    dWeights = {}
    hiddenLayers = weights['hidden_count']


    # Calculate error delta from output. 
    one_hot_labels = one_hot(labels)
    # dZ2 = aWeights['OUTPUT'] - one_hot_labels
    outputErrors = aWeights['OUTPUT'] - one_hot_labels
    # work backwards 
    if hiddenLayers > 1:
        
        for i in range(hiddenLayers,0,-1):
            # print(i)
            # print(dWeights.keys())
            if i == hiddenLayers:
                prevI = i-1
                # Output to Last hidden 
                dWeights['d_H{}_O_W'.format(i)] = 1 / data_rows * outputErrors.dot(aWeights['A_H{}_H{}_W'.format(prevI,i)].T)
                dWeights['d_H{}_O_b'.format(i)] = 1 / data_rows * np.sum(outputErrors)
                nextErrors = weights['H{}_O_W'.format(i)].T.dot(outputErrors)  * ReLU_deriv(aWeights['A_H{}_H{}_W'.format(prevI,i)])
            else:
                prevI = i-1
                nextI = i+1
                
                # Hidden to Previous Hidden
                

                dWeights['d_H{}_H{}_W'.format(i,nextI)] = 1 / data_rows * nextErrors.dot(aWeights['A_H{}_H{}_W'.format(i,nextI)].T)
                dWeights['d_H{}_H{}_b'.format(i,nextI)] = 1 / data_rows * np.sum(nextErrors)
                if i >1:
                    nextErrors = weights['H{}_H{}_W'.format(i,nextI)].T.dot(nextErrors)  * ReLU_deriv(aWeights['A_H{}_H{}_W'.format(prevI,i)])
                else:
                    nextErrors = weights['H{}_H{}_W'.format(i,nextI)].T.dot(nextErrors)  * ReLU_deriv(aWeights['A_I_H{}_W'.format(i)])


        dWeights['d_I_H1_W'] = 1 / data_rows * nextErrors.dot(inputs.T)
        dWeights['d_I_H1_b'] = 1 / data_rows * np.sum(nextErrors)


    else:
        dWeights['d_H1_O_W'] = 1 / data_rows * outputErrors.dot(aWeights['A_I_H1_W'].T)
        dWeights['d_H1_O_b'] = 1 / data_rows * np.sum(outputErrors)
        dZ1 = weights['H1_O_W'].T.dot(outputErrors) * ReLU_deriv(aWeights['I_H1_W'])
        dWeights['d_I_H1_W'] = 1 / data_rows * dZ1.dot(inputs.T)
        dWeights['d_I_H1_b'] = 1 / data_rows * np.sum(dZ1)   
    
    return dWeights




    
def update_params(weights, dWeights, training_rate):

    for x,k in enumerate(weights):
        
        if k != 'hidden_count': 
            weights[k] = weights[k] - training_rate *dWeights['d_'+k]

    return weights

def get_predictions(predictions):
    return np.argmax(predictions,0)


def make_predictions(weights, inputs):
    aWeights = forward_prop(weights, inputs)
    return (get_predictions(aWeights['OUTPUT']))

def get_accuracy(predictions, labels):
    return np.sum(predictions == labels) / labels.size



def train_epoch(weights,inputs,labels,training_rate,epochs):

    data = {'x':[],'y':[]}
    for i in range(epochs):
        aWeights = forward_prop(weights, inputs)
        dWeights = backward_prop(aWeights, weights, inputs,labels)
        weights =update_params(weights, dWeights, training_rate)
        if i % 100 ==0:
            training_rate -= .0001
            training_rate = round(training_rate,4)
            if training_rate < .001:
                training_rate = .001
            predictions = get_predictions(aWeights['OUTPUT'])
            accuracy = get_accuracy(predictions,labels)    
            data['y'].append(accuracy)
            data['x'].append(i)
            print("Iteration:" , i, " Accuracy: ",accuracy, " New Training Rate: ", training_rate)
        if i % 1000 ==0:
            pickle_data(weights,'trained_weights/weights{}.pkl'.format(i))

    return weights,data

def test_prediction(index,weights):
    # current_image = data_dev_inputs[:,index,None]
    # prediction = make_predictions(weights,current_image)[0]
    # label = data_dev_labels[index]
    current_image = data_test_inputs[:,index,None]
    prediction = make_predictions(weights,current_image)[0]
    label = data_test_labels[index]

    print("Prediction: ",prediction, " Label: ",label)

    current_image = current_image.reshape((28,28)) * 255
    plt.gray()
    plt.imshow(current_image,interpolation= 'nearest')
    plt.show()


def display_grid(rows,cols):
    fig = plt.figure()
    plt.tight_layout()
    i = rows * cols
    for index in range(i):
        current_image = data_dev_inputs[:,index,None]
        current_image = current_image.reshape((28,28)) * 255
        pp.pprint(current_image)
        fig.add_subplot(rows,cols,index+1)
        plt.gray()
        # plt.set_cmap('blue')
        plt.axis('off')
        plt.imshow(current_image,interpolation= 'nearest')
    plt.show()

def graph_data(data):
    x_axis = data['x']
    y_axis = data['y']
    plt.plot(x_axis, y_axis)
    plt.title('title name')
    plt.xlabel('x_axis name')
    plt.ylabel('y_axis name')
    plt.show()

def pickle_data(obj,filename):
    with open(filename, 'wb') as outp:  # Overwrites any existing file.
        pickle.dump(obj, outp, pickle.HIGHEST_PROTOCOL)

def load_pickle(filename):
    with open(filename, 'rb') as inp:
        obj = pickle.load(inp)
    return obj
if __name__ == '__main__':
    
    # weights = init_params(784,[25],10)
    weights = load_pickle('trained_weights/weightsfinal.pkl')


    # weights, data =train_epoch(weights,data_train_inputs,data_train_labels,.5,100000)

    # its = 100    
    # init_training_weight = .5

    # display_grid(1,1)
    # for i in range(its):
    #     print(i)
        
    #     training_weight = init_training_weight*(1-(i*.01))
    #     if training_weight < .01:
    #        training_weight = .01
    #     round(training_weight,3)
    #     weights, data =train_epoch(weights,data_train_inputs,data_train_labels,training_weight,1000)
    # pickle_data(data,'graph_data.pkl')
    # pickle_data(weights,'trained_weights/weights{}.pkl'.format('final'))

    # graph_data(data)
    # predictions = make_predictions(weights,data_dev_inputs)
    # print(get_accuracy(predictions,data_dev_labels))
    predictions = make_predictions(weights,data_test_inputs)
    print(get_accuracy(predictions,data_test_labels))

    # test_prediction(1,weights)

    # weights = init_params(784,[10,10,10],10)
    # # print(weights.keys())

    # aWeights = forward_prop(weights,data_train_inputs)
    # # print(aWeights.keys())

    # dWeights = backward_prop(aWeights,weights,data_train_inputs,data_train_labels)
    # # print(dWeights.keys())

    # weights =update_params(weights, dWeights, .5)
    # # print(weights)
