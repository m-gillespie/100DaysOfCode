import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

# Setup / load data as an np array
data = pd.read_csv('./data/train.csv')
data = np.array(data)

# Get data shape
data_rows, number_cols = data.shape

# shuffle data
np.random.shuffle(data)

# setup a smaller dev set, 1000 rows should do it.
# we will also transposne the data, currently each row is one record, first column is label, 2-number_cols is a pixel per col. We want each input as a column 
data_dev = data[0:1000].T
data_dev_labels = data_dev[0] # first row is now the data label
data_dev_inputs = data_dev[1:number_cols]
data_dev_inputs = data_dev_inputs / 255 # normalize to 0-1

# training set, same steps as above
data_train = data[1000:data_rows].T
data_train_labels = data_train[0] # first row is now the data label
data_train_inputs = data_train[1:number_cols]
data_train_inputs = data_train_inputs / 255 # normalize to 0-1

# Get number of training rows
_,training_rows = data_train_inputs.shape


def init_params(inputs,hiddens,outputs):
    # inputs,hiddens,outputs


    # quick hack to take array hiddens 0 (future state will support multiple hiddens)
    hiddens = hiddens[0]

    weights = {
        'I_H1_W': np.random.rand(hiddens,inputs) - 0.5,
        'I_H1_b': np.random.rand(hiddens,1) - 0.5,
        'H1_O_W': np.random.rand(outputs,hiddens) - 0.5,
        'H1_O_b' : np.random.rand(outputs,1) - 0.5

    }
    
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
    # dot product of input layer to hidden layer weights + bias
    aWeights['I_H1_W'] = weights['I_H1_W'].dot(inputs) + weights['I_H1_b']
    # ReLU of dot product (returns positive number, or 0)
    aWeights['A_I_H1_W'] = ReLU(aWeights['I_H1_W'])
    # Resulting A1 weights dot product output weights
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
    one_hot_labels = one_hot(labels)
    dZ2 = aWeights['OUTPUT'] - one_hot_labels
    dWeights['d_H1_O_W'] = 1 / data_rows * dZ2.dot(aWeights['A_I_H1_W'].T)
    dWeights['d_H1_O_b'] = 1 / data_rows * np.sum(dZ2)
    dZ1 = weights['H1_O_W'].T.dot(dZ2) * ReLU_deriv(aWeights['I_H1_W'])
    dWeights['d_I_H1_W'] = 1 / data_rows * dZ1.dot(inputs.T)
    dWeights['d_I_H1_b'] = 1 / data_rows * np.sum(dZ1)
    return dWeights

    
def update_params(weights, dWeights, training_rate):
    weights['I_H1_W'] = weights['I_H1_W'] - training_rate * dWeights['d_I_H1_W']
    weights['I_H1_b'] = weights['I_H1_b'] - training_rate * dWeights['d_I_H1_b']
    weights['H1_O_W'] = weights['H1_O_W'] - training_rate * dWeights['d_H1_O_W']
    weights['H1_O_b'] = weights['H1_O_b'] - training_rate * dWeights['d_H1_O_b']

    return weights

def get_predictions(predictions):
    return np.argmax(predictions,0)


def make_predictions(weights, inputs):
    aWeights = forward_prop(weights, inputs)
    return (get_predictions(aWeights['OUTPUT']))

def get_accuracy(predictions, labels):
    return np.sum(predictions == labels) / labels.size



def train_epoch(inputs,labels,training_rate,epochs):
    weights = init_params(784,[25],10)
    for i in range(epochs):
        aWeights = forward_prop(weights, inputs)
        dWeights = backward_prop(aWeights, weights, inputs,labels)
        weights =update_params(weights, dWeights, training_rate)
        if i % 10 ==0:
            training_rate -= .005
            training_rate = round(training_rate,3)
            if training_rate < .01:
                training_rate = .01
            predictions = get_predictions(aWeights['OUTPUT'])
            accuracy = get_accuracy(predictions,labels)    
            print("Iteration:" , i, " Accuracy: ",accuracy, " New Training Rate: ", training_rate)

    return weights

def test_prediction(index,weights):
    current_image = data_dev_inputs[:,index,None]
    prediction = make_predictions(weights,current_image)[0]
    label = data_dev_labels[index]

    print("Prediction: ",prediction, " Label: ",label)

    current_image = current_image.reshape((28,28)) * 255
    plt.gray()
    plt.imshow(current_image,interpolation= 'nearest')
    plt.show()


if __name__ == '__main__':
    # setup random weights 

    
    # print(data_dev_inputs.shape)
    weights =train_epoch(data_train_inputs,data_train_labels,.5,500)
    # print(weights)
    predictions = make_predictions(weights,data_dev_inputs)
    print(get_accuracy(predictions,data_dev_labels))

    # test_prediction(1,weights)
    