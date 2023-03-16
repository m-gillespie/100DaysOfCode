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


def init_params():
    W1 = np.random.rand(10,784) - 0.5
    # W1 = np.random.rand(1,2) - 1.5
    b1 = np.random.rand(10,1)- 0.5
    W2 = np.random.rand(10,10)- 0.5
    b2 = np.random.rand(10,1)- 0.5
    return W1, b1, W2, b2
    
def ReLU(Z):
    # need to understand this math more. Why can't we return negative numbers?
    return np.maximum(Z,0)

def softmax(Z):
    return np.exp(Z) / sum(np.exp(Z))

def forward_prop(W1, b1, W2, b2, inputs):
    # dot product of input layer to hidden layer weights + bias
    Z1 = W1.dot(inputs) + b1
    # ReLU of dot product (returns positive number, or 0)
    A1 = ReLU(Z1)
    # Resulting A1 weights dot product output weights
    Z2 = W2.dot(A1) + b2
    # softmax those (to ensure sum of all outputs =)
    A2 = softmax(Z2)
    return Z1 , A1 , Z2 , A2

def one_hot(labels):
    # specific to this data set, return an array of label output with 0's except for position of label set to 1.
    one_h = np.zeros((labels.size,labels.max() +1))
    one_h[np.arange(labels.size),labels] =1
    return one_h.T

def ReLU_deriv(Z):
    return Z > 0

def backward_prop(Z1, A1, Z2 , A2, W1, W2, inputs,labels):
    one_hot_labels = one_hot(labels)
    dZ2 = A2 - one_hot_labels
    dW2 = 1 / data_rows * dZ2.dot(A1.T)
    db2 = 1 / data_rows * np.sum(dZ2)
    dZ1 = W2.T.dot(dZ2) * ReLU_deriv(Z1)
    dW1 = 1 / data_rows * dZ1.dot(inputs.T)
    db1 = 1 / data_rows * np.sum(dZ1)
    return dW1, db1, dW2, db2

    
def update_params(W1 , b1, W2, b2, dW1, db1, dW2, db2, training_rate):
    W1 = W1 - training_rate * dW1
    b1 = b1 - training_rate * db1
    W2 = W2 - training_rate * dW2
    b2 = b2 - training_rate * db2

    return W1, b1, W2, b2

def get_predictions(predictions):
    return np.argmax(predictions,0)


def make_predictions(W1 , b1, W2, b2, inputs):
    _,_,_, A2 = forward_prop(W1 , b1, W2, b2, inputs)
    return (get_predictions(A2))

def get_accuracy(predictions, labels):
    return np.sum(predictions == labels) / labels.size



def train_epoch(inputs,labels,training_rate,epochs):
    W1, b1, W2, b2 = init_params()
    for i in range(epochs):
        Z1 , A1, Z2, A2 = forward_prop(W1, b1, W2, b2, inputs)
        dW1, db1, dW2, db2 = backward_prop(Z1, A1, Z2, A2, W1, W2, inputs,labels)
        W1, b1, W2, b2 =update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, training_rate)
        if i % 10 ==0:

            training_rate -= .005
            training_rate = round(training_rate,3)
            if training_rate < .01:
                training_rate = .01
            predictions = get_predictions(A2)
            accuracy = get_accuracy(predictions,labels)    
            print("Iteration:" , i, " Accuracy: ",accuracy, " New Training Rate: ", training_rate)

    return W1, b1, W2, b2

def test_prediction(index,W1,b1,W2,b2):
    current_image = data_dev_inputs[:,index,None]
    prediction = make_predictions(W1,b1,W2,b2,current_image)[0]
    label = data_dev_labels[index]

    print("Prediction: ",prediction, " Label: ",label)

    current_image = current_image.reshape((28,28)) * 255
    plt.gray()
    plt.imshow(current_image,interpolation= 'nearest')
    plt.show()


if __name__ == '__main__':
    # setup random weights 

    
    # print(data_dev_inputs.shape)
    W1, b1, W2, b2 =train_epoch(data_train_inputs,data_train_labels,.5,1000)
    predictions = make_predictions(W1, b1, W2, b2,data_dev_inputs)
    print(get_accuracy(predictions,data_dev_labels))

    # prediction = predict(W1,b1,W2,b2,data_dev_inputs[:,1:10])
    test_prediction(1,W1,b1,W2,b2)
    test_prediction(2,W1,b1,W2,b2)
    test_prediction(3,W1,b1,W2,b2)
    test_prediction(4,W1,b1,W2,b2)
