################################################################################
#
# LOGISTICS
#
#    Kapil Gautam
#    KXG180032
#
# FILE
#
#    nn.py
#
# DESCRIPTION
#
#    MNIST image classification with an xNN written and trained in Python
#
#
# INSTRUCTIONS
#
#    1. Go to Google Colaboratory: https://colab.research.google.com/notebooks/welcome.ipynb
#    2. File - New Python 3 notebook
#    3. Cut and paste this file into the cell (feel free to divide into multiple cells)
#    4. Runtime - Run all
#
# NOTES
#
#    1. A summary of my nn.py code:
#
#       < Forward path code summary / highlights
#       Highlights - a.) Incoming data is normalized and vectorized.
#                    b.) Xavier/He weight and bias initialization.
#                    c.) Each epoch has a randomized training input for better training.
#                    d.) During each epoch, the training data is fed into configurable number of mini-batches.
#       >
#       
#       < Error code summary / highlights
#       Highlights - a.) Softmax cross entropy loss function is used to calculate per epoch training error.
#       >
#       
#       < Backward path code summary / highlights
#       Highlights - Calculation of weights and biases correction from error calculated Mini-batch wise.
#       >
#       
#       < Weight update code summary / highlights
#       Highligts - Mini-batch update of weights and biases
#       >
#       
#       < Anything extra to summary / highlight
#       Number of epochs - 10
#       Batch size - 64
#       
#       RANDOM_SEED = 0 , Accuracy = 97.64, He weight initialization
#       RANDOM_SEED = 0 , Accuracy = 97.46, Xavier weight initialization
#       RANDOM_SEED = 1 , Accuracy = 97.46, Xavier weight initialization
#       >
#
#    2. Accuracy display
#
#       < Per epoch display info cut and pasted from your training output 
#       Epoch: 1  Time: 23.551267385482788  Train error: 13.43 %  Loss: 0.5311040238426818  Test accuracy: 92.27 %
#       Epoch: 2  Time: 41.587279319763184  Train error: 7.123333333333333 %  Loss: 0.24940002994660168  Test accuracy: 93.87 %
#       Epoch: 3  Time: 60.703004598617554  Train error: 5.488333333333333 %  Loss: 0.19392815018348822  Test accuracy: 94.95 %
#       Epoch: 4  Time: 79.4790587425232  Train error: 4.486666666666666 %  Loss: 0.15849604899063918  Test accuracy: 95.78 %
#       Epoch: 5  Time: 98.08142185211182  Train error: 3.7066666666666666 %  Loss: 0.13204591142299724  Test accuracy: 96.28 %
#       Epoch: 6  Time: 116.54344129562378  Train error: 3.1516666666666664 %  Loss: 0.11291096018122591  Test accuracy: 96.73 %
#       Epoch: 7  Time: 134.87150239944458  Train error: 2.75 %  Loss: 0.09738927297380222  Test accuracy: 96.94 %
#       Epoch: 8  Time: 153.71536350250244  Train error: 2.36 %  Loss: 0.08526652300500921  Test accuracy: 97.03 %
#       Epoch: 9  Time: 171.8313808441162  Train error: 2.1083333333333334 %  Loss: 0.07525977257859989  Test accuracy: 97.33000000000001 %
#       Epoch: 10  Time: 190.19611310958862  Train error: 1.8766666666666667 %  Loss: 0.06732086765069359  Test accuracy: 97.46000000000001 %
#       >
#
#       <Final accuracy cut and pasted from your training output
#       Final accuracy: 97.46000000000001 %
#       >
#
#    3. Performance display
#
#       < Total training time cut and pasted from your training output
#       Total time taken =  190.19908499717712 seconds
#       >
#
#       < Per layer info (type, input size, output size, parameter size, MACs, ...) for one sample data
#       # Layer type                  Input Size      Output size          MACs            
#       ---------------------        -----------     -----------      -----------         
#       Data                         1 x 28 x 28     1 x 28 x 28      (784)^2 = 6,14,456         
#       Division by 255.0            1 x 28 x 28     1 x 28 x 28      (784)^2 = 6,14,456         
#       Vectorization                1 x 28 x 28     1 x 784          (784)^2 = 6,14,456             
#       Matrix multiplication        1 x 784         1 x 1000          7,84,000        
#       Addition                     1 x 1000        1 x 1000         (1000)^2 = 10,00,000             
#       ReLU                         1 x 1000        1 x 1000         (1000)^2 = 10,00,000               
#       Matrix multiplication        1 x 1000        1 x 100           1,00,000              
#       Addition                     1 x 100         1 x 100          (100)^2 = 10,000          
#       ReLU                         1 x 100         1 x 100          (100)^2 = 10,000        
#       Matrix multiplication        1 x 100         1 x 10            1,000               
#       Addition                     1 x 10          1 x 10           (10)^2 = 100              
#       Softmax                      1 x 10          1 x 10           (10)^2 = 100    
#       >          
#
################################################################################
# NOTES
#
#   This does not use PyTorch, TensorFlow or any other xNN library
#
################################################################################
#
# IMPORT
#
################################################################################

import os.path
import urllib.request
import gzip
import math
import numpy             as np
import matplotlib.pyplot as plt

################################################################################
#
# PARAMETERS
#
################################################################################

#
# add other hyper parameters here with some logical organization
#

# data
DATA_NUM_TRAIN         = 60000
DATA_NUM_TEST          = 10000
DATA_CHANNELS          = 1
DATA_ROWS              = 28
DATA_COLS              = 28
DATA_CLASSES           = 10
DATA_URL_TRAIN_DATA    = 'http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz'
DATA_URL_TRAIN_LABELS  = 'http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz'
DATA_URL_TEST_DATA     = 'http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz'
DATA_URL_TEST_LABELS   = 'http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz'
DATA_FILE_TRAIN_DATA   = 'train_data.gz'
DATA_FILE_TRAIN_LABELS = 'train_labels.gz'
DATA_FILE_TEST_DATA    = 'test_data.gz'
DATA_FILE_TEST_LABELS  = 'test_labels.gz'

# display
DISPLAY_ROWS   = 8
DISPLAY_COLS   = 4
DISPLAY_COL_IN = 10
DISPLAY_ROW_IN = 25
DISPLAY_NUM    = DISPLAY_ROWS*DISPLAY_COLS

# data pre-processing
BATCH_SIZE = 64
NORMALIZATION_FACTOR = 255.0
RANDOM_SEED = 0

#training
TRAIN_EPOCHS = 10
LEVEL_0_CHANNEL = 1000
LEVEL_1_CHANNEL = 100
OUTPUT_CHANNEL = 10
TRAINING_LR_DEFAULT = 0.03
#not using training rate cosine decay as it decreased the accuracy to 86.22% in my case

################################################################################
#
# DATA
#
################################################################################

# download
if (os.path.exists(DATA_FILE_TRAIN_DATA)   == False):
    urllib.request.urlretrieve(DATA_URL_TRAIN_DATA,   DATA_FILE_TRAIN_DATA)
if (os.path.exists(DATA_FILE_TRAIN_LABELS) == False):
    urllib.request.urlretrieve(DATA_URL_TRAIN_LABELS, DATA_FILE_TRAIN_LABELS)
if (os.path.exists(DATA_FILE_TEST_DATA)    == False):
    urllib.request.urlretrieve(DATA_URL_TEST_DATA,    DATA_FILE_TEST_DATA)
if (os.path.exists(DATA_FILE_TEST_LABELS)  == False):
    urllib.request.urlretrieve(DATA_URL_TEST_LABELS,  DATA_FILE_TEST_LABELS)

# training data
# unzip the file, skip the header, read the rest into a buffer and format to NCHW
file_train_data   = gzip.open(DATA_FILE_TRAIN_DATA, 'r')
file_train_data.read(16)
buffer_train_data = file_train_data.read(DATA_NUM_TRAIN*DATA_ROWS*DATA_COLS)
train_data        = np.frombuffer(buffer_train_data, dtype=np.uint8).astype(np.float32)
train_data        = train_data.reshape(DATA_NUM_TRAIN, 1, DATA_ROWS, DATA_COLS)

# training labels
# unzip the file, skip the header, read the rest into a buffer and format to a vector
file_train_labels   = gzip.open(DATA_FILE_TRAIN_LABELS, 'r')
file_train_labels.read(8)
buffer_train_labels = file_train_labels.read(DATA_NUM_TRAIN)
train_labels        = np.frombuffer(buffer_train_labels, dtype=np.uint8).astype(np.int32)

# testing data
# unzip the file, skip the header, read the rest into a buffer and format to NCHW
file_test_data   = gzip.open(DATA_FILE_TEST_DATA, 'r')
file_test_data.read(16)
buffer_test_data = file_test_data.read(DATA_NUM_TEST*DATA_ROWS*DATA_COLS)
test_data        = np.frombuffer(buffer_test_data, dtype=np.uint8).astype(np.float32)
test_data        = test_data.reshape(DATA_NUM_TEST, 1, DATA_ROWS, DATA_COLS)

# testing labels
# unzip the file, skip the header, read the rest into a buffer and format to a vector
file_test_labels   = gzip.open(DATA_FILE_TEST_LABELS, 'r')
file_test_labels.read(8)
buffer_test_labels = file_test_labels.read(DATA_NUM_TEST)
test_labels        = np.frombuffer(buffer_test_labels, dtype=np.uint8).astype(np.int32)

# debug
# print(train_data.shape)   # (60000, 1, 28, 28)
# print(train_labels.shape) # (60000,)
# print(test_data.shape)    # (10000, 1, 28, 28)
# print(test_labels.shape)  # (10000,)

def mnist_one_hot(labels):
  one_hot_labels = np.zeros((len(labels),DATA_CLASSES))
  for i in range(len(labels)):
    one_hot_labels[i,labels[i]] = 1
  return one_hot_labels

train_labels = mnist_one_hot(train_labels)
test_labels = mnist_one_hot(test_labels)

#data normalization
train_data /= NORMALIZATION_FACTOR
test_data /= NORMALIZATION_FACTOR

#Vectorizing the input
vectorized_size = DATA_ROWS * DATA_COLS
train_data = train_data.reshape(-1,vectorized_size)
test_data = test_data.reshape(-1,vectorized_size)

np.random.seed(RANDOM_SEED)

################################################################################
#
# YOUR CODE GOES HERE
#
################################################################################
#
# feel free to split this into some number of classes, functions, ... if it
# helps with code organization; for example, you may want to create a class for
# each of your layers that store parameters, performs initialization and
# includes forward and backward functions    
plot_epochs = []
plot_accuracy = []


class Simple_Neural_Network():
  # initialize hte learningrate, epoch and input/output parameters
  def __init__(self, sizes, epochs=TRAIN_EPOCHS, l_rate=TRAINING_LR_DEFAULT):
    self.sizes = sizes
    self.epochs = epochs
    self.l_rate = l_rate

    # we save all parameters in the neural network in this dictionary
    self.cache = self.initialization(weight_init='xavier')

  def initialization(self, weight_init=''):
    # number of nodes in each layer
    input_layer=self.sizes[0]
    hidden_1=self.sizes[1]
    hidden_2=self.sizes[2]
    output_layer=self.sizes[3]

    if weight_init == 'normal':
      cache = {
        'w1':np.random.randn(input_layer, hidden_1),
        'w2':np.random.randn(hidden_1,hidden_2),
        'w3':np.random.randn(hidden_2,output_layer)
      }
    elif weight_init == 'xavier':
      #xavier initialization of weights
      #weight = np.random.randn(fan_in,fan_out) / np.sqrt(fan_in)
      cache = {
        'w1':np.random.randn(input_layer, hidden_1) / np.sqrt(input_layer),
        'w2':np.random.randn(hidden_1,hidden_2) / np.sqrt(hidden_1),
        'w3':np.random.randn(hidden_2,output_layer) / np.sqrt(hidden_2)
      }
    else: #he initialization, divide fan_in by 2
      cache = {
        'w1':np.random.randn(input_layer, hidden_1) / np.sqrt(input_layer/2),
        'w2':np.random.randn(hidden_1,hidden_2) / np.sqrt(hidden_1/2),
        'w3':np.random.randn(hidden_2,output_layer) / np.sqrt(hidden_2/2)
      }

    print("initialization,init", weight_init, "w1",cache['w1'].shape,"w2",cache['w2'].shape,"w3",cache['w3'].shape)
    cache.update({
      'b1': np.zeros((1, hidden_1)),
      'b2': np.zeros((1, hidden_2)),
      'b3': np.zeros((1, output_layer))
    })

    return cache

  def ReLU(self, x, derivative=False):
    if derivative:
      return np.where(x > 0, 1.0, 0.0)
    return np.maximum(0,x)

  def softmax(self, x):
    exps = np.exp(x)
    return exps / np.sum(exps, axis=1, keepdims=True)

  def crossEntropyLoss(self, probs, label):
    return -np.sum(label * np.log(probs))

  def forward_pass(self, training_data):
    cache = self.cache
    cache['x0'] = training_data
    
    #input layer to layer 1
    cache['y1'] = np.dot(cache['x0'], cache['w1']) + cache['b1']
    cache['x1'] = self.ReLU(cache['y1'])

    #layer 1 to layer 2
    cache['y2'] = np.dot(cache['x1'], cache['w2']) + cache['b2']
    cache['x2'] = self.ReLU(cache['y2'])

    #layer 2 to output layer
    cache['y3'] = np.dot(cache['x2'], cache['w3']) + cache['b3']
    cache['x3'] = self.softmax(cache['y3'])
    return cache['x3']

  def backward_pass(self, train_label, output):
    # print(train_label.shape, train_label[0], np.argmax(train_label[0]))
    cache = self.cache
    # loss
    num_examples = len(train_data)
    probs = output / np.sum(output, axis=1, keepdims=True)
    corect_logprobs = -np.sum(train_label * np.log(probs))
    loss = np.sum(corect_logprobs)
    cache['epoch_loss'] += 1./num_examples * loss

    error = output - train_label
    
    # Calculate W3 update
    cache['batch_w3'] += np.dot(cache['x2'].T, error)
    cache['batch_b3'] += np.sum(error, axis=0, keepdims=True)

    # Calculate W2 update
    error = np.multiply( np.dot(error, cache['w3'].T), self.ReLU(cache['x2'], derivative=True) )
    cache['batch_w2'] += np.dot(cache['x1'].T, error)
    cache['batch_b2'] += np.sum(error, axis=0)
    # print("backward",output.shape,error.shape,cache['batch_w2'].shape, cache['batch_b2'].shape, cache['b2'].shape,cache['w2'].shape)

    # Calculate W1 update
    error = np.multiply( np.dot(error, cache['w2'].T), self.ReLU(cache['x1'], derivative=True) )
    cache['batch_w1'] += np.dot(cache['x0'].T ,error)
    cache['batch_b1'] += np.sum(error, axis=0)

    for x,y in zip(output,train_label):
      cache['epoch_train_error'] += int(np.argmax(x) != np.argmax(y))
    return self.crossEntropyLoss(output, train_label)

  def update_weights(self):
    cache = self.cache
    cache['b3'] -= self.l_rate * (cache['batch_b3'] / BATCH_SIZE)
    cache['b2'] -= self.l_rate * (cache['batch_b2'] / BATCH_SIZE)
    cache['b1'] -= self.l_rate * (cache['batch_b1'] / BATCH_SIZE)
    cache['w3'] -= self.l_rate * (cache['batch_w3'] / BATCH_SIZE)
    cache['w2'] -= self.l_rate * (cache['batch_w2'] / BATCH_SIZE)
    cache['w1'] -= self.l_rate * (cache['batch_w1'] / BATCH_SIZE)

  def compute_accuracy(self, test_data, test_labels):
    # cycle through the testing data
    output = self.forward_pass(test_data)
    error = 0
    for x,y in zip(output,test_labels):
      error += int(np.argmax(x) != np.argmax(y))
    # print("errors",error)
    # count = 0
    # for x in range(len(test_labels)):
    #   if count < 6:
    #     print(x,np.argmax(output[x]),np.argmax(test_labels[x]))
    #   else:
    #     break
    #   count += 1
    return (1-(error/len(test_data)))*100

  def train(self, train_data, train_labels):
    start_time = time.time()
    n = len(train_data)
    cache = self.cache
    # cycle through the epochs
    for iteration in range(self.epochs):
      cache['epoch_train_error'] = 0
      cache['epoch_loss'] = 0

      random_indices = np.random.permutation(len(train_data))
      nn_train_data = train_data[random_indices]
      nn_train_labels = train_labels[random_indices]
      
      for k in range(0, n, BATCH_SIZE):
        batch_x = nn_train_data[k:k+BATCH_SIZE]
        batch_y = nn_train_labels[k:k+BATCH_SIZE]

        cache['batch_w1'] = cache['batch_w2'] = cache['batch_w3'] = 0.
        cache['batch_b1'] = cache['batch_b2'] = cache['batch_b3'] = 0.
        # cycle through the training data in batches
        # forward pass
        output = self.forward_pass(batch_x)
        # back prop
        loss = self.backward_pass(batch_y, output)
        self.update_weights()
        
      test_accuracy = self.compute_accuracy(test_data, test_labels)
      # per epoch display (epoch, time, training loss, testing accuracy
      print('Epoch:', iteration+1,
            ' Time:', time.time() - start_time,
            ' Train error:', cache['epoch_train_error']/len(train_data)*100,"%",
            ' Loss:', cache['epoch_loss'],
            ' Test accuracy:', test_accuracy,"%")   
      plot_epochs.append(iteration+1)
      plot_accuracy.append(test_accuracy)
          
  def test(self, test_data, test_labels):
    # print("start testing")
    accuracy = self.compute_accuracy(test_data, test_labels)
    print("Final accuracy:", accuracy)

  def predictions(self, test_data):
    # print("single prediction")
    output = self.forward_pass(test_data)
    # count = 0
    # for x in range(len(test_labels)):
    #   if count < 6:
    #     print(x,np.argmax(output[x]),np.argmax(test_labels[x]))
    #   else:
    #     break
    #   count += 1
    return output
  
snn = Simple_Neural_Network(sizes=[DATA_CHANNELS*DATA_ROWS*DATA_COLS, LEVEL_0_CHANNEL, LEVEL_1_CHANNEL, DATA_CLASSES])

import time
train_start_time = time.time()
snn.train(train_data, train_labels)
train_end_time = time.time()

################################################################################
#
# DISPLAY
#
################################################################################

#
# more code for you to write
#

# accuracy display
# final value
snn.test(test_data, test_labels)
# plot of accuracy vs epoch
plt.plot(plot_epochs, plot_accuracy)

# performance display
# total time
print("Total time taken = ", train_end_time - train_start_time)

# per layer info (type, input size, output size, parameter size, MACs, ...)
# Layer                     Output
# ---------------------     -----------
# Data                      1 x 28 x 28
# Division by 255.0         1 x 28 x 28
# Vectorization             1 x 784
# Matrix multiplication     1 x 1000
# Addition                  1 x 1000
# ReLU                      1 x 1000
# Matrix multiplication     1 x 100
# Addition                  1 x 100
# ReLU                      1 x 100
# Matrix multiplication     1 x 10
# Addition                  1 x 10
# Softmax                   1 x 10

# example display
# replace the xNN predicted label with the label predicted by the network
fig = plt.figure(figsize=(DISPLAY_COL_IN, DISPLAY_ROW_IN))
ax  = []

predict = snn.predictions(test_data)
for i in range(DISPLAY_NUM):
    img = test_data[i].reshape(DATA_ROWS, DATA_COLS)
    ax.append(fig.add_subplot(DISPLAY_ROWS, DISPLAY_COLS, i + 1))
    ax[-1].set_title('True: ' + str(np.argmax(test_labels[i])) + ' xNN: ' + str(np.argmax(predict[i])))
    plt.imshow(img, cmap='Greys')
plt.show()