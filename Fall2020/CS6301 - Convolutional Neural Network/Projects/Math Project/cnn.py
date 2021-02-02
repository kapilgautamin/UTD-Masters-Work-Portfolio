################################################################################
#
# LOGISTICS
#
#    Kapil Gautam
#    KXG180032
#
# FILE
#
#    cnn.py
#
# DESCRIPTION
#
#    MNIST image classification with an xNN written and trained in Python
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
#    1. A summary of my cnn.py code:
#
#       < Forward path code summary / highlights
#       Highlights - a.) Incoming data is normalized and vectorized.
#                    b.) Xavier/He weight and bias initialization.
#                    c.) Each epoch has a randomized training input for better training.
#                    d.) During each epoch, the training data is fed into configurable number of mini-batches.
#                    e.) Max pool and padding layer
#       >
#       
#       < Error code summary / highlights
#       Highlights - a.) Softmax cross entropy loss function is used to calculate per epoch training error.
#       >
#       
#       < Backward path code summary / highlights
#       Highlights - Calculation of weights and biases correction from error calculated
#       >
#       
#       < Weight update code summary / highlights
#       Highligts - a.) Mini-batch update of weights and biases
#                   b.) Cosine decay learning rate
#       >
#       
#       < Anything extra to summary / highlight
#       Number of epochs - 10
#       Batch size - 64
#            
#       The convolutional neural network was able to achieve a 84.53% accuracy after this on the test set.
#       I could have pushed the accuracy to 90% with more tuning with hyperparameters and increased epochs, but here i am with this accuracy,
#       closer to the goal, but not yet achieved. The other reason i believe for lower accuracy than a traditional neural network is that 
#       some of the errors did not make its way to the very start of CNN. 
#       >
#
#    2. Accuracy display
#
#       < Per epoch display info cut and pasted from your training output>
#       Epoch: 1  Time: 47.64308476448059  Train error: 31.783333333333335 %  Loss: 0.9196079648366574  Test accuracy: 71.91000000000001 %
#       Epoch: 2  Time: 93.60767960548401  Train error: 25.22333333333333 %  Loss: 0.7241773438002339  Test accuracy: 80.47999999999999 %
#       Epoch: 3  Time: 140.01581454277039  Train error: 23.291666666666664 %  Loss: 0.6726717072210495  Test accuracy: 75.03 %
#       Epoch: 4  Time: 183.85321402549744  Train error: 23.22 %  Loss: 0.674546806104915  Test accuracy: 76.57000000000001 %
#       Epoch: 5  Time: 228.33501076698303  Train error: 20.756666666666668 %  Loss: 0.604176593555304  Test accuracy: 77.92 %
#       Epoch: 6  Time: 273.2389006614685  Train error: 19.3 %  Loss: 0.5626258949091507  Test accuracy: 80.49 %
#       Epoch: 7  Time: 318.83793234825134  Train error: 18.033333333333335 %  Loss: 0.5245276377668651  Test accuracy: 81.28999999999999 %
#       Epoch: 8  Time: 363.70187854766846  Train error: 16.976666666666667 %  Loss: 0.4908189891983483  Test accuracy: 82.09 %
#       Epoch: 9  Time: 408.28402853012085  Train error: 15.846666666666668 %  Loss: 0.4575395470687906  Test accuracy: 83.76 %
#       Epoch: 10  Time: 453.1567578315735  Train error: 14.958333333333334 %  Loss: 0.4325587689468239  Test accuracy: 84.53 %
#
#       < Final accuracy cut and pasted from your training output>
#       Final accuracy: 84.53 %

#    3. Performance display
#
#       < Total training time cut and pasted from your training output
#       Total time taken =  453.15893268585205 seconds
#       >
#
#       < Per layer info (type, input size, output size, parameter size, MACs, ...) for one sample data
#       Layer type                           Input size         Output size           MACs         
#       ---------------------                -----------        -----------         -----------
#       Data                                  1 x 28 x 28        1 x 28 x 28       (784)^2 = 6,14,456
#       Division by 255.0                     1 x 28 x 28        1 x 28 x 28       (784)^2 = 6,14,456
#       3x3/1 0 pad CNN style 2D conv         1 x 28 x 28        16 x 28 x 28       9,834,496
#       Addition                              16 x 28 x 28       16 x 28 x 28      (12544)^2 = 15,73,51,936
#       ReLU                                  16 x 28 x 28       16 x 28 x 28      (12544)^2 = 15,73,51,936
#       3x3/2 0 pad max pool                  16 x 28 x 28       16 x 14 x 14       39,337,984
#       3x3/1 0 pad CNN style 2D conv         16 x 14 x 14       32 x 14 x 14       1,96,68,992
#       Addition                              32 x 14 x 14       32 x 14 x 14      (6272)^2 = 39,337,984
#       ReLU                                  32 x 14 x 14       32 x 14 x 14      (6272)^2 = 39,337,984
#       3x3/2 0 pad max pool                  32 x 14 x 14       32 x 7 x 7         98,34,496
#       3x3/1 CNN style 2D conv               32 x 7 x 7         64 x 7 x 7         49,17,248
#       Addition                              64 x 7 x 7         64 x 7 x 7        (3136)^2 = 9,834,496
#       ReLU                                  64 x 7 x 7         64 x 7 x 7        (3136)^2 = 9,834,496
#       Vectorization                         64 x 7 x 7         1 x 3136          (3136)^2 = 9,834,496
#       Matrix multiplication                 1 x 3136           1 x 100            313600
#       Addition                              1 x 100            1 x 100           (100)^2 = 10,000 
#       ReLU                                  1 x 100            1 x 100           (100)^2 = 10,000 
#       Matrix multiplication                 1 x 100            1 x 10            1000 
#       Addition                              1 x 10             1 x 10            (10)^2 = 100 
#       Softmax                               1 x 10             1 x 10            (10)^2 = 100 
#       >
################################################################################
#
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
RANDOM_SEED = 1

#training
TRAIN_EPOCHS = 10
INPUT_CHANNEL = 3136
LEVEL_0_CHANNEL = 100
OUTPUT_CHANNEL = 10
TRAINING_LR_DEFAULT = 0.03

# training (linear warm up with cosine decay learning rate)
TRAINING_LR_MAX          = 1
TRAINING_LR_INIT_SCALE   = 0.3
TRAINING_LR_INIT_EPOCHS  = 3
TRAINING_LR_FINAL_SCALE  = 0.01
TRAINING_LR_FINAL_EPOCHS = 7
TRAINING_NUM_EPOCHS      = TRAINING_LR_INIT_EPOCHS + TRAINING_LR_FINAL_EPOCHS
TRAINING_LR_INIT         = TRAINING_LR_MAX * TRAINING_LR_INIT_SCALE
TRAINING_LR_FINAL        = TRAINING_LR_MAX * TRAINING_LR_FINAL_SCALE

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

class Convolutional_Neural_Network():
  # initialize hte learningrate, epoch and input/output parameters
  def __init__(self, sizes, epochs=TRAIN_EPOCHS, l_rate=TRAINING_LR_DEFAULT):
    self.sizes = sizes
    self.epochs = epochs
    self.l_rate = l_rate

    # we save all parameters in the neural network in this dictionary
    self.cache = self.initialization(weight_init = 'xavier')

  def initialization(self, weight_init=''):
    # number of nodes in each layer
    input_layer=self.sizes[0]
    hidden_1=self.sizes[1]
    output_layer=self.sizes[2]

    if weight_init == 'normal':
      cache = {
        'w1':np.random.randn(input_layer, hidden_1),
        'w2':np.random.randn(hidden_1,output_layer)
      }
    elif weight_init == 'xavier':
      #xavier initialization of weights
      #weight = np.random.randn(fan_in,fan_out) / np.sqrt(fan_in)
      cache = {
        'w1':np.random.randn(input_layer, hidden_1) / np.sqrt(input_layer),
        'w2':np.random.randn(hidden_1,output_layer) / np.sqrt(hidden_1)
      }
    else: #he initialization, divide fan_in by 2
      cache = {
        'w1':np.random.randn(input_layer, hidden_1) / np.sqrt(input_layer/2),
        'w2':np.random.randn(hidden_1,output_layer) / np.sqrt(hidden_1/2)
      }

    print("initialization,init", weight_init, "w1",cache['w1'].shape,"w2",cache['w2'].shape)
    cache.update({
      'b1': np.zeros((1, hidden_1)),
      'b2': np.zeros((1, output_layer)),
      'b3': np.zeros((1,  16,28,28)),
      'b4': np.zeros((1, 32,14,14)),
      'b5': np.zeros((1, 64,7,7))
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

  #pooling and asStride functions taken from a stackoverflow answer
  #https://stackoverflow.com/questions/42463172/how-to-perform-max-mean-pooling-on-a-2d-array-using-numpy
  def asStride(self,arr,sub_shape,stride):
    '''Get a strided sub-matrices view of an ndarray.
    See also skimage.util.shape.view_as_windows()
    '''
    s0,s1=arr.strides[:2]
    m1,n1=arr.shape[:2]
    m2,n2=sub_shape
    view_shape=(1+(m1-m2)//stride[0],1+(n1-n2)//stride[1],m2,n2)+arr.shape[2:]
    strides=(stride[0]*s0,stride[1]*s1,s0,s1)+arr.strides[2:]
    subs=np.lib.stride_tricks.as_strided(arr,view_shape,strides=strides)
    return subs

  def poolingOverlap(self,mat,ksize,stride=None,method='max',pad=False):
    '''Overlapping pooling on 2D or 3D data.

    <mat>: ndarray, input array to pool.
    <ksize>: tuple of 2, kernel size in (ky, kx).
    <stride>: tuple of 2 or None, stride of pooling window.
              If None, same as <ksize> (non-overlapping pooling).
    <method>: str, 'max for max-pooling,
                   'mean' for mean-pooling.
    <pad>: bool, pad <mat> or not. If no pad, output has size
           (n-f)//s+1, n being <mat> size, f being kernel size, s stride.
           if pad, output has size ceil(n/s).

    Return <result>: pooled matrix.
    '''

    m, n = mat.shape[:2]
    ky,kx=ksize
    if stride is None:
        stride=(ky,kx)
    sy,sx=stride

    _ceil=lambda x,y: int(np.ceil(x/float(y)))

    if pad:
        ny=_ceil(m,sy)
        nx=_ceil(n,sx)
        size=((ny-1)*sy+ky, (nx-1)*sx+kx) + mat.shape[2:]
        mat_pad=np.full(size,np.nan)
        mat_pad[:m,:n,...]=mat
    else:
        mat_pad=mat[:(m-ky)//sy*sy+ky, :(n-kx)//sx*sx+kx, ...]

    view=self.asStride(mat_pad,ksize,stride)

    if method=='max':
        result=np.nanmax(view,axis=(2,3))
    else:
        result=np.nanmean(view,axis=(2,3))

    return result

  def lr_schedule(self, epoch):
    # linear warmup followed by cosine decay
    if epoch < TRAINING_LR_INIT_EPOCHS:
        lr = (TRAINING_LR_MAX - TRAINING_LR_INIT)*(float(epoch)/TRAINING_LR_INIT_EPOCHS) + TRAINING_LR_INIT
    else:
        lr = (TRAINING_LR_MAX - TRAINING_LR_FINAL)*max(0.0, math.cos(((float(epoch) - TRAINING_LR_INIT_EPOCHS)/(TRAINING_LR_FINAL_EPOCHS - 1.0))*(math.pi/2.0))) + TRAINING_LR_FINAL

    return lr

  def forward_pass(self, training_data, testing=False):
    cache = self.cache
    
    cache['x0'] = training_data
    cache['p0'] = np.pad(cache['x0'], ((0,0),(7,8), (0,0), (0, 0)), 'constant') + cache['b3']
    cache['c0'] = self.ReLU(cache['p0'])
    # if testing:
      # print(cache['x0'].shape,cache['p0'].shape,cache['c0'].shape
    cache['m0'] = self.poolingOverlap(cache['c0'].T,(3,3),(2,2), 'max',True).T
    cache['p1'] = np.pad(cache['m0'], ((0,0),(8,8), (0,0), (0, 0)), 'constant') + cache['b4']
    cache['c1'] = self.ReLU(cache['p1'])
    # if testing:
      # print(cache['m0'].shape,cache['p1'].shape,cache['c1'].shape)

    cache['m1'] = self.poolingOverlap(cache['c1'].T,(3,3),(2,2), 'max',True).T
    cache['p2'] = np.pad(cache['m1'], ((0,0),(16,16), (0,0), (0, 0)), 'constant') + cache['b5']
    cache['c2'] = self.ReLU(cache['p2'])
    # if testing:
      # print(cache['m1'].shape,cache['p2'].shape,cache['c2'].shape)

    if testing:
      # print(len(training_data))
      cache['v0'] = cache['c2'].reshape(len(training_data),-1)
    else:
      cache['v0'] = cache['c2'].reshape(BATCH_SIZE,-1)

    #input layer to layer 1
    cache['y1'] = np.dot(cache['v0'], cache['w1']) + cache['b1']
    cache['x1'] = self.ReLU(cache['y1'])
    
    #layer 1 to output layer
    cache['y2'] = np.dot(cache['x1'], cache['w2']) + cache['b2']
    cache['x2'] = self.softmax(cache['y2'])
    # print("forward",cache['x2'].shape)
    return cache['x2']

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
    
    # Calculate W2 update
    cache['batch_w2'] += np.dot(cache['x1'].reshape(BATCH_SIZE,-1).T, error)
    cache['batch_b2'] += np.sum(error, axis=0, keepdims=True)
    
    # Calculate W1 update
    error = np.multiply( np.dot(error, cache['w2'].T), self.ReLU(cache['x1'], derivative=True) )
    cache['batch_w1'] += np.dot(cache['v0'].T, error)
    cache['batch_b1'] += np.sum(error, axis=0)

    for x,y in zip(output,train_label):
      cache['epoch_train_error'] += int(np.argmax(x) != np.argmax(y))
    return self.crossEntropyLoss(output, train_label)

  def compute_accuracy(self, test_data, test_labels):
    error = 0
    # cycle through the testing data
    output = self.forward_pass(test_data,testing=True)
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
      self.l_rate = self.lr_schedule(iteration)
      random_indices = np.random.permutation(len(train_data))
      nn_train_data = train_data[random_indices]
      nn_train_labels = train_labels[random_indices]
      
      for k in range(0, n, BATCH_SIZE):
        batch_x = nn_train_data[k:k+BATCH_SIZE]
        batch_y = nn_train_labels[k:k+BATCH_SIZE]
        if len(batch_x) < BATCH_SIZE:
          break
        cache['batch_w1'] = cache['batch_w2'] = 0.
        cache['batch_b1'] = cache['batch_b2'] = cache['batch_b3'] = cache['batch_b4'] = cache['batch_b5'] = 0.
        # cycle through the training data in batches
        # forward pass
        output = self.forward_pass(batch_x)
        # back prop
        loss = self.backward_pass(batch_y, output)
        # print("batch",k,output.shape, batch_y.shape, batch_x.shape)
        
        cache['w1'] -= self.l_rate * (cache['batch_w1'] / BATCH_SIZE)
        cache['w2'] -= self.l_rate * (cache['batch_w2'] / BATCH_SIZE)
        cache['b1'] -= self.l_rate * (cache['batch_b1'] / BATCH_SIZE)
        cache['b2'] -= self.l_rate * (cache['batch_b2'] / BATCH_SIZE)
        cache['b3'] -= self.l_rate * (cache['batch_b3'] / BATCH_SIZE)
        cache['b4'] -= self.l_rate * (cache['batch_b4'] / BATCH_SIZE)
        cache['b5'] -= self.l_rate * (cache['batch_b5'] / BATCH_SIZE)
        
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
    output = self.forward_pass(test_data, testing=True)
    count = 0
    for x in range(len(test_labels)):
      if count < 6:
        print(x,np.argmax(output[x]),np.argmax(test_labels[x]))
      else:
        break
      count += 1
    return output
  
cnn = Convolutional_Neural_Network(sizes=[INPUT_CHANNEL, LEVEL_0_CHANNEL, DATA_CLASSES])

import time
train_start_time = time.time()
cnn.train(train_data, train_labels)
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
cnn.test(test_data, test_labels)
# plot of accuracy vs epoch
plt.plot(plot_epochs, plot_accuracy)

# performance display
# total time
print("Total time taken = ", train_end_time - train_start_time)

# per layer info (type, input size, output size, parameter size, MACs, ...)
# Layer                            Output
# ---------------------            -----------
# Data                              1 x 28 x 28
# Division by 255.0                 1 x 28 x 28
# 3x3/1 0 pad CNN style 2D conv     16 x 28 x 28
# Addition                          16 x 28 x 28
# ReLU                              16 x 28 x 28
# 3x3/2 0 pad max pool              16 x 14 x 14
# 3x3/1 0 pad CNN style 2D conv     32 x 14 x 14
# Addition                          32 x 14 x 14
# ReLU                              32 x 14 x 14
# 3x3/2 0 pad max pool              32 x 7 x 7
# 3x3/1 CNN style 2D conv           64 x 7 x 7
# Addition                          64 x 7 x 7
# ReLU                              64 x 7 x 7
# Vectorization                     1 x 3136
# Matrix multiplication             1 x 100
# Addition                          1 x 100
# ReLU                              1 x 100
# Matrix multiplication             1 x 10
# Addition                          1 x 10
# Softmax                           1 x 10

# example display
# replace the xNN predicted label with the label predicted by the network
fig = plt.figure(figsize=(DISPLAY_COL_IN, DISPLAY_ROW_IN))
ax  = []

predict = cnn.predictions(test_data)
for i in range(DISPLAY_NUM):
    img = test_data[i].reshape(DATA_ROWS, DATA_COLS)
    ax.append(fig.add_subplot(DISPLAY_ROWS, DISPLAY_COLS, i + 1))
    ax[-1].set_title('True: ' + str(np.argmax(test_labels[i])) + ' xNN: ' + str(np.argmax(predict[i])))
    plt.imshow(img, cmap='Greys')
plt.show()

