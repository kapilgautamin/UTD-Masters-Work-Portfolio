This project uses NLP to compute the Textual Entailment of premise and hypothesis.
The training/testing data is provided through a xml file format.
Training data size = 567
Testing data size = 800
Since this seems like small dataset, owing to the low accuracies in predicting the data.

Model Architecture:
    The architecture of our model for textual entailment is fairly simple. It contains the following layers:
    1. Embedding layer: This layer transforms the integer-encoded representations of the sentences into dense vectors.
    2. Recurrent layer: This is a stacked bi-directional LSTM layer that takes in the vector representation from the Embedding
    layer and outputs another vector.
    3. Fully connected layer: This layer transforms the output of the RNN into a vector of 2 dimensions. (one corresponding
    to each label i.e. Entails and Not Entails)
    A schematic showing the architecture of the model is provided below:

The program can be used on the CLI.
With no default parameters, it takes all the default values.
For help on how to use the program, help is available by:
> python project.py -h
usage: project.py [-h] [--epochs EPOCHS] [--learningrate LEARNINGRATE]
                  [--batch_size BATCH_SIZE] [--dropout DROPOUT]
                  [--bidirectional BIDIRECTIONAL] [--output_size OUTPUT_SIZE]
                  [--hidden_size HIDDEN_SIZE] [--nLayers NLAYERS]

NLP Textual Entailment using LSTM

optional arguments:
  -h, --help            show this help message and exit
  --epochs EPOCHS                   Give the number of epochs, default = 5
  --learningrate LEARNINGRATE       Learning rate for model, default = 0.001
  --batch_size BATCH_SIZE           Batch size, default = 25
  --dropout DROPOUT                 Dropout for LSTM, default = 0.3
  --bidirectional BIDIRECTIONAL     LSTM bidirectional?, default = True  
  --output_size OUTPUT_SIZE         Final output size, default = 2 -> (Entails,NotEntails)       
  --hidden_size HIDDEN_SIZE         Hidden inputs size for LSTM model, default = 256     
  --nLayers NLAYERS                 #Layers for LSTM, default = 2



Example run:
> python project.py
Output :

Parameters : batchsize = 25, learning rate = 0.001 , hidden_size = 256, max_length = 66, vocab_size = 7559
LSTM(
  (embed): Embedding(7559, 66)
  (lstm): LSTM(66, 256, num_layers=2, dropout=0.3, bidirectional=True)
  (fc): Linear(in_features=512, out_features=2, bias=True)
  (sigmoid): Sigmoid()
)
Epoch:0, Loss: 0.69381
Epoch:1, Loss: 0.69234
Epoch:2, Loss: 0.69399
Epoch:3, Loss: 0.69382
Epoch:4, Loss: 0.69315
Training time = 208.02 seconds
Test loss: 0.69208
Testing time = 17.49 seconds
Test Accuracy: 53.000%
              precision    recall  f1-score   support

      ENTAIL       0.52      0.93      0.66       400
  NOT ENTAIL       0.64      0.14      0.22       400

   micro avg       0.53      0.53      0.53       800
   macro avg       0.58      0.53      0.44       800
weighted avg       0.58      0.53      0.44       800

Confusion Matrix:
[[370  30]
 [346  54]]
 
 
 
 The accuracy of the model is about 50-53% depeding on the various hyper parameters and the simple layer model 
being used in the description.
From tuning different parameters, the best accuracy(53%) is observed when
--> Dropout rate = 0.3, batch_size = 25, hidden_size = 256, Epoch = 5, learning rate = 1e-3, nLayers = 2

Other Data for Tuning of hyper-parameters:

--> Dropout rate = 0.5, hidden_size = 256, Epoch = 5, learning rate = 1e-2, nLayers = 2
batch - 20 - 50.500%    Training time = 195.37 seconds  Testing time = 16.17 seconds    Confusion Matrix: [[373  27]  [369  31]]
batch - 25 - 51.375%    Training time = 179.32 seconds  Testing time = 18.72 seconds    Confusion Matrix: [[314  86]  [303  97]]
bathc - 30 - 50.625%    Training time = 154.12 seconds  Testing time = 15.80 seconds    Confusion Matrix: [[374  26]  [369  31]]
bathc - 35 - 49.250%    Training time = 165.05 seconds  Testing time = 15.34 seconds    Confusion Matrix: [[ 35 365]  [ 41 359]]
batch - 40 - 51.625%    Training time = 145.33 seconds  Testing time = 14.47 seconds    Confusion Matrix: [[ 76 324]  [ 63 337]]
batch - 42 - 48.375%    Training time = 143.67 seconds  Testing time = 14.40 seconds    Confusion Matrix: [[353  47]  [366  34]]
batch - 45 - 50.125%    Training time = 151.85 seconds  Testing time = 14.33 seconds    Confusion Matrix: [[340  60]  [339  61]]
bacth - 50 - 48.250%    Training time = 131.30 seconds  Testing time = 13.05 seconds    Confusion Matrix: [[119 281]  [133 267]]

--> Dropout rate = 0.3, hidden_size = 256, Epoch = 5, learning rate = 1e-3, nLayers = 2
batch - 20 - 49.250%  Training time = 164.03 seconds  Testing time = 14.13 seconds  Confusion Matrix:[[353  47] [359  41]]
batch - 25 - 53.000%  Training time = 293.03 seconds  Testing time = 18.08 seconds  Confusion Matrix:[[370  30] [346  54]]
bathc - 30 - 48.625%  Training time = 140.78 seconds  Testing time = 12.75 seconds  Confusion Matrix:[[117 283] [128 272]]
bathc - 35 - 52.000%  Training time = 174.33 seconds  Testing time = 16.29 seconds  Confusion Matrix:[[125 275] [109 291]]
batch - 40 - 51.000%  Training time = 137.91 seconds  Testing time = 12.39 seconds  Confusion Matrix:[[120 280] [112 288]]
batch - 42 - 51.125%  Training time = 137.52 seconds  Testing time = 12.34 seconds  Confusion Matrix:[[122 278] [113 287]]
batch - 45 - 49.000%  Training time = 129.48 seconds  Testing time = 11.93 seconds  Confusion Matrix:[[113 287] [121 279]]
bacth - 50 - 48.000%  Training time = 159.86 seconds  Testing time = 16.01 seconds  Confusion Matrix:[[108 292] [124 276]]


--> batch size = 40 learning rate = 1e-2,Epoch = 5, learning rate = 1e-3, nLayers = 2
Dropout = 0.3  -  51.50%
Dropout = 0.5  -  51.625%
Dropout = 0.7  -  51.250%

--> batch size = 40, dropout = 0.5, learning rate = 1e-2, nLayers = 2, epoch = 5
hidden_size - 64   -  50.875%
hidden_size - 128  -  49.250%
hidden_size - 200  -  50.625%
hidden_size - 256  -  51.625%
hidden_size - 328  -  48.000%
hidden_size - 512  -  49.250%


--> batch size = 40,dropout = 0.5, hidden_size = 256, epoch = 5, nLayers = 2
learning rate - 1e-1  -  49.875%
learning rate - 1e-2  -  51.625%
learning rate - 1e-3  -  50.875%
learning rate - 1e-4  -  51.125%
learning rate - 1e-5  -  51.000%

--> batch size = 25,dropout = 0.5, hidden_size = 256, epoch = 5, nLayers = 2
learning rate - 1e-1  -  50.250%
learning rate - 1e-2  -  51.375%
learning rate - 1e-3  -  52.875%
learning rate - 1e-4  -  52.000%
learning rate - 1e-5  -  51.875%

--> batch size = 25, dropout = 0.35, learning rate = 1e-3, epoch = 5
hidden_size - 64   -  49.750%
hidden_size - 128  -  48.750%
hidden_size - 256  -  52.875%
hidden_size - 328  -  49.125%
hidden_size - 512  -  48.375%

--> batch size = 25, learning rate = 1e-3, epoch = 5,hidden_size = 25, nLayers = 2
dropout 0.7 - 52.875%
dropout 0.3 - 53.000% 

--> batch size = 25, learning rate = 1e-3, epoch = 5,hidden_size = 25, dropout = 0.3, nLayers = 2
nLayers = 2 - 53.000%
nLayers = 3 - 49.375%
nLayers = 4 - 50.875%
nLayers = 5 - 49.5%
