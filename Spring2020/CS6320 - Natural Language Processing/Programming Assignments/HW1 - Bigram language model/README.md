The program uses Python3 to calculate the bigram counts and bigram probabilities
 of the given sentences in the test set and gives the probability of the whole sentence.
The program is trained from a training corpus provided for bigram model.

To run the program:
python3 bigram.py arg1 arg2 arg3

arg1 is the location of the training file.
arg2 is the location of the test file.
arg3 is the one-smoothing - it takes value 0/1

By default it will take arg1 as "train.txt", arg2 as "test.txt", arg3 as '1', unless the parameters
are given on the command line interface.

Example run :
> python3 bigram.py train.txt test.txt 0
For no one-smoothing

Example run:
> python3 bigram.py train.txt test.txt 1
For one-smoothing

The output can be stored in a text file.
Note : Please check that the word wrap feature of text file needs to be disabled,
to properly view the output.
Example run:
> python3 bigram.py train.txt test.txt 1 > output.txt
 

The program must outputs the following:
1. A matrix showing the bigram counts for each sentence
2. A matrix showing the bigram probabilities for each sentence
3. The probability of each sentence

The program was tested on csgrad1 server and was founded to be working as required.