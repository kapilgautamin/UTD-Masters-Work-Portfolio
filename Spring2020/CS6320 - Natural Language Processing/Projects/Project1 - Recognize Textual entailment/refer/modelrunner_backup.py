import tarfile
import sys
import os
import torch
import numpy as np
from string import punctuation
from collections import Counter
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
from torch import optim
from gensim.models import KeyedVectors as w2v
import pickle 
import time 

import pandas as pd
import matplotlib.pyplot as plt

# sys.stdout=open('./output.txt', 'w')

# Task 1: Load the data
# For this task you will load the data, create a vocabulary and encode the reviews with integers

def read_file(path_to_dataset):
    """
    :param path_to_dataset: a path to the tar file (dataset)
    :return: two lists, one containing the movie reviews and another containing the corresponding labels
    """
    reviews_train,labels_train,reviews_test,labels_test=[],[],[],[]
    if not tarfile.is_tarfile(path_to_dataset):
        sys.exit("Input path is not a tar file")
    dirent = tarfile.open(path_to_dataset)
    """
    COMPLETE THE REST OF THE METHOD
    """
    for member in dirent.getmembers():
        if member.isfile() and member.name.split('/')[-1].find('.txt') > -1 and member.name.split('/')[-1][0]!='.':
            # print(member.name)
            train_or_test=member.name.split('/')[-3]
            label=member.name.split('/')[-2]
            f=dirent.extractfile(member)
            if f is not None:
                content=f.read()
                # content=content.decode('utf-8')
                content=str(content,'utf-8')
                content=content.replace('\n','')
                content= content.strip('\\')
                content= content.lower()
                content_text= ''.join([c for c in content if c not in punctuation]) # remove the punctuations 
                content_text.rstrip()
                # print(content_text)
                if train_or_test=='train':
                    reviews_train.append(content_text)
                    labels_train.append('POSITIVE' if label =='pos' else 'NEGATIVE')
                elif train_or_test=='test':
                    reviews_test.append(content_text)
                    labels_test.append('POSITIVE' if label =='pos' else 'NEGATIVE')
           
    dirent.close()

    return reviews_train,labels_train,reviews_test,labels_test

def preprocess(text):
    """
    :param text: list of sentences or movie reviews
    :return: a dict of all tokens you encounter in the dataset. i.e. the vocabulary of the dataset
    Associate each token with a unique integer
    """

    if type(text) is not list:
        sys.exit("Please provide a list to the method")
    """
    COMPLETE THE REST OF THE METHOD
    """
    all_text= ''.join(text)
    # print(all_text)
    words= all_text.split()
    count_words= Counter(words) # Count the occurence of all the words in the combined reviews 
    # print(count_words)

    vocab_to_int = {val:i+1 for i, val in enumerate(count_words.keys())}

    return vocab_to_int


def encode_review(vocab, reviews):
    """
    :param vocab: the vocabulary dictionary you obtained from the previous method
    :param text: list of movie reviews obtained from the previous method
    :return: encoded reviews
    """

    if type(vocab) is not dict or type(reviews) is not list:
        sys.exit("Please provide a list to the method")
    """
    COMPLETE THE REST OF THE METHOD
    """
    # print(vocab)
    encoded_reviews=[]

    for review in reviews : 
        encoded_review=[vocab[word] for word in review.split()]
        # print(encoded_review)
        encoded_reviews.append(encoded_review)
    # print(encoded_reviews)
    

    return encoded_reviews


def encode_labels(labels): # Note this method is optional (if you have not integer-encoded the labels)
    """
    :param labels: list of labels associated with the reviews
    :return: encoded labels
    """

    if type(labels) is not list:
        sys.exit("Please provide a list to the method")
    """
    COMPLETE THE REST OF THE METHOD
    """
    encoded_labels= [1 if label=='POSITIVE' else 0 for label in labels]
    # encoded_labels = np.array(encoded_labels)
    return encoded_labels


def pad_zeros(encoded_reviews, seq_length = 200):
    """
    :param encoded_reviews: integer-encoded reviews obtained from the previous method
    :param seq_length: maximum allowed sequence length for the review
    :return: encoded reviews after padding zeros
    """

    if type(encoded_reviews) is not list:
        sys.exit("Please provide a list to the method")
    """
    COMPLETE THE REST OF THE METHOD
    """
    features = np.zeros((len(encoded_reviews), seq_length), dtype = int)
    for i, review in enumerate(encoded_reviews):
        review_len = len(review)
        
        if review_len <= seq_length:
            zeroes = list(np.zeros(seq_length-review_len))
            new = review + zeroes        
        elif review_len > seq_length:
            new = review[0:seq_length]       
        
        features[i,:] = np.array(new)
    
    return features

# Task 2: Load the pre-trained embedding vectors
# For this task you will load the pre-trained embedding vectors from Word2Vec

def load_embedding_file(embedding_file, token_dict):
    """
    :param embedding_file: path to the embedding file
    :param token_dict: token-integer mapping dict obtained from previous step
    :return: embedding dict: embedding vector-integer mapping
    """
    if not os.path.isfile(embedding_file):
        sys.exit("Input embedding path is not a file")
    if type(token_dict) is not dict:
        sys.exit("Input a dictionary!")
    
    embedded_dict={}
    model=w2v.load_word2vec_format(embedding_file)
    
    for key in token_dict.keys():
        if key not in model :
            # print(key)
            # token_dict[key]=len(token_dict)+1
            embedded_dict[token_dict[key]] = torch.zeros(300).uniform_(-1,1)
        else:
            embedded_dict[token_dict[key]]=torch.Tensor(model[key])

    with open('w2v_embedding_pickle.p','wb') as fp:
        pickle.dump(embedded_dict,fp)

    return embedded_dict

def open_embed_dict():
    with open('w2v_embedding_pickle.p','rb') as f:
        embed_dictionary = pickle.load(f)
    embed_dictionary[0] = torch.zeros(300)
    return embed_dictionary

def get_embeddings(emb_dict, word):
    res= np.zeros(shape=(25,600,300)) # batch_size, pad_seq_length,embed_vec_dim
    
    # print(word)
    for i in range(25):  #batch-size
        for j in range(600):
            res[i,j,:]=emb_dict[int(word[i,j])]
    
    res = torch.from_numpy(res).double()
    return res


# Task 3: Create a TensorDataset and DataLoader

def create_data_loader(encoded_reviews, labels, _batch_size = 32):
    """
    :param encoded_reviews: zero-paddded integer-encoded reviews
    :param labels: integer-encoded labels
    :param batch_size: batch size for training
    :return: DataLoader object
    """

    if type(encoded_reviews) is not np.ndarray or type(labels) is not list:
        sys.exit("Please provide a list to the method")
    """
    COMPLETE THE REST OF THE METHOD
    """
    data = TensorDataset(torch.from_numpy(encoded_reviews),torch.Tensor(labels))
    loader = DataLoader(data,shuffle=True,batch_size=_batch_size)

    #For testing purpose 
    liter= iter(loader)
    sample_x,sample_y = liter.next()
    # print('Sample ip size',sample_x.size())
    # print('Sample ip',sample_x)
    # print('Sample op size',sample_y.size())
    # print('Sample op',sample_y)

    return loader

# Task 4: Define the Baseline model here

# This is the baseline model that contains an embedding layer and an fcn for classification
class BaseSentiment(nn.Module):
    def __init__(self,vocab_size,embedding_dim,hidden_dim,output_size):
        super(BaseSentiment,self).__init__()

        self.emd_dict=open_embed_dict()
        self.em=nn.Embedding(vocab_size,embedding_dim)
        self.fc= nn.Linear(embedding_dim,output_size)
        # self.op=nn.Linear(hidden_dim,output_size)
        self.sigmoid=nn.Sigmoid()

    def forward (self, input_words):
        # batch_size=input_words.size(0)
        # print("Input_size:",input_words.size())
        x=get_embeddings(self.emd_dict,input_words)  
        x=self.fc(x)
        # x=self.op(x)
        x=self.sigmoid(x)

        x = x.view(25,-1)
        x = x[:,-1]

        return x


# Task 5: Define the RNN model here

# This model contains an embedding layer, an rnn and an fcn for classification
class RNNSentiment(nn.Module):
    def __init__(self,model_type,vocab_size,embedding_dim,hidden_dim,output_size,n_layers,isBidirectional,drop_prob=0.5):
        super(RNNSentiment,self).__init__()

        self.vocab_size=vocab_size
        self.embedding_dim=embedding_dim
        self.n_layers=n_layers
        self.hidden_dim=hidden_dim
        self.model_type=model_type

        self.emd_dict=open_embed_dict()
        self.em=nn.Embedding(vocab_size,embedding_dim)

        if model_type==0:
            # print("rnn")
            #RNN
            self.rnn= nn.RNN(embedding_dim, hidden_dim, n_layers, dropout=drop_prob, batch_first=True,bidirectional=False)
            
        elif model_type==1:
            # print("lstm")
            #LSTM
            self.lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers, dropout=drop_prob, batch_first=True,bidirectional=False)
            # dropout layer
            self.dropout = nn.Dropout(0.5)
        elif model_type==2:
            # print("gru")
            #GRU
            self.gru= nn.GRU(embedding_dim, hidden_dim, n_layers, dropout=drop_prob, batch_first=True,bidirectional=False)

        self.fc=nn.Linear(hidden_dim,output_size)
        self.sigmoid=nn.Sigmoid()

    def forward(self, input_words,hidden):
        batch_size=25

        embeds=get_embeddings(self.emd_dict,input_words)  

        if self.model_type==0:
             #RNN
            out,hidden = self.rnn(embeds)
            out = out.contiguous().view(-1, self.hidden_dim)
            out = self.fc(out)
        elif self.model_type==1:
            #LSTM
            out, hidden = self.lstm(embeds, hidden)
            out = out.contiguous().view(-1, self.hidden_dim) # Stack up outputs
            out = self.dropout(out) #dropout and fully-connected layer
        elif self.model_type==2:
            #GRU
            out,hidden = self.gru(embeds)
            out = out.contiguous().view(-1, self.hidden_dim)
            out = self.fc(out)

         

        
        
       

        # sigmoid function
        sig_out = self.sigmoid(out)
        
        # reshape to be batch_size first
        sig_out = sig_out.view(batch_size, -1)
        sig_out = sig_out[:, -1] # get last batch of labels
        
        # return last sigmoid output and hidden state
        return sig_out, hidden

    def init_hidden(self, batch_size):
        ''' Initializes hidden state '''
        # Create two new tensors with sizes n_layers x batch_size x hidden_dim,
        # initialized to zero, for hidden state and cell state of LSTM
        weight = next(self.parameters()).data
        hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_(),
                weight.new(self.n_layers, batch_size, self.hidden_dim).zero_())
        
        return hidden



# Task 6: Define the RNN model here

# This model contains an embedding layer, self-attention and an fcn for classification
class AttentionSentiment(nn.Module):
    def __init__(self,vocab_size,embedding_dim,output_size):
        super(AttentionSentiment,self).__init__()
        self.vocab_size=vocab_size
        self.embedding_dim=embedding_dim
        self.output_size=output_size

        self.emd_dict=open_embed_dict()
        self.em=nn.Embedding(vocab_size,embedding_dim)
        self.attn = nn.MultiheadAttention(embedding_dim, num_heads = 1)

        self.fc=nn.Linear(embedding_dim,output_size)
        self.sigmoid=nn.Sigmoid()

    def forward(self, input_words):
        batch_size=25

        x=get_embeddings(self.emd_dict,input_words)  
        x=self.attn(x,x,x)[0]
        x=self.fc(x)
        x=self.sigmoid(x)

        x = x.view(25,-1)
        x = x[:,-1]

        return x

"""
ALL METHODS AND CLASSES HAVE BEEN DEFINED! TIME TO START EXECUTION!!
"""

# Task 7: Start model training and testing

# Instantiate all hyper-parameters and objects here

# Define loss and optimizer here

# Training starts!!

# Testing starts!!
def baselineModelTrainTest(vocab_size,embedding_dim,hidden_dim,output_size,train_loader,test_loader,validate_loader,num_epochs,lr):
    start=time.time()
    #Instantiate the model 
    model = BaseSentiment(vocab_size,embedding_dim,hidden_dim,output_size)
    print(model)

    # Define loss and optimizer 
    criterion = nn.BCELoss()  #Loss function
    learning_rate=lr
    # print(learning_rate)
    optimizer = optim.SGD(model.parameters(), lr=learning_rate) #Optimizer 

    epochs = num_epochs
    print_every=100
    counter=0

    model=model.double()
    model=model.train()

    for epoch in range(epochs):
        # print("Epoch",epoch)
        for inputs,labels in train_loader:
            counter+=1
            output=model(inputs.double())

            # print(labels.double().size())
            #loss calculation and back propogation
            loss= criterion(output.squeeze(), labels.double())
            loss.backward()
            optimizer.step()

            #Validation 
            if counter % print_every ==0:
                vloss=[]
                model.eval()
                for ip, lbl in validate_loader:
                    op=model(ip.double())
                    vl=criterion(op.squeeze(),lbl.double())
                    vloss.append(vl.item())
                model.train()
                print("Epoch: {}/{}...".format(epoch+1, epochs),
                  "Step: {}...".format(counter),
                  "Loss: {:.6f}...".format(loss.item()),
                  "Val Loss: {:.6f}".format(np.mean(vloss)))
    train_time=time.time()
    #Testing 
    test_losses=[]
    num_correct=0 

    model=model.double()
    model=model.eval()

    for ip,lbl in test_loader:
        # get predicted outputs
        op=model(ip)

        # calculate loss
        test_loss=criterion(op.squeeze(),lbl.double())
        test_losses.append(test_loss.item())

        # convert output probabilities to predicted class (0 or 1)
        pred=torch.round(op.squeeze())

        # compare predictions to true label
        correct_tensor = pred.eq(labels.double().view_as(pred))
        correct = np.squeeze(correct_tensor.numpy())
        num_correct += np.sum(correct)

    # -- stats! -- ##
    # avg test loss
    print("Test loss: {:.3f}".format(np.mean(test_losses)))

    # accuracy over all test data
    test_acc = num_correct/len(test_loader.dataset)
    print("Test accuracy: {:.3f}".format(test_acc))

    test_time=time.time()
    print("Training time: {:.3f}".format((train_time-start)))
    print("Testing time: {:.3f}".format((test_time-train_time)))

def LSTMModelTrainTest(model_type,vocab_size,embedding_dim,hidden_dim,output_size,train_loader,test_loader,validate_loader,n_layers,drop_prob,num_epochs,lr,isBidirectional):
    
    start=time.time()

    #Instantiate the model 
    model = RNNSentiment(model_type,vocab_size,embedding_dim,hidden_dim,output_size,n_layers,isBidirectional,drop_prob)
    print(model)

    # Define loss and optimizer 
    criterion = nn.BCELoss()  #Loss function
    learning_rate = lr
    optimizer = optim.SGD(model.parameters(), lr=learning_rate) #Optimizer 

    epochs = num_epochs
    print_every=50
    counter=0
    clip=5 # gradient clipping 
    batch_size=25

    model=model.double()
    model=model.train()

    
    for epoch in range(epochs):
        # print("Epoch",epoch)
        h=model.init_hidden(batch_size)

        for inputs,labels in train_loader:
            counter+=1

            # Creating new variables for the hidden state, otherwise
            # we'd backprop through the entire training history
            h = tuple([each.data for each in h])
            model.zero_grad()

            output,h=model(inputs.double(),h)

            # print(labels.double().size())
            #loss calculation and back propogation
            loss= criterion(output.squeeze(), labels.double())
            loss.backward()
            optimizer.step()

            #Validation 
            if counter % print_every ==0:
                val_h=model.init_hidden(batch_size)
                vloss=[]
                model.eval()
                for ip, lbl in validate_loader:
                    
                    # Creating new variables for the hidden state, otherwise
                    # we'd backprop through the entire training history
                    val_h = tuple([each.data for each in val_h])

                    op,val_h=model(ip.double(),val_h)
                    vl=criterion(op.squeeze(),lbl.double())
                    vloss.append(vl.item())
                model.train()
                print("Epoch: {}/{}...".format(epoch+1, epochs),
                  "Step: {}...".format(counter),
                  "Loss: {:.6f}...".format(loss.item()),
                  "Val Loss: {:.6f}".format(np.mean(vloss)))
    train_time=time.time()

    #Testing 
    test_losses=[]
    num_correct=0 
    
    h=model.init_hidden(batch_size)

    model=model.double()
    model=model.eval()


    for ip,lbl in test_loader:
        
        # Creating new variables for the hidden state, otherwise
        # we'd backprop through the entire training history
        h = tuple([each.data for each in h])

        # get predicted outputs
        op,h=model(ip,h)

        # calculate loss
        test_loss=criterion(op.squeeze(),lbl.double())
        test_losses.append(test_loss.item())

        # convert output probabilities to predicted class (0 or 1)
        pred=torch.round(op.squeeze())  # rounds to the nearest integer

        # compare predictions to true label
        correct_tensor = pred.eq(labels.double().view_as(pred))
        correct = np.squeeze(correct_tensor.numpy())
        num_correct += np.sum(correct)  

    # -- stats! -- ##
    # avg test loss
    print("Test loss: {:.3f}".format(np.mean(test_losses)))

    # accuracy over all test data
    test_acc = num_correct/len(test_loader.dataset)
    print("Test accuracy: {:.3f}".format(test_acc))

    test_time=time.time()
    print("Training time: {:.3f}".format((train_time-start)))
    print("Testing time: {:.3f}".format((test_time-train_time)))
    
def selfAttentionTrainTest(vocab_size,embedding_dim,output_size,train_loader,test_loader,validate_loader,num_epochs,lr):
    start=time.time()
    #Instantiate the model 
    model = AttentionSentiment(vocab_size,embedding_dim,output_size)
    print(model)

    # Define loss and optimizer 
    criterion = nn.BCELoss()  #Loss function
    learning_rate=lr
    optimizer = optim.SGD(model.parameters(), lr=learning_rate) #Optimizer 

    epochs = num_epochs
    print_every=50
    counter=0

    model=model.double()
    model=model.train()

    for epoch in range(epochs):
        # print("Epoch",epoch)
        for inputs,labels in train_loader:
            counter+=1
            output=model(inputs.double())

            # print(labels.double().size())
            #loss calculation and back propogation
            loss= criterion(output.squeeze(), labels.double())
            loss.backward()
            optimizer.step()

            #Validation 
            if counter % print_every ==0:
                vloss=[]
                model.eval()
                for ip, lbl in validate_loader:
                    op=model(ip.double())
                    vl=criterion(op.squeeze(),lbl.double())
                    vloss.append(vl.item())
                model.train()
                print("Epoch: {}/{}...".format(epoch+1, epochs),
                  "Step: {}...".format(counter),
                  "Loss: {:.6f}...".format(loss.item()),
                  "Val Loss: {:.6f}".format(np.mean(vloss)))
    train_time=time.time()
    #Testing 
    test_losses=[]
    num_correct=0 

    model=model.double()
    model=model.eval()

    for ip,lbl in test_loader:
        # get predicted outputs
        op=model(ip)

        # calculate loss
        test_loss=criterion(op.squeeze(),lbl.double())
        test_losses.append(test_loss.item())

        # convert output probabilities to predicted class (0 or 1)
        pred=torch.round(op.squeeze())

        # compare predictions to true label
        correct_tensor = pred.eq(labels.double().view_as(pred))
        correct = np.squeeze(correct_tensor.numpy())
        num_correct += np.sum(correct)

    # -- stats! -- ##
    # avg test loss
    print("Test loss: {:.3f}".format(np.mean(test_losses)))

    # accuracy over all test data
    test_acc = num_correct/len(test_loader.dataset)
    print("Test accuracy: {:.3f}".format(test_acc))

    test_time=time.time()
    print("Training time: {:.3f}".format((train_time-start)))
    print("Testing time: {:.3f}".format((test_time-train_time)))
