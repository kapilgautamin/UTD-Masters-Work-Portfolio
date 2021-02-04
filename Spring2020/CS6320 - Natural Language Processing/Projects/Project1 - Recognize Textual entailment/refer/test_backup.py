import os 
import sys
from modelrunner import *

import pandas as pd
import matplotlib.pyplot as plt
import argparse


# sys.stdout=open('./output.txt', 'w')

def run():
    # Task 1 : Load the data 

    path =os.path.join(os.getcwd(),"movie_reviews.tar.gz")
    reviews_train,labels_train,reviews_test,labels_test = read_file(path)

    # print(reviews_train,labels_train,reviews_test,labels_test)

    combine_reviews=reviews_train.copy()
    combine_reviews.extend(reviews_test)
    vocabulary = preprocess(combine_reviews)
    # print(vocabulary)
    
    encoded_reviews = encode_review(vocabulary,reviews_train)
    encoded_reviews_test=encode_review(vocabulary,reviews_test)

    # print(encoded_reviews[:5])
    encoded_labels = encode_labels(labels_train)
    encoded_labels_tv=encode_labels(labels_test)
    
    tv_ind=encoded_labels_tv.index(0)
    stv_ind=int(tv_ind/2)
    # print(encoded_labels)

    reviews_len= [len(w) for w in encoded_reviews]
    
    # pd.Series(reviews_len).hist()
    # plt.show()
    # pd.Series(reviews_len).describe()

    seq_length=600
    padded_reviews = pad_zeros(encoded_reviews,seq_length)
    padded_reviews_tv=pad_zeros(encoded_reviews_test,seq_length)

    padded_reviews_test=np.concatenate((padded_reviews_tv[:stv_ind],padded_reviews_tv[tv_ind:tv_ind+stv_ind]))
    padded_reviews_val=np.concatenate((padded_reviews_tv[stv_ind:tv_ind],padded_reviews_tv[tv_ind+stv_ind:]))
    encoded_labels_test=encoded_labels_tv[:stv_ind]+encoded_labels_tv[tv_ind:tv_ind+stv_ind]
    encoded_labels_val=encoded_labels_tv[stv_ind:tv_ind]+encoded_labels_tv[tv_ind+stv_ind:]

    # print(padded_reviews_test,encoded_labels_test)
    # print(padded_reviews_val,encoded_labels_val)
    # print(padded_reviews)

    #Task 2 : Building the embedding dictionary 
    vector_file_path='wiki-news-300d-1M.vec'
    embedding_dictionary = load_embedding_file(vector_file_path,vocabulary)
    # embedding_dictionary=open_embed_dict()
    

    # print(embedding_dictionary)

    # Task 3: Create a TensorDataset and DataLoader
    dl=create_data_loader(padded_reviews,encoded_labels,25)
    dlt=create_data_loader(padded_reviews_test,encoded_labels_test,25)
    dlv=create_data_loader(padded_reviews_val,encoded_labels_val,25)
    # print(dl,dlt,dlv)
    
    parser = argparse.ArgumentParser(description='NLP Sentiment Analysis')
    parser.add_argument("--model",default="baseline",type=str)
    parser.add_argument("--epochs",default=5,type=int)
    parser.add_argument("--learningrate",default=0.01,type=float)
    parser.add_argument("--bidirectional",default=0,type=int)

    args=parser.parse_args()

    #Baseline Model 
    output_size = 1
    embedding_dim = 300
    hidden_dim = 256
    vocab_size=len(vocabulary)+1

    num_layers = 2
    drop_prob = 0.5
    isBidirectional= False if args.bidirectional == 0 else True

    model_type =args.model
    num_epochs= args.epochs
    learning_rate=args.learningrate

    if model_type=="baseline":
        baselineModelTrainTest(vocab_size,embedding_dim,hidden_dim,output_size,dl,dlt,dlv,num_epochs,learning_rate)
    elif model_type=="rnn":
        LSTMModelTrainTest(0,vocab_size,embedding_dim,hidden_dim,output_size,dl,dlt,dlv,num_layers,drop_prob,num_epochs,learning_rate,isBidirectional)
    elif model_type=="lstm":
        LSTMModelTrainTest(1,vocab_size,embedding_dim,hidden_dim,output_size,dl,dlt,dlv,num_layers,drop_prob,num_epochs,learning_rate,isBidirectional)
    elif model_type=="gru":
        LSTMModelTrainTest(2,vocab_size,embedding_dim,hidden_dim,output_size,dl,dlt,dlv,num_layers,drop_prob,num_epochs,learning_rate,isBidirectional)
    elif model_type=="selfattention":
        selfAttentionTrainTest(vocab_size,embedding_dim,output_size,dl,dlt,dlv,num_epochs,learning_rate)
    else:
        print("Please select a valid model. Taking baseline as default for now")

    

    # baselineModelTrainTest(vocab_size,embedding_dim,hidden_dim,output_size,dl,dlt,dlv)
    
    
    # LSTMModelTrainTest(vocab_size,embedding_dim,hidden_dim,output_size,dl,dlt,dlv,num_layers,drop_prob)

    # selfAttentionTrainTest(vocab_size,embedding_dim,output_size,dl,dlt,dlv)

if __name__=="__main__":
    run()
    