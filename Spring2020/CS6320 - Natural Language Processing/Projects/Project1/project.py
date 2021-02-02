import torch
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from torch.utils.data import RandomSampler
from torch.utils.data import SequentialSampler
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report
import torch.nn.functional as F 
import xml.etree.ElementTree as ET
import numpy as np
import time
import argparse
"""
                              Label
                                |
                    Fully Connected Layer
                                |
                  rth = {rth1,rth2,,,,,,rthn}
                    /                   \
                   /                     \
    rt = {rt1,rt2,,,,,,,rtn}        rh = {rh1,rh2,,,,,,rhn}         <--    Recurrent Layer
                |                               |
                |                               |
    et = {et1,et2,,,,,,etn}         ht={ht1,ht2,,,,,htn}            <--    Embedding Layer
                |                               |
                |                               |
    tn = {t1,t2,,,,,tn}             h={h1,h2,,,,,hn}
"""

def padding(listing,max_length):
    for x in listing:
        for i in range(len(x),max_length):
            x.append(0)

def add_to_dic(em_dic,listing):
    word_count = 1 if len(em_dic)==0 else len(em_dic) + 1
    for sent in listing:
        for idx in range(len(sent)):
            if sent[idx] not in em_dic:
                em_dic[sent[idx]] = word_count
                word_count += 1

def word_embedding(premise,hypothesis,embeddings_dict):
    add_to_dic(embeddings_dict,premise)
    add_to_dic(embeddings_dict,hypothesis)

    # print(embeddings_dict)
    return embeddings_dict

def read_xml(file_name):
    premise=[]
    hypothesis=[]
    label=[]
    tree = ET.parse(file_name)
    root = tree.getroot()

    count = 0
    for pair in root.findall('pair'):
        label.append(pair.get('value').lower())
        # premise.append([x.lower() for x in pair.find('t').text.split()])
        # hypothesis.append([x.lower() for x in pair.find('h').text.split()])
        import re
        premise.append([x.lower() for x in re.findall('[\w]+',pair.find('t').text)])
        hypothesis.append([x.lower() for x in re.findall('[\w]+',pair.find('h').text)])

        # count += 1
        # if count == 6:
        #     break
    # print(label,premise,hypothesis,sep='\n')
    return premise,hypothesis,label

def word2int(embeddings,premise,hypothesis,label): 
    #replace words with their embeddings
    premise = [[embeddings[word] for word in sent]  for sent in premise]
    hypothesis = [[embeddings[word] for word in sent] for sent in hypothesis]
    label = [1 if word=='true' else 0 for word in label]

    # print(label,premise,hypothesis,sep='\n')
    return premise,hypothesis,label

class LSTM(torch.nn.Module):
    def __init__(self, batch, output_size, hidden_size, vocab_size, embedding_length, embeddings,n_layers,isBidirectional,drop_prob):
        super(LSTM,self).__init__()

        self.batch = batch
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.embedding_length = embedding_length
        self.embeddings = embeddings
        self.nLayer = n_layers

        self.embed = torch.nn.Embedding(self.vocab_size,self.embedding_length)
        self.lstm = torch.nn.LSTM(self.embedding_length, self.hidden_size, num_layers=n_layers, dropout=drop_prob,bidirectional=isBidirectional) 
        self.fc = torch.nn.Linear(2*hidden_size,output_size)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, input_prem, input_hypo, batch):
        # print(input_prem,input_hypo,batch,sep='\n')
        input_prem = self.embed(input_prem)                 # shape = (batch, num_sequences,  embedding_length) = (25,61,61)
        input_prem = input_prem.permute(1, 0, 2)            # shape = (num_sequences, batch, embedding_length) = (61,25,61)
        
        input_hypo = self.embed(input_hypo)                 # shape = (batch, num_sequences,  embedding_length) = (25,61,61)
        input_hypo = input_hypo.permute(1, 0, 2)            # shape = (num_sequences, batch, embedding_length) = (61,25,61)

        layer_multiple = self.nLayer
        h_0 = torch.randn(2*layer_multiple, batch, self.hidden_size) #shape = (layers,batch,hidden) = (4,25,256)
        c_0 = torch.randn(2*layer_multiple, batch, self.hidden_size) #shape = (layers,batch,hidden) = (4,25,256)
        # print(h_0.shape,c_0.shape)

        output_prem, (final_hidden_state, final_cell_state) = self.lstm(input_prem, (h_0, c_0))
        # output_prem.shape = 61,25,512
        # final_hidden_state.shape = 2,25,512
        # final_cell_state.shape = 2,25,512
        # print("shape= ",output_prem.shape)

        output_hypo, (final_hidden_state, final_cell_state) = self.lstm(input_hypo, (h_0, c_0))
        # output_hypo.shape = 61,25,512

        h_star = torch.cat((output_prem,output_hypo))
        # print(output_prem,output_hypo,h_star,sep='\n')
        # print("shape= ",h_star.shape)
        #h_star.shape = (122,25,512)
        h_star = self.fc(h_star)
        #h_star.shape = (6100,2)
        # print("shape= ",h_star.shape)

        # sigmoid function
        sig_out = self.sigmoid(h_star)

        # reshape to be batch first
        sig_out = sig_out.view(batch, -1)
        sig_out = sig_out[:, -1] # get last batch of labels
        
        return sig_out

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='NLP Textual Entailment using LSTM')
    parser.add_argument("--epochs",default=5,type=int,help="Give the number of epochs, default = 5")
    parser.add_argument("--learningrate",default=1e-3,type=float,help="Learning rate for model, default = 0.001")
    parser.add_argument("--batch_size",default=25,type=int,help="Batch size, default = 25")
    parser.add_argument("--dropout",default=0.3,type=float,help="Dropout for LSTM, default = 0.3")
    parser.add_argument("--bidirectional",default=True,type=bool,help="LSTM bidirectional?, default = True")
    parser.add_argument("--output_size",default=2,type=int,help="Final output size, default = 2 -> (Entails,Not Entails)")
    parser.add_argument("--hidden_size",default=256,type=int,help="Hidden inputs size for LSTM model, default = 256")
    parser.add_argument("--nLayers",default=2,type=int,help="#Layers for LSTM, default = 2")

    args = parser.parse_args()

    epochs = args.epochs
    lrate = args.learningrate
    batch = args.batch_size
    output_size = args.output_size
    hidden_size = args.hidden_size    
    nLayers = args.nLayers     # 1 and 4 are giving same or lower accuracies
    isBidirectional = args.bidirectional
    dropout = args.dropout

    torch.manual_seed(0)
    
    ###task 1 : Prepare dataset###
    premise,hypothesis,label = read_xml('train.xml')
    premise_test,hypothesis_test,label_test = read_xml('test.xml')

    # print(len(label),len(premise),len(hypothesis))
    # print(len(label_test),len(premise_test),len(hypothesis_test))
    
    embeddings = {}
    embeddings = word_embedding(premise,hypothesis,embeddings)
    embeddings = word_embedding(premise_test,hypothesis_test,embeddings)
    # print(embeddings)

    premise,hypothesis,label = word2int(embeddings,premise,hypothesis,label)
    premise_test,hypothesis_test,label_test = word2int(embeddings,premise_test,hypothesis_test,label_test)

    real_test = label_test
    # print(len(label),len(premise),len(hypothesis))
    # print(len(label_test),len(premise_test),len(hypothesis_test))

    max_length_train = max(max([len(h) for h in hypothesis]),max([len(p) for p in premise]))
    max_length_test = max(max([len(h) for h in hypothesis_test]),max([len(p) for p in premise_test]))

    max_length = max(max_length_train,max_length_test)
    # print(f'max_length={max_length},train_maxl={max_length_train},test_maxl={max_length_test}')

    padding(premise,max_length)
    padding(premise_test,max_length)
    padding(hypothesis,max_length)
    padding(hypothesis_test,max_length)
    # print(len(label),len(premise),len(hypothesis),max_length_train)
    # print(len(label_test),len(premise_test),len(hypothesis_test),max_length_test)

    # print(label,premise,hypothesis,embeddings,sep='\n')
    

    ###task 2 : Preparing the inputs for training/testing ###
    # print(len(premise),len(hypothesis),len(label))
    premise = torch.LongTensor(premise)
    hypothesis = torch.LongTensor(hypothesis)
    label = torch.FloatTensor(label)

    premise_test = torch.LongTensor(premise_test)
    hypothesis_test = torch.LongTensor(hypothesis_test)
    label_test = torch.FloatTensor(label_test)

    # print(len(label),len(premise),len(hypothesis),max_length_train)
    # print(len(label_test),len(premise_test),len(hypothesis_test),max_length_test)

    train_ds = TensorDataset(premise, hypothesis,label)
    test_ds = TensorDataset(premise_test, hypothesis_test,label_test)

    train_sampler = RandomSampler(train_ds,replacement=False)
    test_sampler = SequentialSampler(test_ds)
    
    train_dl = DataLoader(train_ds, batch_size=batch,sampler=train_sampler)
    test_dl = DataLoader(test_ds,batch_size=batch,sampler=test_sampler)
    # print(len(train_dl.dataset),len(test_dl.dataset))
    ## Use RandomSampler for training, and SequentialSampler for testing ##
    ###task 3 : Define the model ###
    vocab_size = len(embeddings) + 1
    embedding_length = max_length
    print(f'Parameters : batchsize = {batch}, learning rate = {lrate} , hidden_size = {hidden_size}, max_length = {max_length}, vocab_size = {vocab_size}')
    
    model = LSTM(batch, output_size, hidden_size, vocab_size, embedding_length, embeddings, nLayers, isBidirectional, dropout)
    print(model)

    ###task 4 : Train and Test the model ###
    lossfn = torch.nn.BCELoss()
    # optimizer = torch.optim.Adam(model.parameters(),lr=lrate) #About 50% accuracy
    # optimizer = torch.optim.Adagrad(model.parameters(),lr=lrate) #Faster but About 48.4%
    optimizer = torch.optim.SGD(model.parameters(),lr=lrate,momentum=0.8) # About 52.4% accuracy
    
    model=model.train()
    train_start=time.time()

    for epoch in range(epochs):
        loss_per_epoch = []
        for prem,hypo,lab in train_dl:
            # print(prem,hypo,lab,sep='\n')

            y_pred = model(prem,hypo,len(prem))
            # print(y_pred.shape,lab.shape,sep='\n')
            # print(y_pred)
            loss = lossfn(y_pred,lab)
            loss_per_epoch.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
           
        # print(y_pred,epoch,loss.item(),sep='\n')
        print(f'Epoch:{epoch}, Loss: {np.mean(loss_per_epoch):0.5f}')
    
    train_end = time.time()

    print(f'Training time = {(train_end - train_start):0.2f} seconds')

    model = model.eval()
    test_losses = []
    num_correct = 0 
    test_output = np.array([])
    test_sample = np.array([])

    for prem,hypo,lab in test_dl:
        
        # get predicted outputs
        op = model(prem,hypo,len(prem))

        # calculate loss
        test_loss = lossfn(op,lab)
        test_losses.append(test_loss.item())

        # convert output probabilities to predicted class (0 or 1)
        pred = torch.round(op)  # rounds to the nearest integer

        correct_indices = pred.detach().numpy() == lab.numpy()
        num_correct += np.sum(correct_indices)
        # print("correct=",num_correct,"pred_len=",len(pred),"lab_len=",len(lab))
        
        test_sample = np.append(test_sample,lab.numpy())
        test_output = np.append(test_output,pred.detach().numpy())

        test_sample = test_sample.astype(int)
        test_output = test_output.astype(int)

        # print("checking=",np.sum(output == test_sample))
    
    test_end = time.time()

    # avg test loss
    print(f"Test loss: {np.mean(test_losses):0.5f}")
    print(f'Testing time = {(test_end-train_end):0.2f} seconds')

    # print(f'total test sets = {len(test_sample)},num_correct = {num_correct}')
    # accuracy over all test data
    # test_acc = num_correct/len(test_sample)
    # print("Test accuracy: {:.3f}".format(test_acc))

    print(f'Test Accuracy: {accuracy_score(test_sample,test_output)*100:0.3f}%')
    
    print(classification_report(test_sample, test_output, target_names=['ENTAIL','NOT ENTAIL']))
    print("Confusion Matrix:")
    print(confusion_matrix(test_sample, test_output))
