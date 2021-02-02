import torch
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from torch.utils.data import RandomSampler
from torch.utils.data import SequentialSampler
import torch.nn.functional as F 
"""
    Model Architecture
    The architecture of our model for textual entailment is fairly simple. It contains the following layers:
    1. Embedding layer: This layer transforms the integer-encoded representations of the sentences into dense vectors.
    2. Recurrent layer: This is a stacked bi-directional LSTM layer that takes in the vector representation from the Embedding
    layer and outputs another vector.
    3. Fully connected layer: This layer transforms the output of the RNN into a vector of 2 dimensions. (one corresponding
    to each label i.e. Entails and Not Entails)
    A schematic showing the architecture of the model is provided below:

    We definne the forward pass of our network as follows:
    Let t = {t1; t2; :::tn} denote the premise and h = {h1; h2; :::hn} denote the hypothesis. We first obtain the dense vector
    representations for both sentences by passing them through the same embedding layer. Let et = {et1 ; et2 ; :::etn} denote the
    vector representations for the premise and eh = {eh1 ; eh2 ; :::ehn} denote the vector representations for the hypothesis where
    each vector eti and ehi is of dimension d1. Next, we pass the vector representations through the same LSTM to obtain
    temporal sequences rt and rh respectively, each vector having dimension d2. The vectors rt and rh are concatenated together
    to obtain the vector rth Finally, this concatenated representation rth is passed through the fully connected layer to get vector
    fth, with dimension d3 = 2 (this is because we have 2 labels as discussed previously).


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

def word_embedding(premise,hypothesis):
    word_count = 1
    embeddings_dict = {}
    add_to_dic(embeddings_dict,premise)
    add_to_dic(embeddings_dict,hypothesis)

    # print(embeddings_dict)
    return embeddings_dict

def read_xml(file_name,premise,hypothesis,label):
    import xml.etree.ElementTree as ET
    tree = ET.parse(file_name)
    root = tree.getroot()

    count = 0
    for pair in root.findall('pair'):
        label.append(pair.get('value').lower())
        premise.append([x.lower() for x in pair.find('t').text.split()])
        hypothesis.append([x.lower() for x in pair.find('h').text.split()])

        # count += 1
        # if count == 6:
        #     break
    # print(label,premise,hypothesis,sep='\n')

def task1(premise=[],hypothesis=[],label=[]):
    read_xml('train.xml',premise,hypothesis,label)
    embeddings = word_embedding(premise,hypothesis)
  
    #replace words with their embeddings
    premise = [[embeddings[word] for word in sent]  for sent in premise]
    hypothesis = [[embeddings[word] for word in sent] for sent in hypothesis]
    label = [1 if word=='true' else 0 for word in label]

    # print(label,premise,hypothesis,sep='\n')

    max_length = max(max([len(h) for h in hypothesis]),max([len(p) for p in premise]))
    padding(premise,max_length)
    padding(hypothesis,max_length)
    # print(len(label),len(premise),len(hypothesis),max_length)
    return premise,hypothesis,label,embeddings,max_length

class LSTMClassifier(torch.nn.Module):
    def __init__(self, batch_size, output_size, hidden_size, vocab_size, embedding_length, embeddings):
        super(LSTMClassifier, self).__init__()
        
        """
        Arguments
        ---------
        batch_size : Size of the batch which is same as the batch_size of the data returned by the TorchText BucketIterator
        output_size : 2 = (pos, neg)
        hidden_sie : Size of the hidden_state of the LSTM
        vocab_size : Size of the vocabulary containing unique words
        embedding_length : Embeddding dimension of GloVe word embeddings
        weights : Pre-trained GloVe word_embeddings which we will use to create our word_embedding look-up table 
        
        """
        self.batch_size = batch_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.embedding_length = embedding_length
        self.embeddings = embeddings

        self.embed = torch.nn.Embedding(vocab_size, embedding_length)# Initializing the look-up table.
        # self.word_embeddings.weight = torch.nn.Parameter(torch.from_numpy(weights), requires_grad=False) # Assigning the look-up table to the pre-trained GloVe word embedding.
        # self.lstm = torch.nn.LSTM(embedding_length, hidden_size,num_layers=2,bidirectional=True)
        # self.label = torch.nn.Linear(hidden_size, output_size)
        # self.sigmoid = torch.nn.Sigmoid()

        self.lstm = torch.nn.LSTM(self.embedding_length, self.hidden_size, num_layers=2, dropout=0.5,bidirectional=False)
        self.dropout = torch.nn.Dropout(0.5)
        self.fc = torch.nn.Linear(self.hidden_size,self.output_size)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, prem, hypo,batch_size=None):
        """ 
        Parameters
        ----------
        input_sentence: input_sentence of shape = (batch_size, num_sequences)
        batch_size : default = None. Used only for prediction on a single sentence after training (batch_size = 1)
        
        Returns
        -------
        Output of the linear layer containing logits for positive & negative class which receives its input as the final_hidden_state of the LSTM
        final_output.shape = (batch_size, output_size)
        
        """
        ''' Here we will map all the indexes present in the input sequence to the corresponding word vector using our pre-trained word_embedddins.'''
        # # print(prem,hypo,sep='\n')
        input_prem = self.embed(prem)                       # shape = (batch_size, num_sequences,  embedding_length) = (25,61,61)
        input_prem = input_prem.permute(1, 0, 2)            # shape = (num_sequences, batch_size, embedding_length) = (61,25,61)
        if batch_size == None:
            h_0 = torch.zeros(2, self.batch_size, self.hidden_size) #shape = (layers,batch,hidden) = (2,25,256)
            c_0 = torch.zeros(2, self.batch_size, self.hidden_size) #shape = (layers,batch,hidden) = (2,25,256)
        else:
            # print("batch size now=",batch_size)
            h_0 = torch.zeros(2, batch_size, self.hidden_size) #shape = (layers,batch,hidden) = (2,25,256)
            c_0 = torch.zeros(2, batch_size, self.hidden_size) #shape = (layers,batch,hidden) = (2,25,256)
        output_prem, (final_hidden_state, final_cell_state) = self.lstm(input_prem, (h_0, c_0))
        # output_prem.shape = 61,25,256
        # final_hidden_state.shape = 2,25,256
        # final_cell_state.shape = 2,25,256

        input_hypo = self.embed(hypo)
        input_hypo = input_hypo.permute(1, 0, 2)
        if batch_size == None:
            h_0 = torch.zeros(2, self.batch_size, self.hidden_size) #shape = (layers,batch,hidden) = (2,25,256)
            c_0 = torch.zeros(2, self.batch_size, self.hidden_size) #shape = (layers,batch,hidden) = (2,25,256)
        else:
            # print("batch size now=",batch_size)
            h_0 = torch.zeros(2, batch_size, self.hidden_size) #shape = (layers,batch,hidden) = (2,25,256)
            c_0 = torch.zeros(2, batch_size, self.hidden_size) #shape = (layers,batch,hidden) = (2,25,256)
        output_hypo, (final_hidden_state, final_cell_state) = self.lstm(input_hypo, (h_0, c_0))
        # # # print("output hypo= ",output_hypo)
        # # # final_output2 = self.label(final_hidden_state[-1])

        # # # final_output = final_output1 + final_output2
        # # # # print(final_output1,final_output2,final_output)
        h_star = torch.stack([output_prem,output_hypo]) #h_star.shape = (2,61,25,256) , h_star[-1].shape = (61,25,256)
        # print("shape= ",h_star,"-1 shape= ",h_star[-1],sep='\n')

        # # print("hstar =",h_star[-1])
        h_star = self.fc(h_star[-1])
        # h_star = h_star.view(batch_size,-1)
        h_star = h_star.permute(0,2,1)
        h_star = h_star[-1]
        h_star = h_star[-1]
        # print(h_star.shape)
        # sig_out = F.log_softmax(h_star,dim=1)
        # # print(sig_out.shape)    #(61,25,2)
        # sig_out = sig_out.view(batch_size, -1)
        # # print(sig_out.shape)    #(25,61)
        # sig_out = sig_out[:, -1] # get last batch of labels
        # # print(sig_out.shape)    #(25)

        return h_star
        # # sigmoid function
        # sig_out = self.sigmoid(out)
        
        # # reshape to be batch_size first
        # sig_out = sig_out.view(self.batch_size, -1)
        # sig_out = sig_out[:, -1] # get last batch of labels
        
        # # return last sigmoid output and hidden state
        # return sig_out, hidden

if __name__ == "__main__":
    ###task 1 : Prepare dataset###
    premise,hypothesis,label,embeddings,max_length = task1()
    # print(label,premise,hypothesis,embeddings,sep='\n')

    ###task 2 : Preparing the inputs for training/testing ###
    # print(len(premise),len(hypothesis),len(label))
    premise = torch.LongTensor(premise)
    hypothesis = torch.LongTensor(hypothesis)
    label = torch.FloatTensor(label)

    train_ds = TensorDataset(premise, hypothesis,label)

    train_sampler = RandomSampler(train_ds,replacement=False)
    # test_sampler = SequentialSampler(test_ds)
    bsize = 25
    train_dl = DataLoader(train_ds, batch_size=bsize,sampler=train_sampler)
    
    ## Use RandomSampler for training, and SequentialSampler for testing ##
    ###task 3 : Define the model ###
    """
    Create the model, following the architectural specications provided in the previous section. Be careful when defining the
    parameters of each layer in the model. You may want to use the token-integer mapping dictionary saved previously to define
    the size of the embedding layer.
    """
    learning_rate = 2e-5
    batch_size = bsize
    output_size = 2
    hidden_size = 256
    embedding_length = max_length
    vocab_size = len(embeddings)+1
    print(f'batchsize={batch_size}, hidden_size={hidden_size},max_length={max_length},vocab_size={vocab_size}')
    model = LSTMClassifier(batch_size, output_size, hidden_size, vocab_size, embedding_length, embeddings)
    print(model)

    ###task 4 : Train and Test the model ###
    """
    Look into how the model can be trained and tested. Define a suitable loss and optimizer function. Define suitable values for
    different hyper-parameters such as learning rate, number of epochs and batch size. To test the model, you may use scikit-
    learn's classication report to get the precision, recall, f-score and accuracy values. Additionally, also report the throughput
    of your model (in seconds) at the time of inference.
    """
    epochs = 100
    lrate = 2e-5
    lossfn = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(),lr=lrate)
    # optimizer = torch.optim.SGD(model.parameters(),lr=lrate,momentum=0.9)
    
    model=model.double()
    model=model.train()
    
    for epoch in range(epochs):
        # h=model.init_hidden(batch_size)

        for prem,hypo,lab in train_dl:
            # print(prem,hypo,lab,sep='\n')
            # Creating new variables for the hidden state, otherwise
            # we'd backprop through the entire training history
            # h = tuple([each.data for each in h])

            y_pred = model(prem,hypo,batch_size=len(prem))
            # print(y_pred.shape,lab.shape,sep='\n')
            print(y_pred)
            loss = lossfn(lab,y_pred)
            
            loss.backward()
            optimizer.zero_grad()

            optimizer.step()
            break
        # print(epoch,loss)
        break
        
    ###task 5 : Prepare a report ###
    """
    Prepare a report summarizing your results. Specically, observe the effect of hyper-parameters such as number of LSTM
    layers considered, embedding dimension, hidden dimension of the LSTM layers, etc. on model performance and throughput.
    To get a better understanding of how results are analyzed/summarized, consider the reference paper provided on the webpage.
    1. Daniel Z Korman, Eric Mack, Jacob Jett, and Allen H Renear. Defining textual entailment. Journal of the Association
    for Information Science and Technology, 69(6):763{772, 2018.
    2. Ido Dagan, Oren Glickman, and Bernardo Magnini. The pascal recognising textual entailment challenge. In Machine
    Learning Challenges Workshop, pages 177{190. Springer, 2005.
    """


    