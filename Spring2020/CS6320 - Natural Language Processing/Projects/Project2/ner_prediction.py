# -*- coding: utf-8 -*-
"""NER_Prediction.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1Oc4ZKucV_Y8BIlPEENjo6GWtevnK-rKz

The project uses a Google Colaboratory .ipynb file, since the computing power required was'nt met with my device.
The project successfully ran on a hosted runtime having 35GB RAM and TPU selected.
(One hot encoding takes a lot of memory.)

All the required dependecies of the project can be included by running the cells of the notebook.
You can run the notebook by pressing Ctrl+F9 or by clicking 'Run All' from the 'Runtime' menu.
This notebook can be accessed from here: https://colab.research.google.com/drive/1Oc4ZKucV_Y8BIlPEENjo6GWtevnK-rKz
"""

import numpy as np
import time
import os
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report

import nltk
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tag import pos_tag

import spacy
!python -m spacy download en_core_web_sm

"""# Function Definitions"""

def get_pos_lemmas(tagger,sentence):
    # ex = 'The horse will race tomorrow. Race for outer space. Secretariat is expected to race tomorrow.'
    # ex1 = 'Race for outer space.The horse will race tomorrow.'
    # print("sentence is",sentence)
    if(tagger == 'nltk'):
        lemmatizer = WordNetLemmatizer()
        result = [pos_tag(sentence),[lemmatizer.lemmatize(word) for word in sentence]]
        return result
    elif(tagger == 'spacy'):   
        doc = nlp_spacy(" ".join(sentence)) 
        return [[token.pos_ for token in doc],[token.lemma_ for token in doc]]

def get_lemmas_based_pos_ner(tagger,file, isTest = False, existingVocab={},store_test_tokens_list=[]):
    count_lines = 0
    sentences = [[]]
    sentences_ner = [[]]
    sentence_counter = -1

    tokens_ner_dict = {}
    tokens_pos_dict = {}

    def increment_sentence_counter(counter):
        sentences.append([])
        sentences_ner.append([])
        return counter + 1

    with open(file) as f:
        # while(count_lines < limit):
        while True:
            line = f.readline()
            count_lines += 1

            #end of file, len(line) is 0 OR line is null
            if not line:    
                break

            if line.find("-DOCSTART-") != -1:
                # print(count, line, "Starting new document")
                continue

            #for new lines, len(line) == 1 due to '\n' character
            if line == "\n":
                #skip for the first time
                if sentence_counter < 0:
                    sentence_counter = increment_sentence_counter(sentence_counter)
                    continue

                # print(count, "New Sentence next")
                #Not seperating pos and lemma functions as we need from the same source of 'nltk' or 'spacy' or 'stanford'
                pos_tags, lemmas = get_pos_lemmas(tagger,sentences[sentence_counter])

                #now we got the correct pos_tags and lemmas for the sentence, now we have to store them
                for pos,lemma,ner in zip(pos_tags,lemmas,sentences_ner[sentence_counter]):
                    pos_tag = pos if tagger == 'spacy' else pos[1]
                    if isTest:
                        #lemma not in tokens_pos_dict is also not in tokens_ner_dict
                        if lemma not in existingVocab:
                            tokens_pos_dict[lemma] = 'UNK'
                            tokens_ner_dict[lemma] = {'UNK':1}  
                        elif lemma not in tokens_pos_dict:
                            tokens_pos_dict[lemma] = pos_tag
                            tokens_ner_dict[lemma] = {ner : 1}
                        else:
                            if ner in tokens_ner_dict[lemma]:
                                tokens_ner_dict[lemma][ner] += 1
                            else:
                                tokens_ner_dict[lemma].update({ner : 1})
         
                    else:
                        if lemma not in tokens_pos_dict:
                            tokens_pos_dict[lemma] = pos_tag
                            tokens_ner_dict[lemma] = {ner : 1}
                        else:
                            if ner in tokens_ner_dict[lemma]:
                                tokens_ner_dict[lemma][ner] += 1
                            else:
                                tokens_ner_dict[lemma].update({ner : 1})
                                                      
                # print("sentence no.",sentence_counter)
                sentence_counter = increment_sentence_counter(sentence_counter)        
                continue

            token, ner = line.split()
            token_lower = token.lower()
            if isTest:
                store_test_tokens_list.append(token_lower)
            sentences[sentence_counter].append(token_lower)
            sentences_ner[sentence_counter].append(ner)

    # print(tokens_ner_dict)
    #get the most used ner tag for that lemma
    for lemma in tokens_ner_dict:
        tokens_ner_dict[lemma] = max(zip(tokens_ner_dict[lemma].values(),tokens_ner_dict[lemma].keys()))[1]

    unique_pos_tags = list(np.unique(list(tokens_pos_dict.values())))
    print(f"There are {sentence_counter} sentences, {len(tokens_pos_dict)} unique tokens, and {len(unique_pos_tags)} POS Tags in the '{file}' corpus.")
    return tokens_pos_dict, tokens_ner_dict

def extract_pos_lemma_ner(lemmas_mapped_pos,lemmas_based_ners):
    unique_pos_tags = list(np.unique(list(lemmas_mapped_pos.values())))
    lemmas = list(lemmas_mapped_pos.keys())
    ners = list(lemmas_based_ners.values())
    # print(unique_pos_tags,lemmas,ners)
    return unique_pos_tags,lemmas,ners

def make_integer_mappings(train,test):
    mapped_integer = {}
    uid = 0

    for tr in train:
        mapped_integer[tr] = uid
        uid += 1
    
    unkwown_tag_uid = uid
    mapped_integer['UNK'] = unkwown_tag_uid
    uid += 1

    for te in test:
      if te not in mapped_integer:
        mapped_integer[te] = uid
        uid += 1

    # print(mapped_integer)
    return mapped_integer

def merge_encoded_lemma_pos(lemma_pos,lemma_int_mapping,pos_int_mapping,one_hot_lemmas,one_hot_pos_tags):
    
    merged_lemma_pos = {}
    for lemma in lemma_pos:
        lemma_idx = lemma_int_mapping[lemma]
        pos_idx = pos_int_mapping[lemma_pos[lemma]]

        merged_lemma_pos[lemma] = np.hstack((one_hot_lemmas[lemma_idx],one_hot_pos_tags[pos_idx]))

    # print(merged_lemma_pos)
    return merged_lemma_pos

"""# Task 1 : Data Preprocessing"""

#Task 1 : Data preprocessing#
train_file = 'modified_train.txt'
test_file = 'modified_test.txt'

#available : 'nltk' and 'spacy'
tagger = 'spacy'

if tagger == 'spacy':
    nlp_spacy = spacy.load("en_core_web_sm") 

test_tokens_list = []

train_lemma_mapped_pos, train_lemma_mapped_ner = get_lemmas_based_pos_ner(tagger,train_file)
test_lemma_mapped_pos, test_lemma_mapped_ner = get_lemmas_based_pos_ner(tagger,test_file,True, train_lemma_mapped_pos,test_tokens_list)
# print(train_lemma_mapped_pos,train_lemma_mapped_ner)
# print(test_lemma_mapped_pos,test_lemma_mapped_ner)

train_upos_tags, train_lemmas, train_ners = extract_pos_lemma_ner(train_lemma_mapped_pos,train_lemma_mapped_ner)
test_upos_tags, test_lemmas, test_ners = extract_pos_lemma_ner(test_lemma_mapped_pos,test_lemma_mapped_ner)

#Get the lemma and pos mapping based on the training and testing data
lemma_mapped_integer = make_integer_mappings(train_lemmas,test_lemmas)
pos_mapped_integer = make_integer_mappings(train_upos_tags,test_upos_tags)
# unknown_tag_counter = lemma_unknown + pos_unknown

"""# Task 2 : Feature Engineering"""

#Task 2 : Feature Engineering#

lemma_dimension = len(lemma_mapped_integer)
pos_dimension = len(pos_mapped_integer)
print(lemma_dimension,pos_dimension)
# print(lemma_mapped_integer.keys())

one_hot_lemmas = np.eye(lemma_dimension)
one_hot_pos_tags = np.eye(pos_dimension)

train_one_hot_lemma_pos_encoded = merge_encoded_lemma_pos(train_lemma_mapped_pos,lemma_mapped_integer,pos_mapped_integer,one_hot_lemmas,one_hot_pos_tags)
test_one_hot_lemma_pos_encoded = merge_encoded_lemma_pos(test_lemma_mapped_pos,lemma_mapped_integer,pos_mapped_integer,one_hot_lemmas,one_hot_pos_tags)
# print(one_hot_lemma_pos_encoded)

"""# Task 3 : Learning"""

#Task 3: Learning#
train_input = list(train_one_hot_lemma_pos_encoded.values())
train_output = list(train_lemma_mapped_ner.values())
print(len(train_input),len(train_output))

from sklearn.naive_bayes import GaussianNB
model = GaussianNB()

# from sklearn.ensemble import RandomForestClassifier
# model = RandomForestClassifier()

# from sklearn.linear_model import LogisticRegression
# model = LogisticRegression()

train_start=time.time()
model.fit(train_input,train_output)
train_end = time.time()

"""# Task 4 : Model Performance"""

# Task 4 : Model Performance
test_input = list(test_one_hot_lemma_pos_encoded.values())
test_output = list(test_lemma_mapped_ner.values())
print(len(test_input),len(test_output))
test_start = time.time()
predicted = model.predict(test_input)
test_end = time.time()

training_time = train_end - train_start
testing_time = test_end - test_start
print(f'Training time = {training_time:0.2f} seconds')
print(f'Testing time = {testing_time:0.2f} seconds')
print(f'Model Accuracy: {accuracy_score(predicted,test_output)*100:0.3f}%')
print(f'Model Throughput : {os.stat(test_file).st_size/(testing_time*1000)}')

#Use the predicted NER tags from the model for all the words of the test
lemma_id = {}
idx = 0
for lemma in test_one_hot_lemma_pos_encoded:
    lemma_id[lemma] = idx
    idx += 1

iter = 0
test_token_list_output = []
for token in test_tokens_list:
    if tagger == 'nltk':
        lemmatizer = WordNetLemmatizer()
        lemma = lemmatizer.lemmatize(token)
    else:
        lemma = nlp_spacy(token)[0].lemma_
    if  lemma in test_lemma_mapped_ner:
        idx = lemma_id[lemma]
        test_token_list_output.append(predicted[idx])
    else:
        test_token_list_output.append('UNK')

"""## Remove BIO-tag Violations for the predicted NER tags, if any."""

#Remove the BIO Tag Violations
idx = 0
while idx < len(test_token_list_output):
    spill = test_token_list_output[idx].split('-')
    curr_idx = idx
    while idx >= 0 and len(spill) > 1 and spill[0] == 'I':
        spill_up = test_token_list_output[idx-1].split('-')
        # print(idx,test_tokens_list[idx],test_token_list_output[idx],spill,spill_up)

        if spill_up[0] == 'O':
            test_token_list_output[idx] = "B-"+spill[1]
            break
        elif spill_up[0] == 'B' and spill_up[1] == spill[1]:
            break
        elif spill_up[0] == 'B':
            test_token_list_output[idx] = "I-"+spill_up[1]
            break
        elif len(spill_up) > 1 and spill_up[1] != spill[1]:
            test_token_list_output[idx] = "I-"+spill_up[1]
        
        spill = spill_up
        idx -= 1

    idx = curr_idx + 1

actual = []
output = []
iter = 0
for token in test_tokens_list:
    if tagger == 'nltk':
        lemmatizer = WordNetLemmatizer()
        lemma = lemmatizer.lemmatize(token)
    else:
        lemma = nlp_spacy(token)[0].lemma_
    if  lemma in test_lemma_mapped_ner:
        idx = lemma_id[lemma]
        # print(iter,idx,token,test_token_list_output[iter],test_lemma_mapped_ner[lemma])
        output.append(test_token_list_output[iter])
        actual.append(test_lemma_mapped_ner[lemma])
    iter += 1

"""# Final Accuracy and Confusion Matrix"""

print(f'NER Test Token Accuracy: {accuracy_score(output,actual)*100:0.3f}%')
print(classification_report(output,actual))
print("Confusion Matrix:")
print(confusion_matrix(output,actual))

"""# Analysis Report

There are two kinds of accuracy reported in this:
1. Model accuracy : This is the accuracy for the model being used for the feature 1-hot encodings for unique tokens.
2. NER Test Token Accuracy : This is the accuracy of all the test tokens, using the above models predicted NER tags, and then calculating the accuracy metric and F1 score with that metric. (Also removing the BIO-tag violations)

The best accuracy to be recorded is with GaussianNB ML algorithm, being 87.096%(NER Test Token Accuracy, Model Accuracy	: 57.352%), and it used the 'spacy' library for its POS tagging and lemmatization tasks.

The model can use either 'nltk' OR the 'spacy' library by changing the 'tagger' variable.
For eg: tagger = 'nltk'
or tagger = 'spacy'

*Throughput - Size of testing file / Time taken for making prediction

### Using NLTK library for POS tags and lemmatization:

```
With GaussianNB :
    Training time 			    :	10.03 seconds
    Testing time 			      :	8.46 seconds
    Model Accuracy 			    :	59.558%
    Model Throughput 		    :	44.876647561548765 Kbps
    NER Test Token Accuracy :	86.427%
```

	
				  precision    recall  f1-score   support

		   B-LOC       0.88      0.81      0.84      1721
		  B-MISC       0.78      0.14      0.23      3467
		   B-ORG       0.73      0.71      0.72      1244
		   B-PER       0.88      0.83      0.85      1068
		   I-LOC       0.24      0.12      0.16       389
		  I-MISC       0.25      0.04      0.06       945
		   I-ORG       0.23      0.48      0.31       279
		   I-PER       0.46      0.28      0.35       658
			   O       0.99      0.98      0.99     36660
			 UNK       0.00      0.00      0.00         0

		accuracy                           0.86     46431
	   macro avg       0.55      0.44      0.45     46431
	weighted avg       0.93      0.86      0.88     46431

	Confusion Matrix:
	[[ 1389     6   155    22    66     0    40     4    39     0]
	 [    6   471    32     1    10    70    13     2    88  2774]
	 [   62    14   886    11     7     0   159    20    85     0]
	 [   10     8    23   885     7     0     8   113    14     0]
	 [    0     6     1     0    48     0    61     0    12   261]
	 [    2     0     1     7    20    35    37    41    29   773]
	 [    0     0     2     0    16     0   133     4     8   116]
	 [    0     1     0    24    12     0     3   181     1   436]
	 [  108    96   113    56    10    34   117    25 36101     0]
	 [    0     0     0     0     0     0     0     0     0     0]]
 
 
Other Machine Learning models performed lesser than the GaussianNB in order

	
```
With RandomForestClassifier:
	Training time 				: 	1363.65 seconds
	Testing time 				: 	9.86 seconds
	Model Accuracy				:	55.531%
	Model Throughput 			: 	38.473305967179336 Kbps
	NER Test Token Accuracy		: 	84.179%

With LogisticRegression:
	Training time 				:	35.96 seconds
	Testing time 				: 	0.73 seconds
	Model Accuracy				: 	48.770%
	Model Throughput 			: 	516.8888562358712 Kbps
	NER Test Token Accuracy		: 	78.924%

With MultinomialNB:
	Training time 				: 	2.30 seconds
	Testing time 				: 	0.75 seconds
	Model Accuracy				: 	46.744%
	Model Throughput 			: 	502.7249905190382 Kbps
	NER Test Token Accuracy		: 	78.351%

With BernoulliNB:
	Training time 				: 	6.41 seconds
	Testing time 				:	1.96 seconds
	Model Accuracy				:	46.744%
	Model Throughput 			:	193.69095921814971 Kbps
	NER Test Token Accuracy		:	78.346%
```

SVM was taking very, very long to train, so I did'nt continue with that.
```
							GaussianNB		BernoulliNB		MultinomialNB		LogisticRegression		RandomForestClassifier
Training Time				10.03 s			6.41s			2.3s				35.96s					1363.65s
Testing Time				 8.46s			  1.96s			0.75s				0.73s					9.86s
NER Test Token Accuracy	  86.427%			78.34%	   	78.351%				78.924%				84.179%
Model Throughput			 44.87 Kbps		193.69 Kbps	   502.72 Kbps			516.88 Kbps			38.57 Kbps
```

Overall analysis:
GaussianNB and RandomForestClassifier gave good results, though the training time for RandomForestClassifier was long.

### Using Spacy library for POS tags and lemmatization:

```
With GaussianNB :
	Training time 				: 	6.68 seconds
	Testing time 				:	8.36 seconds
	Model Accuracy				: 	57.352%
	Model Throughput 			: 	45.39114866835412 Kbps
	NER Test Token Accuracy		:	87.096%
```

		   precision    recall  f1-score   support

		   B-LOC       0.79      0.64      0.71      1614
		  B-MISC       0.76      0.15      0.25      2924
		   B-ORG       0.71      0.73      0.72      1173
		   B-PER       0.76      0.64      0.69       980
		   I-LOC       0.28      0.27      0.28       159
		  I-MISC       0.21      0.03      0.06       785
		   I-ORG       0.24      0.62      0.34       206
		   I-PER       0.33      0.26      0.29       564
			   O       0.98      0.98      0.98     37517
			 UNK       0.00      0.00      0.00         0

		accuracy                           0.87     45922
	   macro avg       0.51      0.43      0.43     45922
	weighted avg       0.92      0.87      0.88     45922

	Confusion Matrix:
	[[ 1029     5   133    17    70     0   111     8   241     0]
	 [    4   432    25     5     0    68    13     0    60  2317]
	 [   53    10   853    19     3     0   113    16   106     0]
	 [   17     7    19   623     2     0     7   191   114     0]
	 [    0     6     0     0    43     0    32     1     7    70]
	 [    2     0     7     9    11    27    26    29    19   655]
	 [    0     0     4     0     3     0   128     3    15    53]
	 [    2     0     0    20    10     0     4   148    11   369]
	 [  192   110   167   126    10    36   106    57 36713     0]
	 [    0     0     0     0     0     0     0     0     0     0]]


```
With RandomForestClassifier:
	Training time = 728.77 seconds
	Testing time = 6.92 seconds
	Model Accuracy: 55.720%
	Model Throughput : 54.85789999684634
	NER Test Token Accuracy: 85.922%

With LogisticRegression:
	Training time 				: 	23.21 seconds
	Testing time 				: 	0.62 seconds
	Model Accuracy				: 	53.477%
	Model Throughput 			: 	611.1674115376123
	NER Test Token Accuracy		: 	84.010%
	
With MultinomialNB:
	Training time 				: 	2.42 seconds
	Testing time 				: 	0.81 seconds
	Model Accuracy				: 	47.034%
	Model Throughput 			: 	468.5071185268847 Kbps
	NER Test Token Accuracy		:	81.194%

With BernoulliNB:
	Training time 				: 	5.78 seconds
	Testing time 				: 	1.18 seconds
	Model Accuracy				: 	47.034%
	Model Throughput 			: 	320.9877634215475
	NER Test Token Accuracy		: 	81.194%
```
```
								GaussianNB		BernoulliNB		MultinomialNB		LogisticRegression		RandomForestClassifier
Training Time					6.68 s			5.78s			2.42s				23.21s					728.77s
Testing Time					 8.36s			1.18s			 0.81s				0.62s					 6.92s
NER Test Token Accuracy		 87.096%			81.194%		  78.35%				84.01%				  85.922%
Model Throughput				45.39 Kbps		320.98 Kbps	   468.51 Kbps		  611.16 Kbps			  54.85 Kbps
```

### Closing Notes

Overall Spacy seems to perform better than 'nltk' with the above ML algorithms for the NER prediction task on the given datasets.
However, 'nltk' was much faster to process the 'POS' and to lemmatize a token in comparison to 'spacy'.

GaussianNB algorithm for Naive bayes and RandomForestClassifier for the Ensemble method, both using 'spacy' library performs good on the NER classification task.
"""