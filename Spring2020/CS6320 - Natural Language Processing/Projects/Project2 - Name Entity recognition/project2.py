import numpy as np
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tag import pos_tag 
import spacy
# python -m spacy download en_core_web_sm 

def get_pos_lemmas(tagger,sentence):
    ex = 'The horse will race tomorrow. Race for outer space. Secretariat is expected to race tomorrow.'
    # ex1 = 'Race for outer space.The horse will race tomorrow.'
    # print("sentence is",sentence)
    if(tagger == 'nltk'):
          
        lemmatizer = WordNetLemmatizer()
        result = [pos_tag(sentence),[lemmatizer.lemmatize(word) for word in sentence]]
        # print(result)
        return result
    elif(tagger == 'spacy'):   
         
        doc = nlp(" ".join(sentence)) 
        result = [[token.pos_ for token in doc],[token.lemma_ for token in doc]] 
        # print(result)
        return result
    elif(tagger == 'stanford'):
        from nltk.parse import CoreNLPParser
        pos_tagger = CoreNLPParser(url='http://localhost:9000',tagtype='pos')
        lemmatizer = WordNetLemmatizer()
        result = [list(pos_tagger.tag(sentence)),[lemmatizer.lemmatize(word) for word in sentence]]
        return result


def get_lemmas_based_pos_ner(file, isTest = False, existingVocab={}):
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
        while(count_lines < 100):
        # while True:
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
                tagger = 'stanford'
                pos_tags, lemmas = get_pos_lemmas(tagger,sentences[sentence_counter])

                #now we got the correct pos_tags and lemmas for the sentence, now we have to store them
                for pos,lemma,ner in zip(pos_tags,lemmas,sentences_ner[sentence_counter]):
                    if isTest:
                        if lemma not in existingVocab:
                            tokens_pos_dict['UNK'] = 'UNK'
                            tokens_ner_dict['UNK'] = 'UNK'
                        else:
                            #lemma not in tokens_pos_dict is also not in tokens_ner_dict
                            if lemma not in tokens_pos_dict:
                                tokens_pos_dict[lemma] = pos if tagger == 'spacy' else pos[1]
                                tokens_ner_dict[lemma] = ner
                    else:
                        # if lemma in tokens_pos_dict and tokens_pos_dict[lemma] != pos[1]:
                        #     # print("previous lemma",lemma,tokens_pos_dict[lemma])
                        #     modified_lemma = lemma + '_' + pos[1][0].lower()
                        #     tokens_pos_dict[modified_lemma] = pos[1]
                        #     # print("new lemma",modified_lemma,tokens_pos_dict[modified_lemma])
                        #     tokens_ner_dict[modified_lemma] = ner
                        # elif lemma not in tokens_pos_dict:
                        if lemma not in tokens_pos_dict:
                            tokens_pos_dict[lemma] = pos if tagger == 'spacy' else pos[1]
                            tokens_ner_dict[lemma] = ner
                            #it can happen that a word can have multiple POS tags in different sentences
                            #like Race = Noun / Verb eg:
                            #'Race for outer space.' and 'The horse will race tomorrow.'
                            #For the first store it as noun : race = Noun
                            #For the second one store it as verb : race_v = Verb, so change the lemma for new one
                    
                            #check the previous POS with the current POS, if lemma already in vocab

                sentence_counter = increment_sentence_counter(sentence_counter)        
                continue

            token, ner = line.split()
            token_lower = token.lower()

            sentences[sentence_counter].append(token_lower)
            sentences_ner[sentence_counter].append(ner)

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


if __name__ == "__main__":
    #Task 1 : Data preprocessing#

    train_file = 'modified_train.txt'
    test_file = 'modified_test.txt'

    nlp = spacy.load("en_core_web_sm")  

    train_lemma_mapped_pos, train_lemma_mapped_ner = get_lemmas_based_pos_ner(train_file)
    test_lemma_mapped_pos, test_lemma_mapped_ner = get_lemmas_based_pos_ner(test_file,True, train_lemma_mapped_pos)
    # print(train_lemma_mapped_pos,train_lemma_mapped_ner)
    # print(test_lemma_mapped_pos,test_lemma_mapped_ner)

    train_upos_tags, train_lemmas, train_ners = extract_pos_lemma_ner(train_lemma_mapped_pos,train_lemma_mapped_ner)
    test_upos_tags, test_lemmas, test_ners = extract_pos_lemma_ner(test_lemma_mapped_pos,test_lemma_mapped_ner)


    #Get the lemma and pos mapping based on the training and testing data
    lemma_mapped_integer = make_integer_mappings(train_lemmas,test_lemmas)
    pos_mapped_integer = make_integer_mappings(train_upos_tags,test_upos_tags)

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

    #Task 3: Learning#

    train_input = list(train_one_hot_lemma_pos_encoded.values())
    train_output = list(train_lemma_mapped_ner.values())
    # print(len(train_input))
    # from sklearn.svm import SVC
    # model = SVC(random_state=42)

    from sklearn.naive_bayes import GaussianNB
    model = GaussianNB()

    model.fit(train_input,train_output)

    test_input = list(test_one_hot_lemma_pos_encoded.values())
    test_output = list(test_lemma_mapped_ner.values())
    # print(len(test_input))
    predicted = model.predict(test_input)

    # iter = 0
    # for lemma in test_lemma_mapped_pos:
    #     print(lemma,predicted[iter])
    #     iter += 1

    print("Accuracy: ",np.mean(predicted == test_output))