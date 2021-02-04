import nltk
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag

ex = 'European authorities fined Google a record $5.1 billion on Wednesday for abusing its power in the mobile phone market and ordered the company to alter its practices'

def preprocess(text):
    sent = word_tokenize(text)
    print(sent)
    return pos_tag(sent)

sent = preprocess(ex)
# print(sent)

# #To find Noun Pharases
# pattern = 'NP: {<DT>?<JJ>*<NN>}'
# cp = nltk.RegexpParser(pattern)
# cs = cp.parse(sent)
# print(cs)