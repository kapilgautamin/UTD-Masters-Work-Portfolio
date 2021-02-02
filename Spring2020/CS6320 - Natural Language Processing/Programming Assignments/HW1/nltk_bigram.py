# pip install nltk
import nltk
# from nltk.util import pad_sequence
# from nltk.util import bigrams
# from nltk.util import ngrams
# from nltk.util import everygrams
# from nltk.lm.preprocessing import pad_both_ends
# from nltk.lm.preprocessing import flatten
from nltk import word_tokenize, sent_tokenize 
# import nltk
# nltk.download('punkt')
from nltk.lm.preprocessing import padded_everygram_pipeline
from nltk.lm import MLE
from nltk.lm.models import Laplace

# train = open("train.txt","r")
# # count = 0
# tokenized_text = []
# for line in train:
#     # if count == 5:
#     #     break
#     # count += 1
#     # print(list(bigrams(line)))
#     token = [list(map(str.lower, word_tokenize(sent))) 
#                   for sent in sent_tokenize(line)]
#     tokenized_text.extend(token)
    
# n = 2
# train_data, padded_sents = padded_everygram_pipeline(n, tokenized_text)
# # model = MLE(n)
# model = Laplace(n)

# model.fit(train_data, padded_sents)
# # print(len(model.vocab))
# # for i in range(len(tokenized_text)):
# #     print(model.vocab.lookup(tokenized_text[i]))


# # print(model.counts[['father']]['brown'])
# # print(model.counts[['i']]['think'])
# # print(model.counts[['think']]['i'])

# # x = model.counts[['i']]['want']
# # y = model.counts['i']
# # #notice the square brackets positions 
# # print(x,y,x/y)
# # print(model.score('want','i'.split()))

# # print(model.counts[['a']]['black'] / model.counts['a'])
# # print(model.score('black','a'.split()))


# # x = model.counts[['do']]['not']
# # y = model.counts['do']
# # print(x,y,x/y)
# # print(model.score('not','do'.split()))

# # # model.score('is', 'language'.split())
# test = open("test.txt","r")
# for line in test:
#     token = [list(map(str.lower, word_tokenize(sent))) 
#                   for sent in sent_tokenize(line)]
#     token = token[0]
#     print(token)
#     final_prob = 1
#     for pos in range(len(token)-1):
#         # x = model.counts[[token[pos]]][token[pos+1]]
#         # y = model.counts[token[pos]]
#         x = token[pos]
#         y = token[pos+1]
#         final_prob *= model.score(y,x.split())
#         print(x,y,final_prob)
#     print(final_prob)


# import nltk
# nltk.download('averaged_perceptron_tagger')
# text = word_tokenize("John usually leaves the house at seven.")
# text = word_tokenize("The rumor that john was elected spread rapidly.")
text = word_tokenize("Bill Gates, the founder of Microsoft, generously donates money to charities every year.")
print(nltk.pos_tag(text))