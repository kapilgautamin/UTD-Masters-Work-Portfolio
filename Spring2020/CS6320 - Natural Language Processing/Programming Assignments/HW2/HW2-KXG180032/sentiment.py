#Authored by: Kapil Gautam - KXG180032

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB   #NaiveBayes
from sklearn.linear_model import LogisticRegression #LogisticRegression
from sklearn.pipeline import Pipeline
import numpy as np
# import re 
from sklearn.svm import SVC
import os
from sklearn import metrics

class Review:
    def __init__(self):
        self.corpus = []
        self.target = []
        self.rating = []
        self.count = 0
    
    def add_reviews(self,location,sentiment):
        self.count = 0
        for document in os.listdir(location):
            # print(location+""+document)
            # res = re.split('._', document) 
            # print(res)
            # for s in document.split('_'):
            #     if '.' in s:
            #         self.rating.append(int(s.split('.')[0]))
            with open(location+""+document,"r",encoding="utf8") as file:
                raw_text = file.read()
            raw_text = open(location+""+document,"r",encoding="utf8").read()
            self.corpus.append(raw_text)
            self.target.append(sentiment)
            self.count += 1
            # if self.count > 1000:
            #     break
    
def form_pipeline(representation,classifier,stop_word,regularization):
    if stop_word == 0:
        stop_word = None
    else:
        stop_word = 'english'

    if classifier == 'nbayes':
        model = MultinomialNB()
    elif classifier == 'regression':
        if regularization == 'no':
            regularization = 'none'
            solver = 'lbfgs'
        else:
            solver = 'liblinear'
        model = LogisticRegression(penalty=regularization,solver=solver)

    if representation == 'bow':
        vector = CountVectorizer(stop_words=stop_word)
    elif representation == 'tfidf':
        vector = TfidfVectorizer(stop_words=stop_word)
        
    text_clf = Pipeline([
            ('vec', vector),
            ('clf', model),
        ])

    return text_clf

"""
python program.py <training-set> <test-set> <representation>
<classifier> <stop-words> <regularization>

where <training-set> represents the path to the training folder,
      <test-set> represents the path to the test folder,
      representation : 'bow, tfidf' is a string indicating what representation to use,
      classifier : 'nbayes, regression' is a string indicating what classier to use,
      stop-words: '0, 1' indicates whether or not to use stop words,
      regularization : 'no, l1, l2'  indicates whether to use L1 or L2 regularization or neither (note that this
                        argument is applicable only if you choose logistic regression classier)

For example, the call python program.py train test tfidf regression 0 l1 requires
the program to train logistic regression with L1 regularization on the les present in train
folder and test them on the le present in test folder. The tf-idf representation must be used
without removing any stop words.
"""
if __name__ == "__main__":
    
    import sys

    train_set = "train"
    test_set = "test"
    representation = 'tfidf'
    classifier = 'regression'
    stop_word = 1
    regularization = 'no'
    print("\n\n",sys.argv)

    #rewrite input from command line
    if len(sys.argv) > 5:
        train_set = sys.argv[1]
        test_set = sys.argv[2]
        representation = sys.argv[3]
        classifier = sys.argv[4]
        stop_word = int(sys.argv[5])
        if len(sys.argv) == 7:
            regularization = sys.argv[6]


    train_pos_loc = train_set + "/pos/"
    train_neg_loc = train_set + "/neg/"
    test_pos_loc = test_set + "/pos/"
    test_neg_loc = test_set + "/neg/"

    train_reviews = Review()
    train_reviews.add_reviews(train_pos_loc,'pos')
    train_reviews.add_reviews(train_neg_loc,'neg')

    # for review,senti in zip(train_reviews.corpus,train_reviews.target):
    #     print(review[:10],"=>",senti)

    test_reviews = Review()
    test_reviews.add_reviews(test_pos_loc,'pos')
    test_reviews.add_reviews(test_neg_loc,'neg')

    text_clf = form_pipeline(representation,classifier,stop_word,regularization)
    text_clf.fit(train_reviews.corpus,train_reviews.target)

    predicted = text_clf.predict(test_reviews.corpus)
    # for doc,rating,category in zip(test_reviews.corpus, test_reviews.rating,predicted):
    #     print('%d ==> %s ' % (rating,category))
    print(f"Parameters: Representation = {representation}, Classifier = {classifier}, Stop word = {stop_word}, Regularization = {regularization}")
    print("Accuracy: ",np.mean(predicted == test_reviews.target))
    print(metrics.classification_report(test_reviews.target, predicted,target_names=['neg','pos']))
    print("Confusion Matrix:")
    print(metrics.confusion_matrix(test_reviews.target, predicted))