The program uses Python3 to perform the sentiment analysis("positive","negative") of the movie review dataset.
It uses Bag of words and TfIdf representations, using the NaiveBayes and LogisticRegression classifier,
with stop words and regularization parameters(where applicable).

The output of the whole dataset is provided in 'output.txt'.
The cumulation of the various results are helped by a helper file(batch_run.sh) to run all the possible combinations.

To run the program:
python3 sentiment.py arg1 arg2 arg3 arg4 arg5 arg6

arg1 is the location of the training folder.
arg2 is the location of the test folder.
arg3 is the representation - it takes value {'bow','tfidf'}
arg4 is the classifier - it takes value {'nbayes','regression'}
arg5 is the stop words - it takes value {0,1}
arg6 is the regularization - it takes value {'no','l1','l2'}

By default it will take arg1 as "train/", arg2 as "test/", arg3 as 'tfidf',
arg4 as 'regression', arg5 as '1', arg6 as 'no'.
Unless the parameters are given on the command line interface.

Example run :
> python sentiment.py train test tfidf regression 1 l2
It uses 'TfidfVectorizer' representation with stop words('english') and 'LogisticRegression' classifier with 'L2' regularization.

Example run:
> python sentiment.py train test bow nbayes 0
It uses 'CountVectorizer' representation with no stop words and 'MultinomialNB', a type of NBayes classifier with no regularization.

The output can be stored in a text file.
Note : Please check that the word wrap feature of text file needs to be disabled,
to properly view the output.
Example run:
> time python sentiment.py train test tfidf regression 1 l2 >> output.txt

The program outputs the following:
1. An Accuracy score of sentiment analysis on the given test data.
2. A matrix reporting accuracy, precision, recall and F-score values for the test set.

Results(sorted):
Accuracy:  0.88316  tfidf regression 0 l2
Accuracy:  0.8794   tfidf regression 0 l1
Accuracy:  0.879    tfidf regression 1 l2
Accuracy:  0.87332  tfidf regression 1 l1
Accuracy:  0.86672  bow   regression 0 l2
Accuracy:  0.86444  bow   regression 0 l1
Accuracy:  0.86132  bow   regression 0 no
Accuracy:  0.85912  bow   regression 1 l1
Accuracy:  0.85904  bow   regression 1 l2
Accuracy:  0.85512  tfidf regression 0 no
Accuracy:  0.84844  tfidf regression 1 no
Accuracy:  0.83364  bow   regression 1 no
Accuracy:  0.82992  tfidf nbayes 1
Accuracy:  0.82956  tfidf nbayes 0
Accuracy:  0.81968  bow   nbayes 1
Accuracy:  0.81356  bow   nbayes 0

Analysis:
For the given dataset, LogisticRegression works better than NBayes classifier.
L2 regularization works better than no regularization in LogisticRegression classifier.
Also TfidfVectorizer('tfidf') performs better than CountVectorizer('bow').
Applying Stop words(Using 'english' data from sklearn) do have some effect in NaiveBayes.