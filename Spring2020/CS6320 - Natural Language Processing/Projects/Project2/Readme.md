The project uses a Google Colaboratory .ipynb file, since the computing power required was'nt met with my device.
The project successfully ran on a hosted runtime having 35GB RAM and TPU selected.(One hot encoding takes a lot of memory.)

All the required dependecies of the project can be included by running the cells of the notebook.
You can run the notebook by pressing Ctrl+F9 or by clicking 'Run All' from the 'Runtime' menu.
It can be accessed from here: https://colab.research.google.com/drive/1Oc4ZKucV_Y8BIlPEENjo6GWtevnK-rKz

There are two kinds of accuracy reported in this:
1. Model accuracy : This is the accuracy for the model being used for the feature 1-hot encodings for unique tokens.
2. NER Test Token Accuracy : This is the accuracy of all the test tokens, using the above models predicted NER tags, and then calculating the accuracy metric and F1 score with that metric. (Also removing the BIO-tag violations)

The best accuracy to be recorded is with GaussianNB ML algorithm, being 87.096%(being NER Test Token Accuracy, Model Accuracy	: 57.352%), and it used the 'spacy' library for its POS tagging and lemmatization tasks.

The model can use either 'nltk' OR the 'spacy' library by changing the 'tagger' variable.
For eg: tagger = 'nltk' or tagger = 'spacy'

*Throughput - Size of testing file / Time taken for making prediction

Using NLTK library for POS tags and lemmatization:
With GaussianNB :
	Training time 			:	10.03 seconds
	Testing time 			:	8.46 seconds
	Model Accuracy 			:	59.558%
	Model Throughput 		:	44.876647561548765 Kbps
	NER Test Token Accuracy :	86.427%
	
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
 
 
Other Machine Learning models performed lesser than the GaussianNB in order :

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


SVM was taking very, very long to train, so I did'nt continue with that.

							GaussianNB		BernoulliNB		MultinomialNB		LogisticRegression		RandomForestClassifier
Training Time				10.03 s			6.41s			2.3s				35.96s					1363.65s
Testing Time				8.46s			1.96s			0.75s				0.73s					9.86s
NER Test Token Accuracy		86.427%			78.34%			78.351%				78.924%					84.179%
Model Throughput			44.87 Kbps		193.69 Kbps		502.72 Kbps			516.88 Kbps				38.57 Kbps

Overall analysis:
GaussianNB and RandomForestClassifier gave good results, though the training time for RandomForestClassifier was long.






Using Spacy library for POS tags and lemmatization:
With GaussianNB :
	Training time 				: 	6.68 seconds
	Testing time 				:	8.36 seconds
	Model Accuracy				: 	57.352%
	Model Throughput 			: 	45.39114866835412 Kbps
	NER Test Token Accuracy		:	87.096%
	
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
	
	
								GaussianNB		BernoulliNB		MultinomialNB		LogisticRegression		RandomForestClassifier
Training Time					6.68 s			5.78s			2.42s				23.21s					728.77s
Testing Time					8.36s			1.18s			0.81s				0.62s					6.92s
NER Test Token Accuracy			87.096%			81.194%			78.35%				84.01%					85.922%
Model Throughput				45.39 Kbps		320.98 Kbps		468.51 Kbps			611.16 Kbps				54.85 Kbps

Closing Notes:

Overall Spacy seems to perform better than 'nltk' with the above ML algorithms for the NER prediction task on the given datasets.
However, 'nltk' was much faster to process the 'POS' and to lemmatize a token in comparison to 'spacy'.

GaussianNB algorithm for Naive bayes and RandomForestClassifier for the Ensemble method, both using 'spacy' library performs good on the NER classification task.

