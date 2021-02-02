# KXG180032_PA2.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# personal and educational purposes provided that (1) you do not distribute
# or publish solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UT Dallas, including a link to http://cs.utdallas.edu.
#
# This file is part of Programming Assignment 1 for CS6375: Machine Learning.
# Gautam Kunapuli (gautam.kunapuli@utdallas.edu)
# Sriraam Natarajan (sriraam.natarajan@utdallas.edu),

# Authored by:
# Kapil Gautam, KXG180032

#Instructions and Information about usage:

#The below program uses Python3, numpy and math libraries
#it can be run like:
#python KXG180032_PA2.py

# The program works on a 'train', 'test' dataset of mushroom data. It will be predicting the "bruises?" attribute, with 
# the 'stalk-root' being excluded since it contain a lot of missing values. The data set has been converted from string to
# integer, with the unique feature values being assigned indices starting from 0 in alphabetical order.
# I have downloaded the dataset from UCI and shifted the "bruises?" class to 0th column using pandas library in a 
# seperate file, for keeping the logic different from data preperation. It uses the previous implemented decision 
# tree algo with a little modification to include weights to compute the weighted entropy.
# Finally scikit learn library is being used, to compare my bagging and boosting implementations.

import numpy as np
import math
VERBOSE = False
# VERBOSE = True

def partition(x):
	"""
	Partition the column vector x into subsets indexed by its unique values (v1, ... vk)

	Returns a dictionary of the form
	{ v1: indices of x == v1,
	  v2: indices of x == v2,
	  ...
	  vk: indices of x == vk }, where [v1, ... vk] are all the unique values in the vector z.
	"""
	v = {}
	# print("Number of Instances: " + str(len(x)))
	for row_pos in range(len(x)):
		if x[row_pos] in v:
			v[x[row_pos]].append(row_pos)
		else:
			v[x[row_pos]] = [row_pos]
	# print(v)
	return v


def entropy(y,weight):
	"""
	Compute the entropy of a vector y by considering the counts of the unique values (v1, ... vk), in z
	Returns the entropy of z: H(z) = p(z=v1) log2(p(z=v1)) + ... + p(z=vk) log2(p(z=vk))
	"""
	#Will need to put the formula for entropy for the unique values in z
	entropy = 0
	class_0 = np.sum(weight[y == 0])
	class_1 = np.sum(weight[y == 1])

	# print(np.unique(y, return_counts=True))
	# print(weight[:20])
	class_sum = class_0 + class_1

	probability_of_class_0 = class_0 / class_sum
	probability_of_class_1 = class_1 / class_sum
	if probability_of_class_0:
		entropy += probability_of_class_0 * math.log2(probability_of_class_0) * -1
	if probability_of_class_1:
		entropy += probability_of_class_1 * math.log2(probability_of_class_1) * -1

	if VERBOSE:
		print("Entropy is: " + str(entropy))
	return entropy

#This will be computed after entropy is calculated
def mutual_information(x, y, weight):
	"""
	Compute the mutual information between a data column (x) and the labels (y). The data column is a single attribute
	over all the examples (n x 1). Mutual information is the difference between the entropy BEFORE the split set, and
	the weighted-average entropy of EACH possible split.

	Returns the mutual information: I(x, y) = H(y) - H(y | x)
	"""
	# entropy(y) - entropy(y|x)
	# total_instances = len(y)
	# print("total_instances= ",len(y),len(weight))
	y_entropy = entropy(y,weight)
	# print("Combined entropy is: " + str(y_entropy))
	
	unique_vals = partition(x)
	weighted_entropy = 0
	# print("unique_vals ",unique_vals)
	# calculate the weighted entropy over the partition of x
	for key in unique_vals:
		indices = unique_vals[key]
		ent = entropy(y[indices],weight[indices])
		weighted_entropy += (np.sum(weight[indices])/np.sum(weight)) * ent

	information_gain = y_entropy - weighted_entropy
	if VERBOSE:
		print("The info gain is: " + str(information_gain))
	
	return information_gain

def id3(x, y, weight, attribute_value_pairs=None, depth=0, max_depth=1):
	"""
	Implements the classical ID3 algorithm given training data (x), training labels (y) and an array of
	attribute-value pairs to consider. This is a recursive algorithm that depends on three termination conditions
		1. If the entire set of labels (y) is pure (all y = only 0 or only 1), then return that label
		2. If the set of attribute-value pairs is empty (there is nothing to split on), then return the most common
		   value of y (majority label)
		3. If the max_depth is reached (pre-pruning bias), then return the most common value of y (majority label)
	Otherwise the algorithm selects the next best attribute-value pair using INFORMATION GAIN as the splitting criterion
	and partitions the data set based on the values of that attribute before the next recursive call to ID3.

	The tree we learn is a BINARY tree, which means that every node has only two branches. The splitting criterion has
	to be chosen from among all possible attribute-value pairs. That is, for a problem with two features/attributes x1
	(taking values a, b, c) and x2 (taking values d, e), the initial attribute value pair list is a list of all pairs of
	attributes with their corresponding values:
	[(x1, a),
	 (x1, b),
	 (x1, c),
	 (x2, d),
	 (x2, e)]
	 If we select (x2, d) as the best attribute-value pair, then the new decision node becomes: [ (x2 == d)? ] and
	 the attribute-value pair (x2, d) is removed from the list of attribute_value_pairs.

	The tree is stored as a nested dictionary, where each entry is of the form
					(attribute_index, attribute_value, True/False): subtree
	* The (attribute_index, attribute_value) determines the splitting criterion of the current node. For example, (4, 2)
	indicates that we test if (x4 == 2) at the current node.
	* The subtree itself can be nested dictionary, or a single label (leaf node).
	* Leaf nodes are (majority) class labels

	Returns a decision tree represented as a nested dictionary, for example
	{(4, 1, False):
		{(0, 1, False):
			{(1, 1, False): 1,
			 (1, 1, True): 0},
		 (0, 1, True):
			{(1, 1, False): 0,
			 (1, 1, True): 1}},
	 (4, 1, True): 1}
	"""
	# INSERT YOUR CODE HERE. NOTE: THIS IS A RECURSIVE FUNCTION.
	if attribute_value_pairs == None:
		attribute_value_pairs = []
		for i in range(len(x[0])):
			uniq_values = partition(x[:,i])
			for key in uniq_values:
				attribute_value_pairs.append((i,key))
		if VERBOSE:
			print("Initial attribute value pairs: " + str(attribute_value_pairs))
		
	levels = depth
	tree = {}

	uniq, count = np.unique(y, return_counts=True)
	if VERBOSE:
		print("\n\nuniq and count",uniq,count)
		print("majority: ",uniq[np.argmax(count)])

	if len(uniq) == 1:
		#this means the information gain is 0, or there is whole pure class
		if VERBOSE:
			print("Case 1: The entire set of labels is pure, or there is nothing to split on, label is: " + str(y[0]))
		return y[0]
	elif attribute_value_pairs != None and len(attribute_value_pairs) == 0:
		majority_val = uniq[np.argmax(count)]
		if VERBOSE:
			print("Case 2: All the attribute pairs have been screened. Return majority value: " + str(majority_val))
		return majority_val
	elif depth == max_depth:
		majority_val = uniq[np.argmax(count)]
		if VERBOSE:
			print("Case 3: Pruning condition reached. Return majority value: " + str(majority_val))
		return majority_val
	else:
		if VERBOSE:
			print("Case Split: Find next best pair")
		gains = []
		for (idx,val) in attribute_value_pairs:
			# print("idx,val=",idx,val,"len(y)= ",len(y),len(x))
			gains.append(mutual_information(x[:,idx] == val, y, weight))

		chosen_attribute_val_pair = attribute_value_pairs[np.argmax(gains)]
		if VERBOSE:
			print("The choosen attribute val pair is: " + str(chosen_attribute_val_pair))

		best_attribute_index = chosen_attribute_val_pair[0]
		best_attribute_value = chosen_attribute_val_pair[1]

		# print("BEFORE Remaining attribute val pair is: " + str(attribute_value_pairs))
		new_attribute_value_pairs = []
		for i in attribute_value_pairs:
			if i != chosen_attribute_val_pair:
				new_attribute_value_pairs.append(i)

		#We could have also done list.remove() but then Python uses object reference,
		#  and we don't want that in our recursion
		attribute_value_pairs = new_attribute_value_pairs
		# print("AFTER Remaining attribute val pair is: " + str(attribute_value_pairs))
	
		levels += 1
		#Branch Out to True and False branch
		true_condition = x[:,best_attribute_index] == best_attribute_value
		false_condition = x[:,best_attribute_index] != best_attribute_value

		xtrue = x[true_condition]
		ytrue = y[true_condition]
		wtrue = weight[true_condition]

		xfalse = x[false_condition]
		yfalse = y[false_condition]
		wfalse = weight[false_condition]

		if VERBOSE:
			print("Length before splitting:",len(xtrue),len(ytrue),len(xfalse),len(yfalse))
		
		if len(ytrue):
			# print("Going TRUE for attribute val: " + str(chosen_attribute_val_pair))
			tree[best_attribute_index,best_attribute_value,True] = id3(xtrue, ytrue, wtrue, attribute_value_pairs, depth=levels, max_depth=max_depth)
			# print("Ending TRUE for attribute val: " + str(chosen_attribute_val_pair))
		if len(yfalse):
			# print("Going FALSE for attribute val: " + str(chosen_attribute_val_pair))
			tree[best_attribute_index,best_attribute_value,False] = id3(xfalse, yfalse, wfalse, attribute_value_pairs, depth=levels, max_depth=max_depth)
			# print("Ending FALSE for attribute val: " + str(chosen_attribute_val_pair))
	
	return tree

def predict_example_base_learner(x, tree):
	"""
	Predicts the classification label for a single example x using tree by recursively descending the tree until
	a label/leaf node is reached.
	Returns the predicted label of x according to tree
	"""
	# INSERT YOUR CODE HERE. NOTE: THIS IS A RECURSIVE FUNCTION.
	# print("Tree now" , tree)
	if tree in range(20):
		return tree
	
	# (5,1,True)
	# (5,1,False)
	for keys in tree:
		# print(tree)
		if keys[1] == x[keys[0]] and keys[2] == True:
			return predict_example_base_learner(x,tree[keys])
		elif keys[1] != x[keys[0]] and keys[2] == False:
			return predict_example_base_learner(x,tree[keys])

def compute_error(y_true, y_pred):
	"""
	Computes the average error between the true labels (y_true) and the predicted labels (y_pred)

	Returns the error = (1/n) * sum(y_true != y_pred)
	"""
	# INSERT YOUR CODE HERE
	n = len(y_true)
	return (1/n) * sum(y_true != y_pred)

def bagging(x,y,max_depth,num_trees):
	import random
	random.seed(0)
	lenX = len(x)
	sequence = list(range(len(x)))
	
	hypothesis = {}
	alpha = 1
	weight = np.ones(lenX)
	
	# print("lenX= ",len(x)," lenY= ",len(y))

	for tn in range(num_trees):
		indices = random.choices(sequence,k=lenX)
		# print(indices)
		# print(len(indices),len(np.unique(indices)))
		
		decision_tree = id3(x[indices], y[indices], weight, max_depth=max_depth)
		hypothesis[tn] = (alpha,decision_tree)

	# print(hypothesis)
	return hypothesis
	

def boosting(x,y,max_depth,num_stumps):
	lenX = len(x)
	hypothesis = {}
	weight = np.ones(lenX)/lenX

	for ns in range(num_stumps):
		decision_tree = id3(x, y, weight, max_depth=max_depth)
		y_pred = [predict_example_base_learner(xe, decision_tree) for xe in x]
		
		# incorrect = sum(y != y_pred)
		# print("incorrect" ,incorrect)
		#total error is the sum of weights for incorrectly classified errors
		epsilon = np.sum(weight[y_pred != y]) / np.sum(weight)
		alpha = np.log((1 - epsilon) / epsilon) / 2
		# print(epsilon,alpha)
		
		for i in range(len(y_pred)):
			if y_pred[i] == y[i]:	#correct
				weight[i] *= np.exp(-alpha)
			else:				#incorrect
				weight[i] *= np.exp(alpha)

		weight = weight / np.sum(weight)
		hypothesis[ns] = (alpha, decision_tree)

	# print(hypothesis)
	return hypothesis

def predict_example(x,h_ens,ensemble_type):
	"""
	h_ens is an ensemble of weighted hypotheses.
	The ensemble is represented as an array of pairs [(alpha_i, h_i)], where each hypothesis and weight
	are represented by the pair: (alpha_i, h_i).
	"""
	if ensemble_type == "bagging":
		predictions = []
		for k in h_ens:
			y_pred = predict_example_base_learner(x,h_ens[k][1])
			predictions.append(y_pred)

		# print(predictions)
		predict_egz = max(predictions, key=predictions.count)
		# print(predict_egz)
		return predict_egz
	else:
		predictions = []
		sum_alpha = 0

		for y in h_ens:
			# splitting hypothesis, weight pairs
			alpha, tree = h_ens[y]
			tst_pred = predict_example_base_learner(x, tree)
			# appending prediction
			predictions.append(tst_pred*alpha)
			sum_alpha += alpha
		predict_egz = np.sum(predictions) / sum_alpha
		if predict_egz >= 0.5:
			return 1
		else:
			return 0

def visualize(tree, depth=0):
	"""
	Pretty prints (kinda ugly, but hey, it's better than nothing) the decision tree to the console. Use print(tree) to
	print the raw nested dictionary representation.
	DO NOT MODIFY THIS FUNCTION!
	"""

	if depth == 0:
		print('TREE')

	for index, split_criterion in enumerate(tree):
		sub_trees = tree[split_criterion]

		# Print the current node: split criterion
		print('|\t' * depth, end='')
		print('+-- [SPLIT: x{0} = {1}]'.format(split_criterion[0], split_criterion[1]))

		# Print the children
		if type(sub_trees) is dict:
			visualize(sub_trees, depth + 1)
		else:
			print('|\t' * (depth + 1), end='')
			print('+-- [LABEL = {0}]'.format(sub_trees))

def confusion_mat(y_true,y_pred):
	tp,tn,fp,fn = 0,0,0,0
	for i in range(len(y_pred)):
		if y_pred[i] == 1 and y_true[i] == 1:
			tp += 1
		elif y_pred[i] == 0 and y_true[i] == 0:
			tn += 1
		elif y_pred[i] == 1 and y_true[i] == 0:
			fp += 1
		elif y_pred[i] == 0 and y_true[i] == 1:
			fn += 1
	# mat = np.array([tn,fp,fn,tp])		#This is simlar to sklearn convention
	mat = np.array([tp,fn,fp,tn]).reshape(2,2)		#This is for current assignment
	print("\t\tClassifier Prediction")
	print("\t\t\tPositive\tNegative")
	print("Actual | Positive\t",mat[0][0],"\t\t",mat[0][1])
	print("Value  | Negative\t",mat[1][0],"\t\t",mat[1][1])


if __name__ == '__main__':
	# Load the training data
	# training_sets = ["train","monks-1.train","monks-2.train","monks-3.train"]
	# testing_sets = ["test","monks-1.test","monks-2.test","monks-3.test"]
	# name_sets = ["mushroom data","monks-1 dataset","monks-2 dataset","monks-3 dataset"]

	training_sets = ["train"]
	testing_sets = ["test"]
	name_sets = ["mushroom data"]
	

	for train_idx,test_idx,name in zip(training_sets,testing_sets,name_sets):
		M = np.genfromtxt(train_idx, missing_values=0, skip_header=0, delimiter=',', dtype=int)
		#Divide the data in class labels and attributes, Class label is in the first(0th) column
		ytrn = M[:, 0]
		Xtrn = M[:, 1:]
		# print(Xtrn,ytrn)

		# Load the test data
		M = np.genfromtxt(test_idx, missing_values=0, skip_header=0, delimiter=',', dtype=int)
		ytst = M[:, 0]
		Xtst = M[:, 1:]
		max_depth = 11
		# print(Xtst,ytst)

		training = []
		testing = []
		# Bagging
		print("\nComputing testing and training error Bagging for ", name)
		for depth in [3,5]:
			for bag_size in [10,20]:
				# Learn a decision tree of depth dep
				print("Depth: ", depth, "Bag Size: ", bag_size)
				ensemble_bag = bagging(Xtrn,ytrn,depth,bag_size)

				# Compute the test error
				y_pred = [predict_example(x, ensemble_bag, "bagging") for x in Xtst]
				# print(y_pred)
				
				tst_err = compute_error(ytst, y_pred)
				testing.append(tst_err*100)
				# # print('depth=',depth, end=" ")
				print('Test Error = {0:4.2f}%.'.format(tst_err * 100))
				confusion_mat(ytst,y_pred)
		
		#Boosting
		print("\nComputing testing and training error Boosting for ", name)
		for depth in [1,2]:
			for bag_size in [20,40]:
				# Learn a decision tree of depth dep
				print("Depth: ", depth, "Bag Size: ", bag_size)
				ensemble_boost = boosting(Xtrn,ytrn,depth,bag_size)

				# # Compute the test error
				y_pred = [predict_example(x, ensemble_boost, "boosting") for x in Xtst]
				# # print(y_pred)
				
				tst_err = compute_error(ytst, y_pred)
				testing.append(tst_err*100)
				# print('depth=',depth, end=" ")
				print('Test Error = {0:4.2f}%.'.format(tst_err * 100))
				confusion_mat(ytst,y_pred)

	print("\n\n\nNow using sklearn library\n\n")
	# sklearn_training_set = ["train","monks-1.train","monks-2.train","monks-3.train"]
	# sklearn_testing_set = ["test","monks-1.test","monks-2.test","monks-3.test"]
	# sklearn_names_set = ["mushroom data","monks-1 dataset","monks-2 dataset","monks-3 dataset"]
	sklearn_training_set = ["train"]
	sklearn_testing_set = ["test"]
	sklearn_names_set = ["mushroom data"]
	
	
	for train,test,name in zip(sklearn_training_set,sklearn_testing_set,sklearn_names_set):
		M = np.genfromtxt(train, missing_values=0, skip_header=0, delimiter=',', dtype=int)
		#Divide the data in class labels and attributes, Class label is in the first(0th) column
		ytrn = M[:, 0]
		Xtrn = M[:, 1:]
		# print(Xtrn,ytrn)

		# Load the test data
		M = np.genfromtxt(test, missing_values=0, skip_header=0, delimiter=',', dtype=int)
		ytst = M[:, 0]
		Xtst = M[:, 1:]

		#pip install graphviz
		from sklearn import tree
		from sklearn.ensemble import BaggingClassifier
		from sklearn.ensemble import AdaBoostClassifier
		from sklearn.metrics import confusion_matrix,accuracy_score
		
		# clf = tree.DecisionTreeClassifier(criterion="entropy",random_state=5, max_depth=2)
		max_depth = [3,5]
		bag_size = [10,20]

		for md in max_depth:
			for bs in bag_size:
				
				print("Bagging : max_depth =",md,"bag_size = ",bs," for ",name)
				# clf = BaggingClassifier(tree.DecisionTreeClassifier(criterion="entropy",random_state = 42, max_depth = md),n_estimators = bs)
				clf = BaggingClassifier(tree.DecisionTreeClassifier(random_state = 42, max_depth = md),n_estimators = bs)
				clf = clf.fit(Xtrn, ytrn)
				y_pred = clf.predict(Xtst)
				accuracy = accuracy_score(ytst,y_pred)
				print("test error(%): ", (1-accuracy)*100)
				print("Confusion matrix: \n",confusion_matrix(ytst,y_pred))
				print("\n\n")

		max_depth = [1,2]
		bag_size = [20,40]

		for md in max_depth:
			for bs in bag_size:
				
				print("AdaBoost : max_depth =",md,"bag_size = ",bs," for ",name)
				# clf = AdaBoostClassifier(tree.DecisionTreeClassifier(criterion="entropy",random_state = 42, max_depth = md),n_estimators = bs)
				clf = AdaBoostClassifier(tree.DecisionTreeClassifier(random_state = 42, max_depth = md),n_estimators = bs)
				clf = clf.fit(Xtrn, ytrn)
				y_pred = clf.predict(Xtst)
				accuracy = accuracy_score(ytst,y_pred)
				print("test error(%): ", (1-accuracy)*100)
				print("Confusion matrix: \n",confusion_matrix(ytst,y_pred))
				print("\n\n")