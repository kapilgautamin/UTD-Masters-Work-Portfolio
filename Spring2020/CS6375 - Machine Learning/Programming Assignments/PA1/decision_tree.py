# decision_tree.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# personal and educational purposes provided that (1) you do not distribute
# or publish solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UT Dallas, including a link to http://cs.utdallas.edu.
#
# This file is part of Programming Assignment 1 for CS6375: Machine Learning.
# Gautam Kunapuli (gautam.kunapuli@utdallas.edu)
# Sriraam Natarajan (sriraam.natarajan@utdallas.edu),
#
#
# INSTRUCTIONS:
# ------------
# 1. This file contains a skeleton for implementing the ID3 algorithm for
# Decision Trees. Insert your code into the various functions that have the
# comment "INSERT YOUR CODE HERE".
#
# 2. Do NOT modify the classes or functions that have the comment "DO NOT
# MODIFY THIS FUNCTION".
#
# 3. Do not modify the function headers for ANY of the functions.
#
# 4. You may add any other helper functions you feel you may need to print,
# visualize, test, or save the data and results. However, you MAY NOT utilize
# the package scikit-learn OR ANY OTHER machine learning package in THIS file.


# Authored by:
# Kapil Gautam, KXG180032
# Vishwashri Sairam Venkadathiriappasamy, VXV180043

#The below program uses Python3, numpy and math libraries
#it can be run by
#python KXG180032_PA1.py

#For each given dataset the program computes the decision tree for depths - 1 to 10 and output the tree,
#confusion matrix and training and test error for each of the depth. It also creates a plot of the training
#and the test error for depths 1 to 10.
#In the end Using sklearn library, it computes the tree, and the confusion matrix for the given datasets.
#The program uses monks-1, monks-2, monks-3, mushroom and tic-tac-toe datasets.
#The program assumes that the training and test sets are in the same directory of the program being executed.

import numpy as np
VERBOSE = False

def partition(x):
	"""
	Partition the column vector x into subsets indexed by its unique values (v1, ... vk)

	Returns a dictionary of the form
	{ v1: indices of x == v1,
	  v2: indices of x == v2,
	  ...
	  vk: indices of x == vk }, where [v1, ... vk] are all the unique values in the vector z.
	"""
	# INSERT YOUR CODE HERE
	v = {}
	# print("Number of Instances: " + str(len(x)))
	for row_pos in range(len(x)):
		if x[row_pos] in v:
			v[x[row_pos]].append(row_pos)
		else:
			v[x[row_pos]] = [row_pos]
	# print(v)
	return v
	raise Exception('Function not yet implemented!')


def entropy(y):
	"""
	Compute the entropy of a vector y by considering the counts of the unique values (v1, ... vk), in z

	Returns the entropy of z: H(z) = p(z=v1) log2(p(z=v1)) + ... + p(z=vk) log2(p(z=vk))
	"""
	# INSERT YOUR CODE HERE
	#Will need to put the formula for entropy for the unique values in z
	import math
	sum_of_prod = 0
	total_instances = 0
	for key in y:
		total_instances += len(y[key])
	# print("Total instances are: " + str(total_instances))
	for key in y:
		probability_of_val = len(y[key]) / total_instances
		# print("Prob of value is: " + str(probability_of_val))
		sum_of_prod += probability_of_val * math.log2(probability_of_val) * -1
	# print("Entropy is: " + str(sum_of_prod))
	return sum_of_prod
	raise Exception('Function not yet implemented!')

#This will be computer after entropy is calculated
def mutual_information(x, y):
	"""
	Compute the mutual information between a data column (x) and the labels (y). The data column is a single attribute
	over all the examples (n x 1). Mutual information is the difference between the entropy BEFORE the split set, and
	the weighted-average entropy of EACH possible split.

	Returns the mutual information: I(x, y) = H(y) - H(y | x)
	"""
	# INSERT YOUR CODE HERE
	# entropy(y) - entropy(y|x)
	total_instances = len(y)
	y_entropy = entropy(partition(y))
	# print("Combined entropy is: " + str(y_entropy))
	
	unique_vals = partition(x)
	attribute_entropy_sum = 0
	# print("unique_vals ",unique_vals)
	for key in unique_vals:
		uniq_class = {}
		uniq_class_count = 0
		for indice in unique_vals[key]:
			uniq_class_count += 1
			if y[indice] in uniq_class:
				uniq_class[y[indice]].append(indice)
			else:
				uniq_class[y[indice]] = [indice]
		attribute_entropy_sum += entropy(uniq_class) * (uniq_class_count/total_instances)
		# print("Next",uniq_class,attribute_entropy_sum)
	#For each feature we determined count of class yes/no
	# print("Feature has following yes/no: " + str(uniq_class))
	information_gain = y_entropy - attribute_entropy_sum
	if VERBOSE:
		print("The info gain is: " + str(information_gain))
	
	return information_gain
	raise Exception('Function not yet implemented!')

def id3(x, y, attribute_value_pairs=None, depth=0, max_depth=1):
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
			# print("idx,val=",idx,val)
			gains.append(mutual_information(x[:,idx] == val,y))

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
		xfalse = x[false_condition]
		yfalse = y[false_condition]

		if VERBOSE:
			print("Length before splitting:",len(xtrue),len(ytrue),len(xfalse),len(yfalse))
		
		if len(ytrue):
			# print("Going TRUE for attribute val: " + str(chosen_attribute_val_pair))
			tree[best_attribute_index,best_attribute_value,True] = id3(xtrue,ytrue,attribute_value_pairs,depth=levels,max_depth=max_depth)
			# print("Ending TRUE for attribute val: " + str(chosen_attribute_val_pair))
		if len(yfalse):
			# print("Going FALSE for attribute val: " + str(chosen_attribute_val_pair))
			tree[best_attribute_index,best_attribute_value,False] = id3(xfalse,yfalse,attribute_value_pairs,depth=levels,max_depth=max_depth)
			# print("Ending FALSE for attribute val: " + str(chosen_attribute_val_pair))
	
	return tree
	raise Exception('Function not yet implemented!')

def predict_example(x, tree):
	"""
	Predicts the classification label for a single example x using tree by recursively descending the tree until
	a label/leaf node is reached.

	Returns the predicted label of x according to tree
	"""
	# INSERT YOUR CODE HERE. NOTE: THIS IS A RECURSIVE FUNCTION.
	# print("Tree now" , tree)
	if tree in range(10):
		return tree
	
	# (5,1,True)
	# (5,1,False)
	for keys in tree:
		# print(tree)
		if keys[1] == x[keys[0]] and keys[2] == True:
			return predict_example(x,tree[keys])
		elif keys[1] != x[keys[0]] and keys[2] == False:
			return predict_example(x,tree[keys])
	raise Exception('Function not yet implemented!')


def compute_error(y_true, y_pred):
	"""
	Computes the average error between the true labels (y_true) and the predicted labels (y_pred)

	Returns the error = (1/n) * sum(y_true != y_pred)
	"""

	# INSERT YOUR CODE HERE
	n = len(y_true)
	return (1/n) * sum(y_true != y_pred)
	raise Exception('Function not yet implemented!')


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
	training_sets = ["monks-1.train","monks-2.train","monks-3.train","mushroom.train","tic-tac-toe.train"]
	# training_sets.extend(["breast-cancer.train","SPECT.train","SPECTF.train","car.train","balance-scale.train"])
	testing_sets = ["monks-1.test","monks-2.test","monks-3.test","mushroom.test","tic-tac-toe.test"]
	# testing_sets.extend(["breast-cancer.test","SPECT.test","SPECTF.test","car.test","balance-scale.test"])
	name_sets = ["monks-1 dataset","monks-2 dataset","monks-3 dataset","mushroom dataset","tic-tac-toe dataset"]
	# name_sets.extend(["breast-cancer dataset","SPECT dataset","SPECTF dataset","car dataset","balance-scale dataset"])	

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
		print("\nComputing testing and training error for ", name)
		for depth in range(1,max_depth):
			# Learn a decision tree of depth dep
			print("Depth: ", depth)
			decision_tree = id3(Xtrn, ytrn, max_depth=depth)
			visualize(decision_tree)
			# print(decision_tree)

			# Compute the training error
			y_pred = [predict_example(x, decision_tree) for x in Xtrn]
			# print(y_pred)
			
			train_err = compute_error(ytrn, y_pred)
			training.append(train_err*100)
			# print('depth=',dep, end=" ")
			print('Train Error = {0:4.2f}%.'.format(train_err * 100))

			# Compute the test error
			y_pred = [predict_example(x, decision_tree) for x in Xtst]
			# print(y_pred)
			
			tst_err = compute_error(ytst, y_pred)
			testing.append(tst_err*100)
			# print('depth=',depth, end=" ")
			print('Test Error = {0:4.2f}%.'.format(tst_err * 100))
			confusion_mat(ytst,y_pred)
		
		import matplotlib.pyplot as plt
		depth = [x for x in range(1,11)]
		plt.figure()
		plt.plot(depth,training,label="Training Error")
		plt.plot(depth,testing,label="Testing Error")
		plt.title('Training vs Testing Error for ' + name)
		plt.legend()
		plt.grid(True)
		plt.xlabel('Depth')
		plt.ylabel('Error %')
		# plt.show()
		plt.savefig(name +'.png')



	sklearn_training_set = ["monks-1.train","mushroom.train","tic-tac-toe.train"]
	sklearn_testing_set = ["monks-1.test","mushroom.test","tic-tac-toe.test"]
	sklearn_names_set = ["monks-1","mushroom","tic-tac-toe"]
	
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
		from sklearn.metrics import confusion_matrix,accuracy_score
		from sklearn.tree.export import export_text
		import graphviz
		
		# clf = tree.DecisionTreeClassifier(criterion="entropy",random_state=5, max_depth=2)
		clf = tree.DecisionTreeClassifier()
		clf = clf.fit(Xtrn, ytrn)
		dot_data = tree.export_graphviz(clf, out_file = None)
		graph = graphviz.Source(dot_data) 
		graph.render(name + "_decision_tree")
		# r = export_text(clf)
		# print(r)
		y_pred = clf.predict(Xtst)
		accuracy = accuracy_score(ytst,y_pred)
		print("test error: ", (1-accuracy)*100)
		print("Confusion matrix: \n",confusion_matrix(ytst,y_pred))