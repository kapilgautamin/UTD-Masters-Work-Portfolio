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

import numpy as np

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
	# raise Exception('Function not yet implemented!')


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
	# raise Exception('Function not yet implemented!')

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
	
	max_gain = 0
	max_gain_attribute_col = -1
	#Loop over all features
	#print(x[:5,:])
	for i in range(len(x[0])):
		unique_vals = partition(x[:,i])
		attribute_entropy_sum = 0
		# print(unique_vals)
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
		print("Feature x" + str(i+1) + " has following yes/no: " + str(uniq_class))
		information_gain = y_entropy - attribute_entropy_sum
		print("The info gain is: " + str(information_gain))
		if max_gain < information_gain:
			max_gain = information_gain
			max_gain_attribute_col = i
	
	print("The max gain is " + str(max_gain) + " and the attribute col is: " + str(max_gain_attribute_col))
	return max_gain_attribute_col
	# raise Exception('Function not yet implemented!')

def all_majority_class(pair,y):
	for keys in pair:
		majority_label = [0,0]
		count_key_indices = 0
		# print("For key: " + str(keys))
		for index in pair[keys]:
			# print(y[index],end=" ")
			count_key_indices += 1
			if y[index] == 0:
				majority_label[0] += 1
			else:
				majority_label[1] += 1
		# print()
		#Check if any attribute value is pure first for all keys
		if majority_label[0] == count_key_indices:
			return (True,keys,0)
		elif majority_label[1] == count_key_indices:
			return (True,keys,1)

	for keys in pair:
		majority_label = [0,0]
		for index in pair[keys]:
			if y[index] == 0:
				majority_label[0] += 1
			else:
				majority_label[1] += 1
		if majority_label[0] > majority_label[1]:
			return (False,keys,0)
		else:
			return (False,keys,1)

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
		# print("Initial attribute value pairs: " + str(attribute_value_pairs))
		
	levels = depth
	tree = {}

	attribute_index = mutual_information(x,y)
	curr_instances = len(y)
	uniq_class = partition(x[:,attribute_index])

	class_val = all_majority_class(uniq_class,y)
	majority_val = class_val[2]

	if attribute_index == -1:
		#this means the information gain is 0, or there is whole pure class
		# print("Case 1: The entire set of labels is pure, or there is nothing to split on, label is: " + str(majority_val))
		return majority_val
	
	# print(uniq_class)
	# print(class_val)
	# print("Current depth is: " + str(depth) + " and current instances are: " + str(curr_instances))
	# print("Max depth is: " + str(max_depth))
	
	chosen_attribute_val_pair = (attribute_index, int(class_val[1]))
	print("The choosen attribute val pair is: " + str(chosen_attribute_val_pair))

	if chosen_attribute_val_pair not in attribute_value_pairs:
		print("IT CAME HERE")
		for idx in range(len(attribute_value_pairs)):
			if attribute_index == attribute_value_pairs[idx][0]:
				chosen_attribute_val_pair = (attribute_index,attribute_value_pairs[idx][1])
				print("NEW choosen attribute val pair is: " + str(chosen_attribute_val_pair))
				break

	print("BEFORE Remaining attribute val pair is: " + str(attribute_value_pairs))
	attribute_value_pairs.remove(chosen_attribute_val_pair)
	print("AFTER Remaining attribute val pair is: " + str(attribute_value_pairs))

	if attribute_value_pairs != None and len(attribute_value_pairs) == 0:
		# print("Case 2: All the attribute pairs have been screened. Return majority value: " + str(majority_val))
		return int(majority_val)
	elif depth == max_depth:
		# print("Case 3: Pruning condition reached. Return majority value: " + str(majority_val))
		return int(majority_val)
	else:
		# print("Not terminated, Good to GO!!")
		levels += 1
		#Branch Out to True and False branch
		count_attribute = len(uniq_class[class_val[1]])

		xtrue,ytrue = np.zeros((count_attribute,len(x[0]))), np.zeros(count_attribute)
		xfalse,yfalse = np.zeros((curr_instances-count_attribute,len(x[0]))), np.zeros(curr_instances-count_attribute)

		counter_true = 0
		counter_false = 0
		# print(count_attribute)
		# print(curr_instances-count_attribute)
		for index in range(curr_instances):
			if x[index, attribute_index] == class_val[1]:
				for aid in range(len(x[0])):
					if aid != attribute_index:
						xtrue[counter_true,aid] = x[index,aid]
				ytrue[counter_true] = y[index]
				counter_true += 1
			else:
				for aid in range(len(x[0])):
					if aid != attribute_index:
						xfalse[counter_false,aid] = x[index,aid]
				yfalse[counter_false] = y[index]
				counter_false += 1

		# print(xtrue, ytrue)
		# print(xfalse, yfalse)
		# print("\n\n\nGoing TRUE for attribute val: " + str(chosen_attribute_val_pair))
		tree[attribute_index+1,int(chosen_attribute_val_pair[1]),True] = id3(xtrue,ytrue,attribute_value_pairs,depth=levels,max_depth=max_depth)
		# print("Ending TRUE for attribute val: " + str(chosen_attribute_val_pair))
		# print("\n\n\nGoing FALSE for attribute val: " + str(chosen_attribute_val_pair))
		tree[attribute_index+1,int(chosen_attribute_val_pair[1]),False] = id3(xfalse,yfalse,attribute_value_pairs,depth=levels,max_depth=max_depth)
		# print("Ending FALSE for attribu	te val: " + str(chosen_attribute_val_pair))
	
	return tree
	# raise Exception('Function not yet implemented!')


def predict_example(x, tree):
	"""
	Predicts the classification label for a single example x using tree by recursively descending the tree until
	a label/leaf node is reached.

	Returns the predicted label of x according to tree
	"""

	# INSERT YOUR CODE HERE. NOTE: THIS IS A RECURSIVE FUNCTION.
	# print("Tree now" , tree)
	if tree == 0 or tree == 1:
		return tree

	# (5,1,True)
	# (5,1,False)
	for keys in tree:
		# print(tree)
		if keys[1] == x[keys[0]-1] and keys[2] == True:
			return predict_example(x,tree[keys])
		elif keys[1] != x[keys[0]-1] and keys[2] == False:
			return predict_example(x,tree[keys])
	# raise Exception('Function not yet implemented!')


def compute_error(y_true, y_pred):
	"""
	Computes the average error between the true labels (y_true) and the predicted labels (y_pred)

	Returns the error = (1/n) * sum(y_true != y_pred)
	"""

	# INSERT YOUR CODE HERE
	n = len(y_pred)
	return (1/n) * sum(y_true != y_pred)
	# raise Exception('Function not yet implemented!')


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


if __name__ == '__main__':
	# Load the training data
	M = np.genfromtxt('./monks-1.train', missing_values=0, skip_header=0, delimiter=',', dtype=int)
	#Divide the data in class labels and attributes, Class label is in the first(0th) column
	ytrn = M[:, 0]
	Xtrn = M[:, 1:]
	# print(Xtrn,ytrn)

	# Load the test data
	M = np.genfromtxt('./monks-1.test', missing_values=0, skip_header=0, delimiter=',', dtype=int)
	ytst = M[:, 0]
	Xtst = M[:, 1:]
	# print(Xtst,ytst)
	# for dep in range(11):
	# Learn a decision tree of depth 3
	decision_tree = id3(Xtrn, ytrn, max_depth=3)
	visualize(decision_tree)
	# print(decision_tree)

	# Compute the test error
	y_pred = [predict_example(x, decision_tree) for x in Xtst]
	# print(y_pred)
	tst_err = compute_error(ytst, y_pred)
	# print('depth=',dep, end=" ")
	print('Test Error = {0:4.2f}%.'.format(tst_err * 100))
