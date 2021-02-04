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

def rec(p1,p2,p3=None,depth=0,max_level = 3):
    if p3 == None:
        p3= (p1,p2)
        
    level = depth
    dic = {}

    if level == max_level:
        return dic
    
    #for true
    dic[depth,True] = rec(p1,p2,p3,depth=level+1)
    dic[depth,False] = rec(p1,p2,p3,depth=level+1)

    return dic

tree = rec(4,5)
visualize(tree)