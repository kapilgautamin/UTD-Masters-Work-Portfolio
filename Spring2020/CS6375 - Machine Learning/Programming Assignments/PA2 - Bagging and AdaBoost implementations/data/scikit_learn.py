# from sklearn import tree
import numpy as np
# from sklearn.datasets import load_iris

# clf = tree.DecisionTreeClassifier(criterion = "entropy")
# iris = load_iris()
# clf = clf.fit(iris.data, iris.target)

M = np.genfromtxt('./monks-1.train', missing_values=0, skip_header=0, delimiter=',', dtype=int)
#Divide the data in class labels and attributes, Class label is in the first(0th) column
ytrn = M[:, 0]
Xtrn = M[:, 1:]
# clf = clf.fit(Xtrn, ytrn)

# tree.export_graphviz(clf, out_file ="myTreeName.dot")


from id3 import Id3Estimator
from id3 import export_graphviz

feature_names = ['x1','x2','x3','x4','x5','x6']
estimator = Id3Estimator()
estimator.fit(Xtrn, ytrn)
export_graphviz(estimator.tree_, 'tree.dot', feature_names)