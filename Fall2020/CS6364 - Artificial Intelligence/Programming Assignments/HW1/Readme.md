Name - Kapil Gautam
Email - kapil.gautam@utdallas.edu
Class - AI CS6364
Homework# - 1

Programming language used: Python

Files used:
1. search_problem.py : This file is a solution for the missionaries and cannibals problem 2 of the homework. It uses the Problem class of the search.py AIMA code as a base class and override it's functions.

2. seattle_to_dallas.py : This file is a solution for the road trip problem of travelling from Seattle to Dallas. It uses the GraphProblem class of the search.py AIMA code as a base class and override it's functions.

Supporting files from aima-python (https://github.com/aimacode/aima-python/)
1. search.py (https://github.com/aimacode/aima-python/blob/master/search.py)
2. utils.py (https://github.com/aimacode/aima-python/blob/master/utils.py)

Code Operation:
The files used for solving the problems of homework imports the search strategies as required in the homework from the search.py file. It then derives the search.py Problem and GraphProblem base classes functions and overrides the functions for the required functionality.

Special features:
1. Heuristic feature for problem 2 : missionaries and cannibals: h(x) = Total people on left side - 1
    The boat takes two people, but after each trip across the boat has to come back to the left side and so at least one person must paddle back. This heuristic is admissible, since all boat trips (except the last one) can result in a net transfer of at most one person to the destination side.

Note: Print commands inside the functions have been commented for iterative_deeping search strategy to keep the output of program short enough. In order to print more information, please uncomment them.