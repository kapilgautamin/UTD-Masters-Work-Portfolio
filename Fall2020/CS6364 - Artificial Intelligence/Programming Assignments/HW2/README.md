Name: Kapil Gautam
Email : kapil.gautam@utdallas.edu
Class : AI CS6364
Homework number : 2
Problem it solves: 
   Problem 1 (1.1, 1.2, 1.3, 1.4) - However, i have not done the extra-credit part at the end of problem 1
   Problem 2
   Problem 3

Progrmamming langauge used: Python

File included:
game_iterative.py - This file contains the code for the Problem 1 Question 1
minimax_game.py - This file contains the code for the Problem 1 Question 3
output_iterative.txt - Output file of game_iterative.py
output_minimax.txt - Output file of minimax_game.py
Homework2_KXG180032.pdf - Main file for homework

Code operation:
   The given code required python language to be installed.
   The given python files can be executed using CLI by typing the following commands.
   - python game_iterative.py
   - python minimax_game.py
   These files can be written to output file which already has been attached. The command to write to a file is:
   - python game_iterative.py > output_iterative.txt
   - python minimax_game.py > output_minimax.txt
   
Special features:
The code for both the files explore whole depth of the game tree by using the queue data structure by following
breadth first search on the nodes. The game works turn-by-turn as given in the problem like P1 -> P2 -> P3 -> P4 -> P1
and the cycle repeats itself until a terminal node is found. It utlises the classes position, Player and Iterative/Backup_Minimax_Game
for the functionality of the game.