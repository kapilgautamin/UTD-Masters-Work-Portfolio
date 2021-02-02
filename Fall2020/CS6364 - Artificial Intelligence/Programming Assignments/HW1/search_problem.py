#Search strategies implementation taken from AIMA code
from search import *

#Overriding the Problem class of AIMA code
class Search_Problem(Problem):
    def __init__(self, initial, goal):
        Problem.__init__(self, initial, goal)
        self.state = initial
        
    def h(self, state):
        heuristic1 = self.state[0] + self.state[1] - 1
        # print("changing1", heuristic1)
        return heuristic1

    def actions(self, state):
        """
        List of actions which can be executed in the given state
        """
        missionary = state[0]
        cannibal = state[1]
        side = state[2]

        if(side=='right'): #Boat is on right side, have to move the boat to the left side
           if(missionary==3): 
               if(cannibal==2): 
                   return ['C']
               else: 
                   return ['C', 'CC']
           elif(missionary==2):
               if(cannibal==2): 
                   return ['M', 'MC', 'C']
               else: #does not satisfy constraint
                   return []; 
           elif(missionary==1): 
               if(cannibal==1):
                   return ['MM', 'M', 'MC', 'C', 'CC']
               else: #does not satisfy constraint
                   return []; 
           else: 
               if(cannibal==3): 
                   return ['M', 'MM']
               elif(cannibal==2): 
                   return ['M', 'MM', 'MC', 'C']
               else: 
                   return ['M', 'MM', 'MC', 'CC']
        if(side=='left'): #Boat is on left side, have to move the boat to the right side
           if(missionary==3): #if there are 3 missionaries on the right side
               if(cannibal>=2): 
                   return ['MM', 'M', 'MC', 'C', 'CC']
               elif(cannibal==1): 
                   return ['MM', 'M', 'MC', 'C']
               else: 
                   return ['MM', 'M']
           elif(missionary==2): #if there are 2 missionaries on the right side and 1 on the left
               if(cannibal==2):
                   return ['MM', 'M', 'MC', 'C', 'CC']
               else: #does not satisfy constraint
                   return []; 
           elif(missionary==1): #if there is 1 missionarie on the right side and 2 on the left
               if(cannibal==1):
                   return ['M', 'MC', 'C']
               else: #does not satisfy constraint
                   return []; 
           else: #No missionaries on the right side, 3 on the left side
               if(cannibal>=2): #if there are 2 or 3 cannibals on the right side, 0 or 1 on the left
                   return ['C', 'CC']
               else: #if there is 1 cannibal on the right, 2 on the left
                   return ['C']
        else:
            return None

    def result(self, state, action):
        """
        Return the state after executing the action from the list of actions from above function
        """
        missionary = state[0]
        cannibal = state[1]
        side = state[2]

        if (side=='left'):
            if action=='M':
                missionary=missionary-1
                side='right'
            elif action=='MM':
                missionary=missionary-2
                side='right'
            elif action=='MC':
                missionary=missionary-1
                cannibal=cannibal-1
                side='right'
            elif action=='C':
                cannibal=cannibal-1
                side='right'
            else:
                cannibal=cannibal-2
                side='right';
        else:
            if action=='M':
                missionary=missionary+1
                side='left'
            elif action=='MM':
                missionary=missionary+2
                side='left'
            elif action=='MC':
                missionary=missionary+1
                cannibal=cannibal+1
                side='left'
            elif action=='C':
                cannibal=cannibal+1
                side='left'
            else:
                cannibal=cannibal+2
                side='left'
        
        state = list(state)
        state[0] = missionary
        state[1] = cannibal
        state[2] = side
        state = tuple(state)

        self.state=state
        return self.state
      
    def goal_test(self, state):
        """
        Check if the given state is the goal state or not
        """
        if state==self.goal:
            return True
        else:
            return False

initial_state=(3, 3, 'left')
goal_state=(0, 0, 'right')

m_and_c = Search_Problem(initial_state, goal_state)

uninformed_strategies = {"Uniform cost search" : uniform_cost_search,
                        "Iterative deepening search" : iterative_deepening_search}

informed_strategies = {"Greedy best-first search" : greedy_best_first_graph_search, 
                        "A* search" : astar_search,
                        "Recursive best-first search": recursive_best_first_search}

for strategy in uninformed_strategies:
    print("\n", strategy)
    solution = uninformed_strategies[strategy](m_and_c)
    print("\n", solution.path())

for strategy in informed_strategies:
    print("\n", strategy)
    solution = informed_strategies[strategy](m_and_c, m_and_c.h)
    print("\n", solution.path())