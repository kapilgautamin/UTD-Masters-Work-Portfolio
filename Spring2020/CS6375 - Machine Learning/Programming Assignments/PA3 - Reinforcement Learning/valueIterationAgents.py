# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


import mdp, util

from learningAgents import ValueEstimationAgent
import collections

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.availableStates = self.mdp.getStates()
        self.runValueIteration()

    def runValueIteration(self):
        # Write value iteration code here
        "*** YOUR CODE HERE ***"
        # print("getStates",self.availableStates)
        # print("runValueIteration START")
        for i in range(self.iterations): 
            newValues = util.Counter()
            for state in self.availableStates:
                # q_state_values = []            
                # availableActions = self.mdp.getPossibleActions(state)       #('north', 'west', 'south', 'east')
                # for action in availableActions:
                #     q_state_values.append(self.getQValue(state,action))
                # vstar = max(q_state_values)         #V* = MAX (Q*(s,a))
                # newValues[state] = vstar
                best_action = self.getAction(state)
                if best_action:
                    newValues[state] = self.getQValue(state,best_action)
            self.values = newValues.copy()
            # print("Completed for iteration ",i)
        # print("Value iteration complete")
        # print("runValueIteration END")
                    
                

    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        # print("Value in",state,self.values[state])
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
          q-state(s,a) = sum(transition_prob * (reward + nextStepValue))

          #Ex. getTransitionStatesAndProbs for current state = (3, 0), action = east =  [((3, 0), 0.9), ((3, 1), 0.1)]
          # A = [R(3,0) + V(3,0)] * 0.9 
          # B = [R(3,1) + V(3,1)] * 0.1
          # return A+B 
        """
        "*** YOUR CODE HERE ***"
        # util.raiseNotDefined()
        # print("computeQValueFromValues START")
        qStateValue = 0
        tsp = self.mdp.getTransitionStatesAndProbs(state,action)
        # print(f"current state = {state}, action = {action}, getTransitionStatesAndProbs : {tsp}")
        
        for transition in tsp:
            nextState = transition[0]
            nextStateTransitionProb = transition[1]
            nextStateReward = self.mdp.getReward(state,action,nextState)
            nextStateValue = self.discount * self.getValue(nextState)   #in the future, so discounted
            # print(state,action,nextState,nextStateReward,nextStateValue,nextStateTransitionProb * (nextStateReward + nextStateValue))
            qStateValue += nextStateTransitionProb * (nextStateReward + nextStateValue)
        # print("qstateValue",qStateValue)
        # print("computeQValueFromValues END")
        return qStateValue
        
                        

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.

          Basically, what action to perform in 'prersent' from 
          (updated) self.values after each runValueIteration
        """
        "*** YOUR CODE HERE ***"
        # print("computeActionFromValues START")
        if self.mdp.isTerminal(state):
            # print("computeActionFromValues TERMINAL END")
            return None

        availableActions = self.mdp.getPossibleActions(state)       #('north', 'west', 'south', 'east')
        max_val = self.getQValue(state,availableActions[0])
        best_action = availableActions[0]
        
        for action in availableActions:
            qvalue = self.getQValue(state,action)
            if qvalue > max_val:
                max_val = qvalue
                best_action = action
        
        # print(f"state = {state}, best_Action = {best_action}")
        # print("computeActionFromValues END")
        return best_action


    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)

class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        for i in range(self.iterations): 
            newValues = util.Counter()
            for state in self.availableStates:
                best_action = self.getAction(state)
                if best_action:
                    newValues[state] = self.getQValue(state,best_action)
            cyclic_change = self.availableStates[i % len(self.availableStates)]
            # print(cyclic_change,self.mdp.isTerminal(cyclic_change))
            if self.mdp.isTerminal(cyclic_change) is not None:
                self.values[cyclic_change] = newValues[cyclic_change]

class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        
        util.raiseNotDefined()

