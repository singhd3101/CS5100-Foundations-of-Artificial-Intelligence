# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for 
# educational purposes provided that (1) you do not distribute or publish 
# solutions, (2) you retain this notice, and (3) you provide clear 
# attribution to UC Berkeley, including a link to 
# http://inst.eecs.berkeley.edu/~cs188/pacman/pacman.html
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero 
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and 
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"

        ghostPosition = []
        for ghost in newGhostStates:
            ghostPosition.append(ghost.getPosition())

        ghostScore = 0.0
        for position in ghostPosition:
            distance = manhattanDistance(position, newPos)
            ghostScore -= 1.0 / (2 * (1.0 if (distance == 0) else distance))

        foodScore = 0.0
        foodList = newFood.asList()
        for food in foodList:
            foodScore += 1.0 / manhattanDistance(food, newPos)

        return successorGameState.getScore() + foodScore + ghostScore

def scoreEvaluationFunction(currentGameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
      This class provides some common elements to all of your
      multi-agent searchers.  Any methods defined here will be available
      to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

      You *do not* need to make any changes here, but you can if you want to
      add functionality to all your adversarial search agents.  Please do not
      remove anything, however.

      Note: this is an abstract class: one that should not be instantiated.  It's
      only partially specified, and designed to be extended.  Agent (game.py)
      is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.

          Here are some method calls that might be useful when implementing minimax.

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game
        """
        "*** YOUR CODE HERE ***"

        #calling maximizer for starting node as pacman with depth 0, agentID 0

        return maximizer(gameState, 0, 0, self.depth, self.evaluationFunction)[1]

        util.raiseNotDefined()

def maximizer(state, depth, agentIndex, maxDepth, evaluationFunction):
    actions = state.getLegalActions(agentIndex)
    if len(actions) == 0 or depth == maxDepth:
        return evaluationFunction(state), None
    v = -float("inf")
    nodeAction = None

    for thisAction in actions:
        successorValue = minimizer(state.generateSuccessor(0, thisAction), depth, 1, maxDepth, evaluationFunction)[0]
        if successorValue > v:
            v, nodeAction = successorValue, thisAction
    return v, nodeAction

def minimizer(state, depth, agentIndex, maxDepth, evaluationFunction):
    actions = state.getLegalActions(agentIndex)
    if len(actions) == 0 or depth == maxDepth:
        return evaluationFunction(state), None

    v = float("inf")
    nodeAction = None

    for thisAction in actions:
        if agentIndex == state.getNumAgents() - 1:
            successorValue = maximizer(state.generateSuccessor(agentIndex, thisAction),
                                       depth + 1, 0, maxDepth, evaluationFunction)[0]
        else:
            successorValue = minimizer(state.generateSuccessor(agentIndex, thisAction),
                                       depth, agentIndex + 1, maxDepth, evaluationFunction)[0]
        if successorValue < v:
            v, nodeAction = successorValue, thisAction
    return v, nodeAction

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"

        # calling alphaValue for starting node as pacman with depth 0, agentID 0,
        # alpha as negative infinity and beta as positive infinity

        return alphaValue(gameState, 0, 0, self.depth, self.evaluationFunction,
                          -float("inf"), float("inf"))[1]

        util.raiseNotDefined()

def alphaValue(state, depth, agentIndex, maxDepth, evaluationFunction, alpha, beta):
    actions = state.getLegalActions(agentIndex)
    if len(actions) == 0 or depth == maxDepth:
        return evaluationFunction(state), None
    v = -float("inf")
    nodeAction = None

    for thisAction in actions:
        successorValue = betaValue(state.generateSuccessor(0, thisAction),
                                   depth, 1, maxDepth, evaluationFunction, alpha, beta)[0]
        if successorValue > v:
            v, nodeAction = successorValue, thisAction
        if v > beta:
            return v, nodeAction
        alpha = max(alpha, v)
    return v, nodeAction

def betaValue(state, depth, agentIndex, maxDepth, evaluationFunction, alpha, beta):
    actions = state.getLegalActions(agentIndex)
    if len(actions) == 0 or depth == maxDepth:
        return evaluationFunction(state), None

    v = float("inf")
    nodeAction = None

    for thisAction in actions:
        if agentIndex == state.getNumAgents() - 1:
            successorValue = alphaValue(state.generateSuccessor(agentIndex, thisAction),
                                        depth + 1, 0, maxDepth, evaluationFunction, alpha, beta)[0]
        else:
            successorValue = betaValue(state.generateSuccessor(agentIndex, thisAction),
                                        depth, agentIndex + 1, maxDepth, evaluationFunction, alpha, beta)[0]
        if successorValue < v:
            v, nodeAction = successorValue, thisAction
        if v < alpha:
            return v, nodeAction
        beta = min(beta, v)
    return v, nodeAction

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        "*** YOUR CODE HERE ***"

        # calling maxValueExpectimax for starting node as pacman with depth 0, agentID 0

        return maxValueExpectimax(gameState, 0, 0, self.depth, self.evaluationFunction)[1]

        util.raiseNotDefined()

def maxValueExpectimax(state, depth, agentIndex, maxDepth, evaluationFunction):
    actions = state.getLegalActions(agentIndex)
    if len(actions) == 0 or depth == maxDepth:
        return evaluationFunction(state), None
    v = -float("inf")
    nodeAction = None

    for thisAction in actions:
        successorValue = expValueExpectimax(state.generateSuccessor(0, thisAction),
                                   depth, 1, maxDepth, evaluationFunction)[0]
        if successorValue > v:
            v, nodeAction = successorValue, thisAction
    return v, nodeAction

def expValueExpectimax(state, depth, agentIndex, maxDepth, evaluationFunction):
    actions = state.getLegalActions(agentIndex)
    if len(actions) == 0 or depth == maxDepth:
        return evaluationFunction(state), None

    v = float("inf")
    nodeValues = []
    nodeAction = None

    for thisAction in actions:
        if agentIndex == state.getNumAgents() - 1:
            successorValue = maxValueExpectimax(state.generateSuccessor(agentIndex, thisAction),
                                       depth + 1, 0, maxDepth, evaluationFunction)[0]
        else:
            successorValue = expValueExpectimax(state.generateSuccessor(agentIndex, thisAction),
                                       depth, agentIndex + 1, maxDepth, evaluationFunction)[0]
        nodeValues.append(successorValue)
    v = sum(nodeValues)/len(nodeValues)
    return v, nodeAction

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """

    """
      Description: Most of the code is same as of reflexAgent. Made some
      minor tweaks to enhance the evaluation function by adding weights
      to the current game score and the evaluated score. 
      The evaluated score consist of inverted manhattan distance of pacman
      with food, scared ghosts and normal ghost. Score increases on basis of
      distance from food pellets and scared ghosts, decreases for normal ghost.
      Modified factors in food score and scared ghost score calculation
      for score improvement.
    """
    newPos = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood()
    newGhostStates = currentGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

    "*** YOUR CODE HERE ***"
    ghostPosition = []
    for ghost in newGhostStates:
        ghostPosition.append(ghost.getPosition())

    ghostScore = 0.0
    for position in ghostPosition:
        distance = manhattanDistance(position, newPos)
        ghostScore -= 1.0 / 0.5 * (1.0 if (distance == 0) else distance)

    foodScore = 0.0
    foodList = newFood.asList()
    for food in foodList:
        foodScore += 1.0 / (0.5 * manhattanDistance(food, newPos))

    scareScore = 0.0
    for scare in newScaredTimes:
        if scare != 0:
            scareScore += 1.0 / 0.5 * scare

    return (0.25 * currentGameState.getScore()) + (0.75 * (foodScore + ghostScore + scareScore))

    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction

