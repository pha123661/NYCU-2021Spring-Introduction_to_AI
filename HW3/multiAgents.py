# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random
import util

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
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and child states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(
            gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(
            len(scores)) if scores[index] == bestScore]
        # Pick randomly among the best
        chosenIndex = random.choice(bestIndices)
        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        The evaluation function takes in the current and proposed child
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        childGameState = currentGameState.getPacmanNextState(action)
        newPos = childGameState.getPacmanPosition()
        newFood = childGameState.getFood()
        newGhostStates = childGameState.getGhostStates()
        newScaredTimes = [
            ghostState.scaredTimer for ghostState in newGhostStates]
        return childGameState.getScore()


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

    def __init__(self, evalFn='scoreEvaluationFunction', depth='2'):
        self.index = 0  # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)


class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent
    """

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.getNextState(agentIndex, action):
        Returns the child game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        # Begin your code (Part 1)
        def minimax(agent, depth, gameState):
            # leaf node
            if gameState.isWin() or gameState.isLose() or depth >= self.depth:
                return self.evaluationFunction(gameState)

            # max layer
            if agent == 0:
                maximum = -1.8446744e+19
                action = None
                for move in gameState.getLegalActions(agent):
                    nextState = gameState.getNextState(agent, move)
                    value = minimax(agent=1, depth=depth, gameState=nextState)
                    if value > maximum:
                        maximum = value
                        action = move
                if depth == 0:  # initial call returns action
                    return action
                else:
                    return maximum

            # min layer
            else:
                next_agent = agent + 1
                if next_agent == gameState.getNumAgents():  # go to next depth
                    next_agent = 0
                    depth += 1
                minimum = 1.8446744e+19
                for move in gameState.getLegalActions(agent):
                    nextState = gameState.getNextState(agent, move)
                    value = minimax(agent=next_agent, depth=depth,
                                    gameState=nextState)
                    minimum = min(minimum, value)
                return minimum

        return minimax(agent=0, depth=0, gameState=gameState)
        # End your code (Part 1)


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        # Begin your code (Part 2)
        ###### prun if alpha >= beta ######

        def ab_search(agent, depth, gameState, alpha, beta):
            # leaf node
            if gameState.isWin() or gameState.isLose() or depth >= self.depth:
                return self.evaluationFunction(gameState)

            # max layer
            if agent == 0:
                maximum = -1.8446744e+19
                action = None
                for move in gameState.getLegalActions(agent):
                    nextState = gameState.getNextState(agent, move)
                    value = ab_search(agent=1, depth=depth,
                                      gameState=nextState, alpha=alpha, beta=beta)
                    if value > maximum:
                        maximum = value
                        action = move

                    # pruning
                    if maximum > alpha:
                        alpha = maximum
                    if maximum > beta:
                        if depth == 0:  # initial call returns action
                            return action
                        else:
                            return maximum

                if depth == 0:
                    return action
                else:
                    return maximum

            # min layer
            else:
                next_agent = agent + 1
                if next_agent == gameState.getNumAgents():  # go to next depth
                    next_agent = 0
                    depth += 1
                minimum = 1.8446744e+19
                for move in gameState.getLegalActions(agent):
                    nextState = gameState.getNextState(agent, move)
                    value = ab_search(agent=next_agent, depth=depth,
                                      gameState=nextState, alpha=alpha, beta=beta)
                    minimum = min(minimum, value)

                    # pruning
                    if minimum < beta:
                        beta = minimum
                    if minimum < alpha:
                        return minimum

                return minimum

        alpha = -1.8446744e+19
        beta = 1.8446744e+19
        return ab_search(agent=0, depth=0, gameState=gameState, alpha=alpha, beta=beta)
        # End your code (Part 2)


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        # Begin your code (Part 3)
        def expectimax(agent, depth, gameState):
            # leaf node
            if gameState.isWin() or gameState.isLose() or depth >= self.depth:
                return self.evaluationFunction(gameState)

            # max layer
            if agent == 0:
                maximum = -1.8446744e+19
                action = None
                for move in gameState.getLegalActions(agent):
                    nextState = gameState.getNextState(agent, move)
                    value = expectimax(agent=1, depth=depth,
                                       gameState=nextState)
                    if value > maximum:
                        maximum = value
                        action = move
                if depth == 0:  # initial call returns action
                    return action
                else:
                    return maximum

            # min layer
            else:
                next_agent = agent + 1
                if next_agent == gameState.getNumAgents():  # go to next depth
                    next_agent = 0
                    depth += 1
                sum = 0
                for move in gameState.getLegalActions(agent):
                    nextState = gameState.getNextState(agent, move)
                    sum += expectimax(agent=next_agent,
                                      depth=depth, gameState=nextState)

                # returns average instead of minimum
                return sum / len(gameState.getLegalActions(agent))

        return expectimax(agent=0, depth=0, gameState=gameState)
        # End your code (Part 3)


def betterEvaluationFunction(currentGameState):
    """
    Your extreme evaluation function
    """
    # Begin your code (Part 4)
    gameState = currentGameState
    score = gameState.getScore()
    pac_pos = gameState.getPacmanPosition()
    K = 7

    Foods = gameState.getFood().asList()
    food_list = []
    for food_pos in Foods:
        tmp = util.manhattanDistance(pac_pos, food_pos)
        food_list.append(tmp)
    food_list.sort()

    if len(food_list) == 0:
        score_food = 1
    else:
        score_food = 0
        K = min(len(food_list), K)
        for i in range(K):
            score_food += food_list[i]
        score_food /= K

    sum_distance = 0
    ghost_around = 0
    for ghost_pos in gameState.getGhostPositions():
        distance = util.manhattanDistance(pac_pos, ghost_pos)
        sum_distance += distance
        if distance == 1:
            ghost_around += 1

    num_capsules = len(gameState.getCapsules())

    score_food = max(score_food, 1)
    sum_distance = max(sum_distance, 1)
    return score + 1/float(score_food) - 1/float(sum_distance) - (ghost_around + num_capsules)
    # End your code (Part 4)


# Abbreviation
better = betterEvaluationFunction
