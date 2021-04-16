import random
import tensorflow as tf
import valueFunction as vfunc
import Globals as g
import Classes as c
import MCTS
import AImodels as trial
import valueFunction as func
import random as rd
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.backend import reshape
from keras.utils.np_utils import to_categorical
import pandas as pd
import numpy as np
import csv

#Creates board, all empty spots wiht zeros
def initBoard():
    board = []
    for i in range(0,19):
        row = []
        for ii in range(0,19):
            row.append(0)
        board.append(row)
    return board


#creates a list of all possible moves (not sure what this is used for, was in tutorial
def getMoves(board):
    moves = []
    for i in range(len(board)):
        for j in range(len(board[i])):
            if board[i][j] == 0:
                moves.append((i, j))
    return moves


#should be called after every move to check if anyone has won
def getWinner(board):
    candidate = 0
    won = 0

    # Check rows
    for i in range(len(board)):
        candidate = 0
        how_many_so_far = 0
        for j in range(len(board[i])):
            # Make sure there are no gaps
            if candidate != board[i][j]:
                how_many_so_far = 1
                candidate = board[i][j]
            else:
                how_many_so_far += 1
            # Determine whether the front-runner has all the slots
            if how_many_so_far == 5 and candidate > 0:
                won = candidate
                #print(candidate, how_many_so_far)
            #print(how_many_so_far)
    if won > 0:
        return won

    # Check columns
    for j in range(len(board[0])):
        candidate = 0
        how_many_so_far = 0
        for i in range(len(board)):

            # Make sure there are no gaps
            if candidate != board[i][j]:
                how_many_so_far = 1
                candidate = board[i][j]
            else:
                how_many_so_far += 1
            # Determine whether the front-runner has all the slots
            if how_many_so_far == 5 and candidate > 0:
                won = candidate

    if won > 0:
        return won

    # Check diagonals
    for i in range(len(board)):
        for j in range(len(board[i])):
            candidate = board[i][j]
            how_many_so_far = 0
            if candidate > 0 and i <= len(board) - 5 and j <= len(board) - 5:
                for d in range(0,5):
                # Make sure there are no gaps
                    if i + d >= len(board) or j + d > len(board) or candidate != board[i + d][j + d]:
                        break
                    else:
                        how_many_so_far += 1
                # Determine whether the front-runner has all the slots
                    if how_many_so_far == 5 and candidate > 0:
                        won = candidate
                        return won


            how_many_so_far = 0
            if candidate > 0 and i <= len(board):
                for d in range(0, 5):
                    # Make sure there are no gaps
                    if  i + d >= len(board) or j - d < 0 or candidate != board[i + d][j - d]:
                        break
                    else:
                        how_many_so_far += 1
                    # Determine whether the front-runner has all the slots
                    if how_many_so_far == 5 and candidate > 0:
                        won = candidate
                        return won
                #print(candidate, how_many_so_far)
            #print(how_many_so_far)
    if won > 0:
        return won

    # Still no winner?
    if (len(getMoves(board)) == 0):
        # It's a draw
        return 0
    else:
        # Still more moves to make
        return -1

#prints ascii board from the board given
def printBoard(board):
    for i in range(len(board)):
        for j in range(len(board[i])):
            mark = '_'
            if board[i][j] == 1:
                mark = 'X'
            elif board[i][j] == 2:
                mark = 'O'
            elif board[i][j] == 3:
                mark = '1'
            elif board[i][j] == 4:
                mark = '2'
            if (j == len(board[i]) - 1):
                print(mark)
            else:
                print(str(mark) + "|", end='')


def movesToBoard(moves):
    board = initBoard()
    for move in moves:
        player = move[0]
        coords = move[1]
        board[coords[0]][coords[1]] = player
    return board

#outputs a list of moves made, "History" that when given to the next function generates the board state
def simulateGame(p1=None, p2=None, rnd=0):
    board = initBoard()
    moveList = []
    playertoMove = 1
    history = []
    tempList = []
    moves = getMoves(board)
    temp = None
    rand = False
    move2 = None
    move1 = None
    move = None

    # Moves keeps track of the available coordinates (x, y) left on the board
    # Board keeps track of the player occupation on board (0, 1, 2)

    while getWinner(board) == -1:
        move = None
        # Chosen move for player1
        if playertoMove == 1 and rand == False:
            moveList = trial.getBestMovesInAnArray(board)
            move = rd.choice(moveList)
        elif playertoMove == 1 and rand == True:
            move = trial.getBestMove(board)
        elif playertoMove == 2 and rand == True:
            # import pdb; pdb.set_trace() n = nextline s = step in c = continue q = quit
            move = trial.getBestMove(board)

        # Player move now occupies board (0, 1, 2)
        board[move.x][move.y] = playertoMove

        # Append playertoMove and coordinates to history
        history.append([playertoMove, [move.x, move.y]])

        # Switch between player 1 and player 2
        playertoMove = 1 if playertoMove == 2 else 2

        # Swap AI approach
        rand = True if rand == False else True

    # moveList = trial.getBestMovesInAnArray(board)
    # move = rd.choice(moveList)

    # print(f"{move.x} {move.y}")
    # print(f"{move2.x} {move2.y}")
    printBoard(movesToBoard(history))
    print()
    return history
#this function creates a board given a history from the previous simulated game function




#ignore this code, trying out the functions

# b = initBoard()
# b[8][8] = 2
# b[7][7] = 2
# b[6][6] = 2
# b[5][5] = 2
# b[4][4]= 2
#
# printBoard(b)
#
# print(getMoves(b))
# print(getWinner(b))


# game_sim = simulateGame()
# print("turns: " + str(len(game_sim)))
# board =  movesToBoard(game_sim)
# printBoard(board)
# print("winner: " + str(getWinner(board)))


#this function is for viewing stats from the list of a lot of games simulated
def gameStats(games, player=1):
    stats = {"win": 0, "loss": 0, "draw": 0}
    #counter = 0
    for game in games:
        result = getWinner(movesToBoard(game))
        #counter += 1
        #print(str(counter) + ":   " + str(result))
        if result == -1:
            continue
        elif result == player:
            stats["win"] += 1
        elif result == 0:
            stats["draw"] += 1
        else:
            stats["loss"] += 1

    winPct = stats["win"] / len(games) * 100
    lossPct = stats["loss"] / len(games) * 100
    drawPct = stats["draw"] / len(games) * 100

    print("Results for player %d:" % (player))
    print("Wins: %d (%.1f%%)" % (stats["win"], winPct))
    print("Loss: %d (%.1f%%)" % (stats["loss"], lossPct))
    print("Draw: %d (%.1f%%)" % (stats["draw"], drawPct))

'''
def bestMove(board):
    combinations = []
    playerCombinations = []
    values = [[0 for i in range(g.width)] for j in range(g.width)]
    playerValues = [[0 for i in range(g.width)] for j in range(g.width)]    
    aiGuessDimensions = c.CoordinatePair(0, 0)
    vfunc.populate(combinations, values)
    vfunc.populate(playerCombinations, playerValues)
    RePopulate(board, values, playerValues, combinations, playerCombinations)
    vfunc.updateValues(board, combinations, values, True)
    vfunc.updateValues(board, playerCombinations, playerValues, True)

    curPotential = c.Value(0, 0, 0)
    for i in range(g.width):
        for j in range(g.width):
      #Takes the Aggregate Potential of both player's possible moves
            if (values[j][i].thirdPriority > -1 and playerValues[j][i].thirdPriority > -1 and (values[i][j]+playerValues[j][i]) > curPotential):
                curPotential.firstPriority = values[j][i].firstPriority + playerValues[j][i].firstPriority
                curPotential.secondPriority = values[j][i].secondPriority + playerValues[j][i].secondPriority
                curPotential.thirdPriority = values[j][i].thirdPriority + playerValues[j][i].thirdPriority
                aiGuessDimensions = c.CoordinatePair(j, i)

    return aiGuessDimensions

def getMove(board):
    combinations = []
    playerCombinations = []
    values = [[0 for i in range(g.width)] for j in range(g.width)]
    playerValues = [[0 for i in range(g.width)] for j in range(g.width)]    
    aiGuessDimensions = c.CoordinatePair(0, 0)
    vfunc.populate(combinations, values)
    vfunc.populate(playerCombinations, playerValues)
    RePopulate(board, values, playerValues, combinations, playerCombinations)
    vfunc.updateValues(board, combinations, values, True)
    vfunc.updateValues(board, playerCombinations, playerValues, True)

    #Potential of Aggressive moves
    curPotential = c.Value(0, 0, 0)

    for i in range(g.width):
        for j in range(g.width):
        #Check to make sure spot is not taken
            if (values[j][i].thirdPriority > -1 and values[j][i] > curPotential):
                #If first priority is higher, OR if first priority is the same AND second
                #priority is higher, OR if first AND second priorities are the same but THIRD
                #priority is higher, make this the preferred move
                curPotential = values[j][i]
                aiGuessDimensions = c.CoordinatePair(j, i)

    #Defensive moves
    playerCurPotential = c.Value(0, 0, 0)

    #updateValues(board, playerCombinations, playerValues, curPlayer);
    for i in range(g.width):
        for j in range(g.width):
            if (playerValues[j][i].thirdPriority > -1 and playerValues[j][i] > playerCurPotential):
                playerCurPotential = playerValues[j][i]
                if (playerCurPotential > curPotential):
                    aiGuessDimensions = c.CoordinatePair(j, i)

    return aiGuessDimensions

#Runs a Monty-Carlo search using the moves it finds off of getBestMovesInAnArray
#A Little slow, but accurate
def getMCTS_Move(board):
    #Variables
    combinations = []
    playerCombinations = []
    values = [[0 for i in range(g.width)] for j in range(g.width)]
    playerValues = [[0 for i in range(g.width)] for j in range(g.width)]
    #Populate
    func.populate(combinations, values)
    func.populate(playerCombinations, playerValues)
    #Correct according to the current boardstate 
    RePopulate(board, values, playerValues, combinations, playerCombinations)
    #Update the new values
    func.updateValues(board, combinations, values, True)
    func.updateValues(board, playerCombinations, playerValues, True)

    return MCTS.MCTS_Move(board, combinations, values, playerCombinations, playerValues, False)

def RePopulate(board, values, playerValues, combinations, playerCombinations):
    for i in range(len(board)):
        for j in range(len(board[i])):
            if (board[i][j]==1):
                vfunc.removePotential(combinations, values, c.CoordinatePair(i, j))
                values[i][j].thirdPriority = -1
                playerValues[i][j].thirdPriority = -1
            elif (board[i][j]==2):
                vfunc.removePotential(playerCombinations, playerValues, c.CoordinatePair(i, j))
                values[i][j].thirdPriority = -1
                playerValues[i][j].thirdPriority = -1
'''
#run 1000 games and show their stats,
#should be a "control" to show improvement with model
#games = [simulateGame() for _ in range(1000)]
#gameStats(games)
#print()
#gameStats(games, player=2)
def highlightWin(board):
    candidate = 0
    won = 0
    winning_combination = []
    # Check rows
    for i in range(len(board)):
        candidate = 0
        how_many_so_far = 0
        winning_combination = []
        for j in range(len(board[i])):
            # Make sure there are no gaps
            if candidate != board[i][j]:
                winning_combination = []
                how_many_so_far = 1
                candidate = board[i][j]
                if(candidate > 0):
                    winning_combination.append((i ,j))
            else:
                how_many_so_far += 1
                if (candidate > 0):
                    winning_combination.append((i, j))

            # Determine whether the front-runner has all the slots
            if how_many_so_far == 5 and candidate > 0:
                won = candidate
                for move in winning_combination:
                    board[move[0]][move[1]] = candidate + 2
                # print(candidate, how_many_so_far)
            # print(how_many_so_far)
    if won > 0:
        return won

    # Check columns
    for j in range(len(board[0])):
        candidate = 0
        how_many_so_far = 0
        winning_combination = []
        for i in range(len(board)):

            # Make sure there are no gaps
            if candidate != board[i][j]:
                winning_combination = []
                how_many_so_far = 1
                candidate = board[i][j]
                if (candidate > 0):
                    winning_combination.append((i, j))
            else:
                how_many_so_far += 1
                if (candidate > 0):
                    winning_combination.append((i, j))
            # Determine whether the front-runner has all the slots
            if how_many_so_far == 5 and candidate > 0:
                won = candidate
                for move in winning_combination:
                    board[move[0]][move[1]] = candidate + 2

    if won > 0:
        return won

    # Check diagonals
    winning_combination = []
    for i in range(len(board)):
        for j in range(len(board[i])):
            candidate = board[i][j]
            how_many_so_far = 0
            winning_combination = []
            if candidate > 0 and i <= len(board) - 5 and j <= len(board) - 5:
                for d in range(0, 5):
                    # Make sure there are no gaps
                    if i + d >= len(board) or j + d > len(board) or candidate != board[i + d][j + d]:
                        break
                    else:
                        how_many_so_far += 1
                        if (candidate > 0):
                            winning_combination.append((i +d, j + d))
                    # Determine whether the front-runner has all the slots
                    if how_many_so_far == 5 and candidate > 0:
                        won = candidate
                        for move in winning_combination:
                            board[move[0]][move[1]] = candidate + 2
                        return won
            winning_combination = []
            how_many_so_far = 0
            if candidate > 0 and i <= len(board):
                for d in range(0, 5):
                    # Make sure there are no gaps
                    if i + d >= len(board) or j - d < 0 or candidate != board[i + d][j - d]:
                        break
                    else:
                        how_many_so_far += 1
                        if (candidate > 0):
                            winning_combination.append((i+d, j-d))
                    # Determine whether the front-runner has all the slots
                    if how_many_so_far == 5 and candidate > 0:
                        won = candidate
                        for move in winning_combination:
                            board[move[0]][move[1]] = candidate + 2
                        return won
                # print(candidate, how_many_so_far)
            # print(how_many_so_far)
    if won > 0:
        return won

    # Still no winner?
    if (len(getMoves(board)) == 0):
            # It's a draw
        return 0
    else:
        # Still more moves to make
        return -1


#print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

games = []
for i in range(5000):
    sim = simulateGame()
    print(i)
    games.append(sim)

with open('simulated_games.csv', mode='w') as sim_games:
    sim_games = csv.writer(sim_games, delimiter=',', escapechar=' ', quoting=csv.QUOTE_NONE)
    for game in games:
        sim_games.writerow(game)


# gamesNP = np.array(games)
# df = pd.DataFrame(gamesNP)
# df.to_csv('sim_games.csv')

# for i in games:
#
#     print(i)
#     board = movesToBoard(i)
#     printBoard(board)
#     print(getWinner(board))
#
# gameStats(games)
# print()
# gameStats(games, player=2)


# df = pd.read_csv('simulated_games.csv')
# print(df)
#now here we should write, train, test and print out the model results
'''
def getModel():
    numCells = 361 #because 19x19
    outcomes = 3
    This is how we tell the model that we want it to output an array of probabilities.
     These probabilities project the model’s confidence in each of the three game outcomes for a given board.The three outcomes being win, draw and loss, refer to tutorial
    model = Sequential()
    model.add(Dense(200, activation='relu', input_shape=(361, ))) #i changed the 9 here from input shape to 361, The 200 in the dense should be replaced but I'm not sure with what number
    #if 200 was used for a 3x3 board, I imagine it must be a lot for the 19x19 but don't go overboard, increment slowly and see results
    model.add(Dropout(0.2)) #this is used for avoiding overfitting, could be changed to higher if the test accuracy is bad and training is very high
    model.add(Dense(125, activation='relu'))#this is the second hdiden layer, they usually get smaller the more layers
    model.add(Dense(75, activation='relu'))#the relu activation function is used, you can look up activation functions and experiment with others that you think might work well
    model.add(Dropout(0.1))
    model.add(Dense(25, activation='relu'))
    model.add(Dense(outcomes, activation='softmax')) #normalizes the output into probabilites, the model will evaluate whether it thinks its winning, losing or drawing so far
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['acc'])
    return model


#check tutorial for explanation
def gamesToWinLossData(games):
    X = []
    y = []
    for game in games:
        winner = getWinner(movesToBoard(game))
        for move in range(len(game)):
            X.append(movesToBoard(game[:(move + 1)]))
            y.append(winner)

    X = np.array(X).reshape((-1, 9))
    y = to_categorical(y)

    # Return an appropriate train/test split
    trainNum = int(len(X) * 0.8)
    return (X[:trainNum], X[trainNum:], y[:trainNum], y[trainNum:])
'''
'''From here on copy and paste the tutorial code and run it every time the tutorial shows output and let us know how it goes'''