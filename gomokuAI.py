import random
import re
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
from keras.models import load_model
import pandas as pd
import numpy as np
import csv
from timer import Timer

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
        #bot1_Roll = rd.randint(1, 7)
        #bot2_Roll = rd.randint(1, 19)

        # Chosen move for player1
        if playertoMove == 1 and rand == False:
            moveList = trial.getBestMovesInAnArray(board)
            move = rd.choice(moveList)
        
        elif playertoMove == 2: #and rand == True:
            # import pdb; pdb.set_trace() n = nextline s = step in c = continue q = quit
            moveList = trial.getBestMovesInAnArray(board)
            move = rd.choice(moveList)

        # Player move now occupies board (0, 1, 2)
        board[move.x][move.y] = playertoMove

        # Append playertoMove and coordinates to history
        history.append([playertoMove, [move.x, move.y]])

        # Switch between player 1 and player 2
        playertoMove = 1 if playertoMove == 2 else 2

        # Swap AI approach
        #rand = True if rand == False else True

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

def gameHuman(board):

    currPlayer = 1
    history = []
    moveList = []
    move = None
    botOpener = True

    human = int(input("Would you like to be player 1 or 2 (Enter 1 or 2):"), 3)

    print(human)

    while getWinner(board) == -1:
        
        print("in getWinner currPlayer = ", currPlayer)

        if(currPlayer == human):
            inputX = int(input("X: "), 20)
            inputY = int(input("Y: "), 20)

            board[inputX][inputY] = currPlayer

            history.append([currPlayer, [inputX, inputY]])

            currPlayer = 1 if currPlayer == 2 else 2

            botOpener = False

            printBoard(board)

        else:
            
            if(botOpener == True):

                moveList = trial.getBestMovesInAnArray(board)
                move = rd.choice(moveList)

                board[move.x][move.y] = currPlayer

                history.append([currPlayer, [move.x, move.y]])

                currPlayer = 1 if currPlayer == 2 else 2

                botOpener = False

                print("Bot opening move")
                printBoard(board)

            else:
                move = trial.getMCTS_Move(board)

                board[move.x][move.y] = currPlayer

                history.append([currPlayer, [move.x, move.y]])

                currPlayer = 1 if currPlayer == 2 else 2

                print("bot MCTS move")
                printBoard(board)


def dataMagic(garbageData):
    test_string = garbageData

    testStr = test_string.replace(" ", '').replace("[", '').replace("]", '') #remove all but numbers and commas
    testStr = testStr.split(',') #split by commas, leave only numbers

    history = []
   
    for index, element in enumerate(testStr):
        if index % 3 == 0:
            history.append([int(element), [int(testStr[index + 1]), int(testStr[index + 2])]])

    return(history)

def getModel():
    numCells = 361
    outcomes = 3
    model = Sequential()
    model.add(Dense(1020, activation='relu', input_shape=(361, )))
    model.add(Dense(1020, activation='relu'))
    model.add(Dense(1020, activation='relu'))
    model.add(Dense(1020, activation='relu'))
    model.add(Dense(1020, activation='relu'))
    model.add(Dense(outcomes, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['acc'])
    return model

def gamesToWinLossData(games):
    X = []
    y = []
    for game in games:
        winner = getWinner(movesToBoard(game))
        for move in range(len(game)):
            X.append(movesToBoard(game[:(move + 1)]))
            y.append(winner)

    X = np.array(X).reshape((-1, 361))
    y = to_categorical(y)
    
    # Return an appropriate train/test split
    trainNum = int(len(X) * 0.8)
    return (X[:trainNum], X[trainNum:], y[:trainNum], y[trainNum:])

def simmulateNNGame(p1=None, p2=None, rnd=0):
    history = []
    board = initBoard()
    playerToMove = 1
    temp = None


    while getWinner(board) == -1:

        #Chose a move
        if playerToMove == 1 and p1 != None:
            move = bestMove(board, p1, playerToMove, rnd)
            print(playerToMove, ": [", move[0], ", ", move[1], "]")

        elif playerToMove == 2 and p2 != None:
            move = bestMove(board, p2, playerToMove, rnd)
            print(playerToMove, ": [", move[0], ", ", move[1], "]")

        else:
            temp = trial.getBestMove(board)
            move = (temp.x, temp.y)
            #print(type(move))
            #moves = getMoves(board)
            #move = moves[random.randint(0, len(moves) - 1)]
            print(playerToMove, ": [", move[0], ", ", move[1], "]")

        # Make the move
        board[move[0]][move[1]] = playerToMove

        # Add the move to the history
        history.append((playerToMove, move))

        # Switch the active player
        playerToMove = 1 if playerToMove == 2 else 2
    
    printBoard(movesToBoard(history))
    print(history)
    return history

#print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

def bestMove(board, model, player, rnd=0):
    scores = []
    moves = getMoves(board)
    #print(moves)

    for i in range(len(moves)):
        future = np.array(board)
        future[moves[i][0]][moves[i][1]] = player
        prediction = model.predict(future.reshape((-1, 361)))[0]

        if player == 1:
            winPrediction = prediction[1]
            lossPrediction = prediction[2]
        else:
            winPrediction = prediction[2]
            lossPrediction = prediction[1]
        drawPrediction = prediction[0]

        #import pdb; pdb.set_trace()

        if winPrediction - lossPrediction > 0:
            scores.append(winPrediction - lossPrediction)
        else:
            scores.append(drawPrediction - lossPrediction)
    #print(scores)
    
    # Choose the best move with a random factor
    #bestMoves = np.flip(np.argsort(scores))
    bestMoves = np.array(scores)
    bestMove = np.amax(bestMoves)
    index = np.where(bestMoves == np.amax(bestMoves))
    #import pdb; pdb.set_trace()

    
    #for i in range(len(bestMoves)):
        #if random.random() * rnd < 0.3:
    #import pdb; pdb.set_trace()
    if len(index[0]) > 1:
        return moves[index[0][0].item()]

    return moves[index[0].item()]
    
    
    # Choose a move completely at random
    return moves[random.randint(0, len(moves) - 1)]

'''
games = []
for i in range(10):
    sim = simulateGame()
    print(i)
    games.append(sim)
t.stop()

with open('testingConversion.csv', mode='a') as sim_games:
    sim_games = csv.writer(sim_games, delimiter=',', escapechar=' ', quoting=csv.QUOTE_NONE, lineterminator='\n')
    for game in games:
        sim_games.writerow(game)
'''

#board = initBoard()
#gameHuman(board)


'''
t = Timer()
t.start()

with open('new.csv', 'r') as csvfile:
    csvtext = csvfile.readlines()

myList = []
for line in csvtext:
    myList.append(line)



totalGames = []

counter = 0
for i in myList:
    history = dataMagic(myList[counter])
    board = movesToBoard(history)
    totalGames.append(history)
    counter += 1
t.stop()
print()
print(counter)


model = getModel()
X_train, X_test, y_train, y_test = gamesToWinLossData(totalGames)
modelHistory = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=100, batch_size=100)

# save model
model.save('my_model100bs37.h5')
del model
'''

# returns compiled model
# identical to the previous one


model = load_model('my_model.h5')

games2 = [simmulateNNGame(p1=model) for _ in range(10)]
gameStats(games2)

#print(myList[9])
#testHistory = dataMagic(myList[9])

#board = movesToBoard(testHistory)
#printBoard(board)


'''
history = dataMagic(myList[2])
print("it's working and shit")
board = movesToBoard(history)
printBoard(board)
'''





#for j in range(len(data[0])):
#    print(data[0][j])



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