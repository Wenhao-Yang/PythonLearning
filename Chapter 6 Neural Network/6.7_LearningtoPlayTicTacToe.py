#!/usr/bin/env python
# encoding: utf-8

"""
@Author: yangwenhao
@Contact: 874681044@qq.com
@Software: PyCharm
@File: 6.7_LearningtoPlayTicTacToe.py
@Time: 2019/2/25 10:42
@Overview: In the recipe, we will try to use a neural network to learn the optimal response for a number of different boards. The reason is that the game is a deterministic game and the optimal moves are already known.
"""
import tensorflow as tf
import matplotlib.pyplot as plt
import csv
import random
import numpy as np

batch_size = 50

# Define the function that outputs the Tic Tac Toe boards with Xs and Os
def print_board(board):
    symbols = ['O', ' ', 'X']
    board_plus1 = [int(x) + 1 for x in board]
    print(' ' + symbols[board_plus1[0]] + ' | ' + symbols[board_plus1[1]] + ' | ' + symbols[board_plus1[2]])
    print('___ ___ ___')
    print(' ' + symbols[board_plus1[3]] + ' | ' + symbols[board_plus1[4]] + ' | ' + symbols[board_plus1[5]])
    print('___ ___ ___')
    print(' ' + symbols[board_plus1[6]] + ' | ' + symbols[board_plus1[7]] + ' | ' + symbols[board_plus1[8]])

# Define a function that will return a new board and optimal response position under a transformation
def get_symmetry(board, response, transformation):
    """
    :param board:list of integers 9 long: opposing mark = -1; friendly mark = 1; empty space = 0
    :param response:
    :param transformation: One of five transformations on a board: rotate180, rotate90, rotate270, flip_v, flip_h
    :return: tuple:(new_board, new_response)
    """
    if transformation == 'rotate180':
        new_response = 8 - response
        return (board[::-1], new_response)
    elif transformation == 'rotate90':
        new_response = [6, 3, 0, 7, 4, 1, 8, 5, 2].index(response)
        tuple_board = list(zip(board[6:9], board[3:6], board[0:3]))
        return ([value for item in tuple_board for value in item], new_response)
    elif transformation == 'rotate270':
        new_response = [2, 5, 8, 1, 4, 7, 0, 3, 6].index(response)
        tuple_board = list(zip(board[0:3], board[3:6], board[6:9]))[::-1]
        return ([value for item in tuple_board for value in item], new_response)
    elif transformation == 'flip_v':
        new_response = [6, 7, 8, 3, 4, 5, 0, 1, 2].index(response)
        return (board[6:9] + board[3:6] + board[0:3], new_response)
    elif transformation == 'flip_h':
        new_response = [2, 1, 0, 5, 4, 3, 8, 7, 6].index(response)
        new_board = board[::-1]
        return (new_board[6:9] + new_board[3:6] + new_board[0:3], new_response)
    else:
        raise ValueError('Method not implemented.')

# Define a function that load the .csv file with the f boards and responses and store it as a list of tuples
def get_move_from_csv(csv_file):
    """
    :param csv_file: csv file location
    :return: moves: list of moves with index of best response
    """
    moves = []
    with open(csv_file, 'rt') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            moves.append(([int(x) for x in row[0:9]], int(row[9])))
    return(moves)

# Define a function that will return a randomly transformed board and response
def get_rand_move(moves, rand_transforms=2):
    (board, response) = random.choice(moves)
    possible_transforms = ['rotate180', 'rotate90', 'rotate270', 'flip_v', 'flip_h']
    for i in range(rand_transforms):
        random_transform = random.choice(possible_transforms)
        (board, response) = get_symmetry(board, response, random_transform)
    return(board, response)

# Initialize the graph
sess = tf.Session()
moves = get_move_from_csv('../LocalData/base_tic_tac_toe_moves.csv')
train_length = 500
train_set = []
for t in range(train_length):
    train_set.append(get_rand_move(moves))

# The best move for the following board will be play at index number six
test_board = [-1, 0, 0, 1, -1, -1, 0, 0, 1]
train_set = [x for x in train_set if x[0] != test_board]

# Create model function
def init_weight(shape):
    return (tf.Variable(tf.random_normal(shape=shape)))
def model(X, A1, A2, bias1, bias2):
    layer1 = tf.nn.sigmoid(tf.add(tf.matmul(X, A1), bias1))
    layer2 = tf.add(tf.matmul(layer1, A2), bias2)
    return (layer2)

# Declare variables
X = tf.placeholder(dtype=tf.float32, shape=[None, 9])
Y = tf.placeholder(dtype=tf.int32, shape=[None])
A1 = init_weight([9, 81])
bias1 = init_weight([81])
A2 = init_weight([81, 9])
bias2 = init_weight([9])
model_output = model(X, A1, A2, bias1, bias2)

# Loss function
loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=model_output, labels=Y))
train_step = tf.train.GradientDescentOptimizer(0.025).minimize(loss)
prediction = tf.argmax(model_output, 1)

# Loop
init = tf.global_variables_initializer()
sess.run(init)
loss_vec = []
for i in range(10000):
    rand_indices = np.random.choice(len(train_set), size=batch_size)
    batch_data = [train_set[i] for i in rand_indices]
    x_input = [x[0] for x in batch_data]
    y_target = np.array([y[1] for y in batch_data])

    sess.run(train_step, feed_dict={X: x_input, Y: y_target})

    temp_loss = sess.run(loss, feed_dict={X: x_input, Y: y_target})
    loss_vec.append(np.sqrt(temp_loss))

    if (i + 1) % 500 == 0:
        print('Iteration: ' + str(i + 1) + '. Loss = ' + str(temp_loss))

plt.plot(loss_vec, 'k-', label='Loss')
plt.title('Loss (MSE) per Generation')
plt.xlabel('Generation')
plt.ylabel('Cross Entropy Loss Value')
plt.legend(loc='upper right')
plt.show()

# Test the model
test_boards = [test_board]
feed_dict = {X: test_boards}
logits = sess.run(model_output, feed_dict=feed_dict)
predictions = sess.run(prediction, feed_dict=feed_dict)
print('The test board result is' + str(predictions))

# Define function that know when to stop asking for more moves
def check(board):
    wins = [[0, 1, 2], [3, 4, 5], [6, 7, 8], [0, 3, 6], [1, 4, 7], [2, 5, 8], [0, 4, 8], [2, 4, 6]]
    for i in range(len(wins)):
        if board[wins[1][0]]==board[wins[i][1]]==board[wins[i][2]]==1.:
            return (1)
        elif board[wins[1][0]]==board[wins[i][1]]==board[wins[i][2]]==-1.:
            return (1)
    return (0)
# Loop and play games with the model
game_tracker = [0., 0., 0., 0., 0., 0., 0., 0., 0.]
win_logical = False
num_moves = 0
while not win_logical:
    player_index = input('Input index of your move(0-8):')
    num_moves += 1
    game_tracker[int(player_index)] = 1.
    if check(game_tracker)==1 or num_moves>=5:
        print('\n==Game over!==')
        win_logical = True
    [potential_moves] = sess.run(model_output, feed_dict={X: [game_tracker]})
    # Find all allow moves where game_tracker is 0
    allowed_moves = [ix for ix,x in enumerate(game_tracker) if x==0.0]
    # Find best move bu taking argmax of logits if they are in allowed moves
    model_move = np.argmax([x if ix in allowed_moves else -999.0 for ix,x in enumerate(potential_moves)])

    # Add model move
    game_tracker[int(model_move)] = -1.
    print('Model has moved.')
    print_board(game_tracker)

    print(game_tracker)
    if check(game_tracker)==1 or num_moves>=5:
        print('\n==Game over!==')
        win_logical = True