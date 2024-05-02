

# The number of tiles along one axis of the board
board_size = 5

# The number of cells in the board
board_area = board_size*board_size

'''
The number of elements in the game state, concatenating the following properties into a fixed length vector
    * one element for each space on the board: 0 if that space is occupied by white and 1 otherwise.
    * one element for each space on the board: 0 if that space is occupied by black and 1 otherwise.
    * one element for each space on the board: 0 if that space is occupied and 1 otherwise.
    * 2 elements: a one-hot encoding of the turn. (White then black)
'''
game_state_size = board_area \
                + board_area \
                + board_area \
                + 2


# The number of elements in the internal representation/hidden state/abstract embedding space of the model
state_size = game_state_size