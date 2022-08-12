from calendar import c
from typing import Iterator, Mapping, Tuple
from absl import app
import haiku as hk
import jax
import jax.numpy as jnp
from jax.scipy import signal
import numpy as np
import optax
import colorama
import time



jax.config.update('jax_platform_name', 'cpu')


################################## Game Mechanics ###################################
def new_game_state(size: np.unsignedinteger) -> jnp.array:
  game_state: jnp.ndarray = jnp.ones([2, size, size])
  return game_state

def place_blue_piece(game_state: jnp.ndarray, x: np.unsignedinteger, y: np.unsignedinteger):
  return game_state.at[0,y,x].set(0)

def place_red_piece(game_state: jnp.ndarray, x: np.unsignedinteger, y: np.unsignedinteger):
  return game_state.at[1,x,y].set(0)

def check_free(game_state: jnp.ndarray, x: np.unsignedinteger, y: np.unsignedinteger):
  return game_state[0][y][x] * game_state[1][x][y]

def check_win(game_state: jnp.ndarray, color: np.unsignedinteger) -> bool:
  size = game_state.shape[1]
  tempState = jnp.zeros(game_state[color].shape)
  tempState = tempState.at[0].set(game_state[color][0])
  kernel = jnp.array([
    [0, 1, 1],
    [1, 1, 1],
    [1, 1, 0]
  ])
  for i in range(size**2):
    tempState = signal.convolve2d(tempState, kernel, mode='same')
    tempState = jnp.minimum(tempState, 1)
    tempState = jnp.multiply(tempState, game_state[color])
  return jnp.sum(tempState[size-1]) == 0

def swap(game_state: jnp.ndarray):
  t = game_state[1]
  game_state.at[1].set(game_state[0])
  game_state.at[0].set(t)

def print_game_state(game_state: jnp.ndarray):
  size = game_state.shape[1]
  print(colorama.Fore.RED + '-'*(size*2+1) + colorama.Fore.RESET)
  for i in range(size):
    print(' '*i, end='')
    print(colorama.Fore.BLUE + '\\' + colorama.Fore.RESET, end='')
    for j in range(size):
      character = '.'
      if game_state[0][i][j]==0:
        character = colorama.Fore.BLUE+'B'+colorama.Fore.RESET
      elif game_state[1][j][i]==0:
        character = colorama.Fore.RED+'R'+colorama.Fore.RESET

      print(character, end=' ')
    print(colorama.Fore.BLUE + '\\' + colorama.Fore.RESET, end='')

    print()
  print(' '*i, end=' ')
  print(colorama.Fore.RED + '-'*(size*2+1) + colorama.Fore.RESET)


################################## END Game Mechanics ###################################







###################################### AI Model #########################################
def net_fn(game_state: jnp.ndarray):
  size = game_state.shape[1]
  num_spots = size*size
  mlp = hk.Sequential([
      hk.Flatten(),
      hk.Linear(num_spots), jax.nn.relu,
      hk.Linear(num_spots), jax.nn.relu,
      hk.Linear(num_spots),
      hk.Reshape((size,size))
  ])
  return mlp(game_state)






  


def load_dataset(batch_size: int):
  # TODO
  pass







def main(_):
  # Make the network and optimiser.
  net = hk.without_apply_rng(hk.transform(net_fn))
  opt = optax.adam(1e-3)

  # Game config
  board_size = 11


  # Colorama for code coloring
  colorama.init()







  def estimate_best_move(
    network_parameters: hk.Params, 
    current_board_state: jnp.ndarray,
    current_turn_color: jnp.unsignedinteger
  ):
    '''
    Finds what the AI thinks the probabilities of players winning are
    0 -> Red wins
    1 -> Blue wins
    '''
    predicted_probabilities = net.apply(network_parameters, current_board_state)
    predicted_probabilities = predicted_probabilities[0]
    # If red is playing, subtract from one so we're always trying to maximize the score
    if current_turn_color != 0:
      predicted_probabilities = jnp.subtract(1, predicted_probabilities)
    
    # Filter out illegal moves
    size = current_board_state.shape[1]
    for x in range(size):
      for y in range(size):
        if check_free(current_board_state, x, y) == 0:
          predicted_probabilities = predicted_probabilities.at[x,y].set(0)
    
    # Without any illegal move, now try to find the most ideal one
    index = predicted_probabilities.argmax()
    index_unraveled = np.unravel_index(index, predicted_probabilities.shape)
    return index_unraveled

  # Try it out with a new game and random network parameters
  my_board_state = new_game_state(board_size)
  network_parameters = net.init(jax.random.PRNGKey(time.time()), my_board_state)
  print(estimate_best_move(network_parameters, my_board_state, 0))









  def make_best_move(
    network_parameters: hk.Params, 
    current_board_state: jnp.ndarray, 
    current_turn_color: jnp.unsignedinteger
  ):
    '''
    Uses best_move to determine the best move for a certain color.
    Makes that move and returns the new game state.
    '''
    # Estimate the best move
    current_best_move = estimate_best_move(network_parameters, current_board_state, current_turn_color)

    # Make that move
    next_board_state = current_board_state
    if current_turn_color == 0:
      next_board_state = place_blue_piece(next_board_state, current_best_move[0], current_best_move[1])
    else:
      next_board_state = place_red_piece(next_board_state, current_best_move[0], current_best_move[1])
    return next_board_state

  my_board_turn_color = 0
  keep_going = True
  my_turn_count = 0
  while(keep_going):
    my_turn_count += 1

    # Make the best move
    my_board_state = make_best_move(network_parameters, my_board_state, my_board_turn_color)

    # Print the results
    print("-------------- Turn %d --------------" % (my_turn_count))
    print_game_state(my_board_state)
    print()

    # Check for a win
    if my_board_turn_color == 0 and check_win(my_board_state, 0):
      print("Blue wins!")
      keep_going = False
    elif my_board_turn_color == 1 and check_win(my_board_state, 1):
      print("Red wins!")
      keep_going = False

    # If not a win, let the next player take a turn
    my_board_turn_color = not my_board_turn_color



    


  # def play_myself(
  #   network_parameters: hk.Params, 
  #   current_board_state: np.ndarray, 
  #   current_color: np.unsignedinteger, 
  #   depth=5) :
  #   '''
  #   Plays one turn against a suped-up version of itself. 
  #   Returns a tuple with the following:
  #     The game state after the most plausible legal move, assessed by the AI
  #     The probability of winning at each space, assessed by the AI
  #     The probability of winning at each space, assessed by the super-AI.

  #     The super-AI just searches the game tree a few steps ahead

  #   If the game is over, current_board_state returns an empty board
  #   '''
  #   predicted_probabilities = net.apply(network_parameters, current_board_state)
  #   predicted_probabilities_super = predicted_probabilities.copy()
  #   if depth > 0:
  #     for i in range(board_size):
  #       for j in range(board_size):
  #         # Place a piece
  #         next_board_state = current_board_state.copy()
  #         next_color       = 0 # blue=0, red=1
  #         if current_color: # red
  #           place_red_piece(next_board_state, i, j)
  #           if check_win(next_board_state, 1):
  #             predicted_probabilities_super[i][j]=1
  #         else:             # blue
  #           place_blue_piece(next_board_state, i, j)
  #           next_color = 1
  #           if check_win(next_board_state, 0):
  #             predicted_probabilities_super[i][j]=1
  #         if predicted_probabilities_super[i][j] < 1:
  #           opponent_play = play_myself(network_parameters, next_board_state, next_color, depth-1)
  #           predicted_probabilities_super = opponent_play
  #   # Get the next board state
  #   next_board_state = current_board_state.copy()

          





  # game_size = 11
  # initial_game_state = new_game_state(game_size)

  # batch_size = 1000
  # game_states    = jnp.tile(initial_game_state, (batch_size, 1, 1, 1))
  # game_turnColor = jnp.ones(batch_size, dtype=jnp.uint8)*game_size
  
  # auto_batch_checkwin = jax.vmap(check_win)
  # print(auto_batch_checkwin(game_states, game_turnColor))

if __name__ == "__main__":
  app.run(main)
