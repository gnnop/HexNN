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
from time import time



jax.config.update('jax_platform_name', 'cpu')




def timer_func(func):
    # This function shows the execution time of
    # the function object passed
    def wrap_func(*args, **kwargs):
        t1 = time()
        result = func(*args, **kwargs)
        t2 = time()
        print(f'Function {func.__name__!r} executed in {(t2-t1):.4f}s')
        return result
    return wrap_func




###################################### AI Model #########################################
def net_fn(game_state: jnp.ndarray):
  x = game_state.astype(jnp.float32)
  size = game_state.shape[1]
  num_spots = size*size
  mlp = hk.Sequential([
      hk.Flatten(),
      hk.Linear(num_spots), jax.nn.relu,
      hk.Linear(num_spots), jax.nn.relu,
      hk.Linear(num_spots),
      hk.Reshape((size,size))
  ])
  return mlp(x)









def load_dataset(batch_size: int):
  # TODO
  pass







def main(_):



  # Game config
  board_size = 9

  ################################## Game Mechanics ###################################
  def new_game_state() -> jnp.array:
    game_state: jnp.ndarray = jnp.ones([2, board_size, board_size], dtype=jnp.uint8)
    return game_state

  def place_blue_piece(game_state: jnp.ndarray, x: jnp.unsignedinteger, y: jnp.unsignedinteger):
    return game_state.at[0,y,x].set(0)

  def place_red_piece(game_state: jnp.ndarray, x: jnp.unsignedinteger, y: jnp.unsignedinteger):
    return game_state.at[1,x,y].set(0)

  def check_free(game_state: jnp.ndarray, x: jnp.unsignedinteger, y: jnp.unsignedinteger):
    return game_state[0][y][x] * game_state[1][x][y]

  def free_cells(game_state: jnp.ndarray):
    return jnp.multiply(game_state[0], jnp.transpose(game_state[1]))

  @jax.jit
  def check_win(game_state: jnp.ndarray, color: jnp.unsignedinteger) -> bool:
    x = jnp.where(color, game_state[1][0], game_state[0][0])
    x = x.astype(jnp.float32)
    tempState = jnp.zeros((board_size, board_size), dtype=jnp.float32)
    tempState = tempState.at[0].set(x)
    kernel = jnp.array([
      [0, 1, 1],
      [1, 1, 1],
      [1, 1, 0]
    ])
    for i in range(int(board_size*board_size/2)):
      tempState = signal.convolve2d(tempState, kernel, mode='same')
      tempState = jnp.minimum(tempState, 1)
      tempState = jnp.multiply(tempState, jnp.where(color, game_state[1], game_state[0]))
    return jnp.sum(tempState[board_size-1]) == 0

  def swap(game_state: jnp.ndarray):
    t = game_state[1]
    game_state = game_state.at[1].set(game_state[0])
    game_state = game_state.at[0].set(t)

  def print_game_state(game_state: jnp.ndarray):
    # top red bar
    s = "" # the string to print
    s += colorama.Fore.RED + '-'*(board_size*2+1) + colorama.Fore.RESET + '\n'
    for i in range(board_size):
      # spacing to line up rows hexagonally
      s += ' '*i
      # left blue bar
      s += colorama.Fore.BLUE + '\\' + colorama.Fore.RESET
      # print a row of the game state
      for j in range(board_size):
        character = '.'
        if game_state[0][i][j]==0:
          character = colorama.Fore.BLUE+'B'+colorama.Fore.RESET
        elif game_state[1][j][i]==0:
          character = colorama.Fore.RED+'R'+colorama.Fore.RESET
        s += character + ' '
      # right blue bar and end of row
      s += colorama.Fore.BLUE + '\\' + colorama.Fore.RESET + '\n'
    # bottom red bar
    s += ' '*i + ' '
    s += colorama.Fore.RED + '-'*(board_size*2+1) + colorama.Fore.RESET
    print(s)
  # end print_game_state

  ################################## END Game Mechanics ###################################







  # Make the network and optimiser.
  net = hk.without_apply_rng(hk.transform(net_fn))
  opt = optax.adam(1e-3)


  # Colorama for code coloring
  colorama.init()




  def estimate_best_move(
    network_parameters: hk.Params,
    current_board_state: jnp.ndarray,
    current_turn_color: jnp.unsignedinteger
  ) -> Tuple:
    '''
    Finds what the AI thinks the probabilities of players winning are
    0 -> Red wins
    1 -> Blue wins
    '''
    predicted_probabilities = net.apply(network_parameters, current_board_state)
    predicted_probabilities = predicted_probabilities[0]
    # If red is playing, subtract from one so we're always trying to maximize the score
    predicted_probabilities = jnp.where(
      current_turn_color,
      jnp.subtract(1, predicted_probabilities),
      predicted_probabilities
      )

    # Filter out illegal moves
    predicted_probabilities = jnp.multiply(free_cells(current_board_state).astype(jnp.float32).transpose(), predicted_probabilities)

    # Without any illegal move, now try to find the most ideal one
    index = predicted_probabilities.argmax()
    index_unraveled = jnp.unravel_index(index, predicted_probabilities.shape)
    return index_unraveled
  # end estimate_best_move

  # Try it out with a new game and random network parameters
  b = new_game_state()
  network_parameters = net.init(jax.random.PRNGKey(int(time())), b)
  print(estimate_best_move(network_parameters, b, 0))







  @jax.jit
  def make_best_move(
    current_network_parameters: hk.Params,
    current_board_state: jnp.ndarray,
    current_turn_color: jnp.unsignedinteger
  ) -> jnp.ndarray:
    '''
    Uses best_move to determine the best move for a certain color.
    Makes that move and returns the new game state.
    '''
    # Estimate the best move
    current_best_move = estimate_best_move(current_network_parameters, current_board_state, current_turn_color)

    # Make that move
    next_board_state = jnp.where(
      current_turn_color,
      place_red_piece(current_board_state, current_best_move[0], current_best_move[1]),
      place_blue_piece(current_board_state, current_best_move[0], current_best_move[1])
    )

    return next_board_state
  # end make_best_move

  # Try it out by playing a test game
  @timer_func
  def play_a_game(
    current_network_parameters: hk.Params
  ):
    current_board_state = new_game_state()
    current_board_turn_color = 0
    keep_going = True
    current_turn_count = 0
    while(keep_going):
      current_turn_count += 1

      # Make the best move
      current_board_state = make_best_move(current_network_parameters, current_board_state, current_board_turn_color)

      # Check for a win
      keep_going = not check_win(current_board_state, current_board_turn_color)

      # If not a win, let the next player take a turn
      current_board_turn_color = not current_board_turn_color
    # end while loop
    return (current_board_turn_color, current_turn_count, current_board_state)
  # end play_a_game



  # while True:
  #   network_parameters = net.init(jax.random.PRNGKey(int(time())), b)
  #   final_board_turn_color, final_turn_count, final_board_state = play_a_game(network_parameters)
  #   # Check for a win
  #   if final_board_turn_color:
  #     print("Blue wins!")
  #   else:
  #     print("Red wins!")

  # # Print the results
  # print("-------------- Turn %d --------------" % (final_turn_count))
  # print_game_state(final_board_state)
  # print()


  @timer_func
  @jax.jit
  def play_benchmark(
    current_network_parameters: hk.Params
    ):
    current_board_state = new_game_state()

    def body_function(i, a):
      b0, b1 = a
      b1 = jnp.where(
        check_win(b1, 0),
        new_game_state(),
        make_best_move(b0, b1,0)
      )
      b1 = jnp.where(
        check_win(b1, 1),
        new_game_state(),
        make_best_move(b0, b1, 1)
      )
      return (b0, b1)

    # fori_loop prevents loop unrolling (a default feature in JAX)
    r = jax.lax.fori_loop(
      0, 5000, # Each game averages 100 moves on an 11x11 board. It's playing 100 games?!
      body_function,
      (current_network_parameters, current_board_state)
    )


    # for i in range(1000):
    #   # Make the best move
    #   current_board_state = jnp.where(
    #     check_win(current_board_state, not current_board_turn_color),
    #     new_game_state(),
    #     make_best_move(current_network_parameters, current_board_state, current_board_turn_color)
    #   )
    #   current_board_turn_color = not current_board_turn_color
    # # end for loop
    return r[1]
  # end play_benchmark

  while True:
    network_parameters = net.init(jax.random.PRNGKey(int(time()*100)), b)
    s = play_benchmark(network_parameters)
    print_game_state(s)






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
