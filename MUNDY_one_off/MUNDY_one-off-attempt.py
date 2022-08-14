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
from functools import partial
import pickle



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


######################################   MAIN   #########################################
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

  def place_piece(game_state: jnp.ndarray, x: jnp.unsignedinteger, y: jnp.unsignedinteger, color: jnp.unsignedinteger):
    return jnp.where(
      color,
      place_red_piece(game_state, x, y),
      place_blue_piece(game_state, x, y)
    )

  def check_free(game_state: jnp.ndarray, x: jnp.unsignedinteger, y: jnp.unsignedinteger):
    return game_state[0][y][x] * game_state[1][x][y]

  def free_cells(game_state: jnp.ndarray):
    return jnp.multiply(game_state[0], jnp.transpose(game_state[1]))

  # @jax.jit
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

  def next_color(color):
    return (color+1)%2

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





  # SETUP

  # Make the network and optimiser.
  net = hk.without_apply_rng(hk.transform(net_fn))
  network_parameters = net.init(jax.random.PRNGKey(int(time())), new_game_state())
  try:
    network_parameters = pickle.load(open( 'trained-model.dat', 'rb'))
  except Exception as e:
    print("Warning: unable to open saved parameters")
    print(e)
  print(net)

  opt = optax.adam(1e-3)
  opt_state = opt.init(network_parameters)

  # Colorama for code coloring
  colorama.init()


  def predict_raw_probability(
    network_parameters: hk.Params,
    current_board_state: jnp.ndarray,
  ) -> np.ndarray:
    '''
    Finds what the AI thinks the probabilities of players winning are
    0 -> Red wins
    1 -> Blue wins
    '''
    predicted_probabilities = net.apply(network_parameters, current_board_state)
    predicted_probabilities = predicted_probabilities[0]
    return predicted_probabilities
  # end predict_raw_probability

  def predict_probability(
    network_parameters: hk.Params,
    current_board_state: jnp.ndarray,
    current_turn_color: jnp.unsignedinteger
  ) -> np.ndarray:
    '''
    Finds what the AI thinks the probabilities of players winning are
    0 -> Red wins
    1 -> Blue wins

    Differences between predict_raw_probability:
      Shifts the probabilities so the "best move" is always the maximum
      Filters out illegal moves
    '''
    predicted_probabilities = predict_raw_probability(network_parameters, current_board_state)
    # If red is playing, subtract from one so we're always trying to maximize the score
    predicted_probabilities = jnp.where(
      current_turn_color,
      jnp.subtract(1, predicted_probabilities),
      predicted_probabilities
      )

    # Filter out illegal moves
    predicted_probabilities = jnp.multiply(free_cells(current_board_state).astype(jnp.float32).transpose(), predicted_probabilities)

    return predicted_probabilities
  # end predict_probability


  @partial(jax.jit, static_argnums=(3,))
  def super_AI(
      current_network_parameters: hk.Params,
      game_state: jnp.ndarray, 
      color: jnp.unsignedinteger, 
      level=2):
    '''
    Creates a super powerful version of the AI,
    which is most likely still dumb
    '''

    '''
    Prefixes: 
    a_ just your average AI
    s_ super AI
    '''

    a_predicted_probabilities = predict_raw_probability( # average predicted probabilities
      current_network_parameters,
      game_state
    )

    s_predicted_probabilities = a_predicted_probabilities # super predicted probabilites
    # Compute the super AI's predicted probabilites
    # This is dumb; just a few layers of BFS in the game tree
    # Overridden by checking for winning game states
    def calculate_SuperAI_prediction(index, temp_s_predicted_probabilities: jnp.ndarray):
      '''
      A subroutine to update s_predicted_probabilites
      '''
      i, j = jnp.unravel_index(index, (board_size, board_size))

      # Check for a winning state
      # Override anything the AI comes up with
      b0 = jnp.where(
        check_win(game_state, color),
        next_color(color),
        temp_s_predicted_probabilities
      )
      # Let the super AI think through the next move
      if level > 0:
        next_game_state = place_piece(game_state, i, j, color)
        s_a_predicted_probabilities, s_s_predicted_probabilities = super_AI(current_network_parameters, next_game_state, next_color(color), level-1)
        b0 = b0.at[i,j].set(
          jnp.where(
            color,
            jnp.maximum(
              b0[i][j],
              jnp.max(
                s_s_predicted_probabilities
              )
            ),
            jnp.minimum(
              b0[i][j],
              jnp.min(
                s_s_predicted_probabilities
              ) # min
            ), # minimum
          ) # where
        ) # set
      #end if level > 0
      return b0

    #end sr
    s_predicted_probabilities = jax.lax.fori_loop(
      0, board_size**2,
      calculate_SuperAI_prediction,
      s_predicted_probabilities
    )
    # End iterating over all cells

    return a_predicted_probabilities, s_predicted_probabilities
  # end super_AI

  # Try out the super AI
  # b = new_game_state()
  # network_parameters = net.init(jax.random.PRNGKey(int(time())), b)
  # print(super_AI(network_parameters, b, 0))



  def estimate_best_move(
    network_parameters: hk.Params,
    current_board_state: jnp.ndarray,
    current_turn_color: jnp.unsignedinteger
  ) -> Tuple:
    predicted_probabilities = predict_probability(
      network_parameters,
      current_board_state,
      current_turn_color,
    )
    # Without any illegal move, now try to find the most ideal one
    index = predicted_probabilities.argmax()
    index_unraveled = jnp.unravel_index(index, predicted_probabilities.shape)
    return index_unraveled
  # end estimate_best_move

  # Try it out with a new game and random network parameters
  # b = new_game_state()
  # network_parameters = net.init(jax.random.PRNGKey(int(time())), b)
  # print(estimate_best_move(network_parameters, b, 0))



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


  # while True:
  #   network_parameters = net.init(jax.random.PRNGKey(int(time()*100)), b)
  #   s = play_benchmark(network_parameters)
  #   print_game_state(s)

  ###############################   TRAINING    #####################################
  @timer_func
  def train_me(
    current_network_parameters: hk.Params,
    current_opt_state: optax.OptState
  ) -> hk.Params:

    current_game_state = new_game_state()
    current_color = 0

    @jax.jit
    def generate_turn_batch(random_key, batch_size = 150):
      batch = jnp.tile(new_game_state(), (batch_size*2,1,1,1))
      def body_function(i, a):
        '''
        a0: batch
        a1: random key
        '''
        b, rk = a
        gs=b[i]

        # Get a random position
        x, y = jax.random.randint(rk, shape=(2,), minval=0, maxval=board_size, dtype=jnp.uint8); k, rk = jax.random.split(rk)
        # Place a blue piece at that random position
        gs1 = jnp.where(
          check_win(gs, 0),
          new_game_state(),
          place_blue_piece(gs,x,y) 
        )

        
        # Get a random position
        x, y = jax.random.randint(rk, shape=(2,), minval=0, maxval=board_size, dtype=jnp.uint8); k, rk = jax.random.split(rk)
        # Place a red piece at that position
        gs2 = jnp.where(
          check_win(gs1, 1),
          new_game_state(),
          place_red_piece(gs1,x,y) 
        )
        b.at[i*2].set(gs1)
        b.at[i*2+1].set(gs2)
        return (b, rk)
      # end body_function
      # fori_loop prevents loop unrolling (a default feature in JAX)
      r = jax.lax.fori_loop(
        0, batch_size,
        body_function,
        (batch, random_key)
      )
      new_batch = r[0]
      new_key = r[1]
      return jnp.asarray(new_batch)
    # end generate_turn_batch

    @jax.jit
    def loss(params: hk.Params, batch: jnp.ndarray):
      '''
      Sum-squared difference

      Batch: 
        0-predicted 1-expected
          for all trials in batch
            game_state rows
              game_state columns
      For example, a 9-size board has the batch shape (2,81,9,9)
      '''
      # For each index i between 0 and the batch size
      def bf(i, l):
        predicted, expected = super_AI(params,batch[i],i%2)
        l = l + jnp.sum(
            jnp.square(
              jnp.subtract(
                predicted,
                expected
              )
            )
          )
        return l
      # end bf


      l = jax.lax.fori_loop(
        0, len(batch),
        bf,
        0
      )
      return l
    # end loss

    @jax.jit
    def update(
        params: hk.Params,
        opt_state: optax.OptState,
        batch: jnp.ndarray
    ) -> Tuple[hk.Params, optax.OptState]:
      """Learning rule (stochastic gradient descent)."""
      grads = jax.grad(loss)(params, batch)
      r = opt.update(grads, opt_state)
      updates, opt_state = r
      new_params = optax.apply_updates(params, updates)
      return new_params, opt_state
    # end update

    # Training
    random_key = jax.random.PRNGKey(int(time()))
    batch = generate_turn_batch(random_key)
    next_network_parameters, next_opt_state = update(
      current_network_parameters,
      current_opt_state,
      batch
    )

    # Evaluation
    random_key = jax.random.PRNGKey(int(time()))
    batch = generate_turn_batch(random_key)
    print("Loss: %f" % (loss(current_network_parameters, batch)))
    # end evaluation

    return next_network_parameters, next_opt_state
  # end train_me

  # We're back in the main loop
  # Start the training process
  while True:
    network_parameters, opt_state = train_me(network_parameters, opt_state)
    # Save the model for further analysis later
    file = open('trained-model.dat', 'wb')
    pickle.dump(network_parameters, file)
    file.close()


###################################  END MAIN   #########################################










if __name__ == "__main__":
  app.run(main)
