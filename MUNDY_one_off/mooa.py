# MUNDY_one-off-attempt

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
import hex
from multiprocessing import Pool



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
      hk.Linear(num_spots*2), jax.nn.relu,
      hk.Linear(num_spots),   jax.nn.relu,
      hk.Linear(num_spots),   jax.nn.relu,
      hk.Linear(num_spots),   jax.nn.sigmoid,
      hk.Reshape((size,size))
  ])
  return mlp(x)

net = hk.without_apply_rng(hk.transform(net_fn))

def load_model(filename: str = 'trained-model.dat') -> hk.Params:
  print("Loading MOOA model: %s ... " % filename, end='')
  network_parameters = net.init(jax.random.PRNGKey(int(time())), hex.new_game_state())
  try:
    network_parameters = pickle.load(open(filename, 'rb'))
    print("Loaded!")
  except Exception as e:
    print("Error!")
    print("Warning: unable to open saved parameters")
    print(e)
  print(net)
  return network_parameters


def save_model(network_parameters: hk.Params, filename: str = 'trained-model.dat'):
    # Save the model for further analysis later
    file = open('trained-model.dat', 'wb')
    pickle.dump(network_parameters, file)
    file.close()

def predict_raw_probability(
  network_parameters: hk.Params,
  current_board_state: jnp.ndarray,
) -> np.ndarray:
  '''
  Finds what the AI thinks the probabilities of players winning are
  0 -> Red wins
  1 -> Blue wins

  Good for training
  '''
  predicted_probabilities = net.apply(network_parameters, current_board_state)
  predicted_probabilities = predicted_probabilities[0]
  return predicted_probabilities
# end predict_raw_probability

def filter_illegal_moves(
  current_board_state: jnp.ndarray,
  predicted_probabilities: jnp.ndarray
) -> np.ndarray:

  # Filter out illegal moves
  predicted_probabilities = jnp.multiply(hex.free_cells(current_board_state).astype(jnp.float32).transpose(), predicted_probabilities)

  return predicted_probabilities
# end predict_probability


@partial(jax.jit, static_argnames=['level'])
def super_AI(
    current_network_parameters: hk.Params,
    game_state: jnp.ndarray, 
    color: jnp.unsignedinteger, 
    level=1):
  '''
  Creates a super powerful version of the AI,
  which is most likely still dumb
  '''

  '''
  Prefixes: 
  a_ just your average AI
  s_ super AI
  '''

  # Compute the super AI's predicted probabilites
  # This is dumb; just a few layers of BFS in the game tree
  # Overridden by checking for winning game states
  def calculate_SuperAI_prediction(index: jnp.ndarray):
    '''
    A subroutine to update s_predicted_probabilites
    '''
    j, i = jnp.unravel_index(index, (hex.board_size, hex.board_size))


    # Let the super AI think through the next move
    next_game_state = hex.place_piece(game_state, i, j, color)
    next_color = hex.next_color(color)
    if level > 0:
      s_s_predicted_probabilities = super_AI(current_network_parameters, next_game_state, next_color, level-1)
    else:
      s_s_predicted_probabilities = predict_raw_probability(
        current_network_parameters,
        game_state
      )

    r = jnp.where(
      color,
      jnp.max(
          s_s_predicted_probabilities
        ),
      jnp.min(
          s_s_predicted_probabilities
        )
    ) # where

    # Check for a losing state
    # Override anything the AI comes up with
    r = jnp.where(
      hex.check_win(next_game_state, next_color),
      color,
      r
    )
    # Penalize invalid moves as harshly as losing moves
    r = jnp.where(
      hex.check_free(game_state, i, j),
      r,
      color
    )
    return r


  #end superAI_prediction

  sAI_batch = jax.vmap(calculate_SuperAI_prediction)
  # super predicted probabilites
  s_predicted_probabilities = sAI_batch(jnp.arange(hex.board_size ** 2)).reshape((hex.board_size, hex.board_size))

  return s_predicted_probabilities
# end super_AI

# Try out the super AI
# b = hex.new_game_state()
# network_parameters = net.init(jax.random.PRNGKey(int(time())), b)
# print(super_AI(network_parameters, b, 0))

def estimate_best_move(
  network_parameters: hk.Params,
  current_board_state: jnp.ndarray,
  current_turn_color: jnp.unsignedinteger
) -> Tuple:


  predicted_probabilities = predict_raw_probability(
    network_parameters,
    current_board_state
  )

  # If red is playing, subtract from one so we're always trying to maximize the score
  predicted_probabilities = jnp.where(
    current_turn_color,
    jnp.subtract(1, predicted_probabilities),
    predicted_probabilities
    )

  # filter illegal moves
  predicted_probabilities = filter_illegal_moves(current_board_state, predicted_probabilities)

  # Without any illegal move, now try to find the most ideal one
  index = predicted_probabilities.argmax()
  index_unraveled = jnp.unravel_index(index, predicted_probabilities.shape)
  return index_unraveled
# end estimate_best_move

# Try it out with a new game and random network parameters
# b = hex.new_game_state()
# network_parameters = net.init(jax.random.PRNGKey(int(time())), b)
# print(estimate_best_move(network_parameters, b, 0))


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
    hex.place_red_piece(current_board_state, current_best_move[0], current_best_move[1]),
    hex.place_blue_piece(current_board_state, current_best_move[0], current_best_move[1])
  )

  return next_board_state
# end make_best_move


@timer_func
@jax.jit
def play_benchmark(
  current_network_parameters: hk.Params
  ):
  current_board_state = hex.new_game_state()

  def body_function(i, a):
    b0, b1 = a
    b1 = jnp.where(
      hex.check_win(b1, 0),
      hex.new_game_state(),
      make_best_move(b0, b1,0)
    )
    b1 = jnp.where(
      hex.check_win(b1, 1),
      hex.new_game_state(),
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
  #     hex.check_win(current_board_state, not current_board_turn_color),
  #     hex.new_game_state(),
  #     make_best_move(current_network_parameters, current_board_state, current_board_turn_color)
  #   )
  #   current_board_turn_color = not current_board_turn_color
  # # end for loop
  return r[1]
# end play_benchmark

# Try it out
# while True:
#   network_parameters = net.init(jax.random.PRNGKey(int(time()*100)), b)
#   s = play_benchmark(network_parameters)
#   print_game_state(s)

###############################   TRAINING    #####################################
@timer_func
def train_me(
  current_network_parameters: hk.Params,
  opt: optax.GradientTransformation,
  current_opt_state: optax.OptState,
  evaluate=False
) -> hk.Params:

  batch_size = 1000

  @jax.jit
  def generate_turn_batch(random_key):
    batch = jnp.tile(hex.new_game_state(), (batch_size*2,1,1,1))
    def body_function(i, a):
      '''
      a0: batch
      a1: random key
      '''
      b, rk = a
      gs=b[i]

      # Get a random position
      x, y = jax.random.randint(rk, shape=(2,), minval=0, maxval=hex.board_size, dtype=jnp.uint8); k, rk = jax.random.split(rk)
      # Place a blue piece at that random position
      gs1 = jnp.where(
        hex.check_win(gs, 0),
        hex.new_game_state(),
        hex.place_blue_piece(gs,x,y) 
      )

      
      # Get a random position
      x, y = jax.random.randint(rk, shape=(2,), minval=0, maxval=hex.board_size, dtype=jnp.uint8); k, rk = jax.random.split(rk)
      # Place a red piece at that position
      gs2 = jnp.where(
        hex.check_win(gs1, 1),
        hex.new_game_state(),
        hex.place_red_piece(gs1,x,y) 
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
    return jnp.asarray(new_batch)
  # end generate_turn_batch

  @jax.jit
  def loss(params: hk.Params, inputs: jnp.ndarray, expected_outputs: jnp.ndarray):
    '''
    Sum-squared difference of expected values and predicted ones

    Batch: 
      0-predicted 1-expected
        for all trials in batch
          game_state rows
            game_state columns
    For example, a 9-size board has the batch shape (2,81,9,9)
    '''

    predict_raw_probabilities = jax.vmap(lambda gs: predict_raw_probability(params, gs))

    predicted_vals = predict_raw_probabilities(inputs)

    return jnp.sum(
      jnp.square(
        jnp.subtract(
          expected_outputs,
          predicted_vals
        )
      )
    )
  # end loss

  @jax.jit
  def update(
      params: hk.Params,
      opt_state: optax.OptState,
      inputs: jnp.ndarray,
      expected_outputs: jnp.ndarray,
  ) -> Tuple[hk.Params, optax.OptState]:
    """Learning rule (stochastic gradient descent)."""
    grads = jax.grad(loss)(params, inputs, expected_outputs)
    r = opt.update(grads, opt_state)
    updates, opt_state = r
    new_params = optax.apply_updates(params, updates)
    return new_params, opt_state
  # end update

  # Training data
  random_key = jax.random.PRNGKey(int(time()))
  inputs = generate_turn_batch(random_key)
  turn_colors = jnp.tile(jnp.array([0,1]), (batch_size))
  super_AI_batch = jax.vmap(lambda i, c: super_AI(current_network_parameters, i, c))
  expected_outputs = super_AI_batch(inputs, turn_colors)
  

  # Evaluation
  if evaluate:
    print("Loss: %f" % (loss(current_network_parameters, inputs, expected_outputs)))
  # end evaluation


  # Training
  next_network_parameters, next_opt_state = update(
    current_network_parameters,
    current_opt_state,
    inputs,
    expected_outputs
  )


  return next_network_parameters, next_opt_state
# end train_me




######################################   MAIN   #########################################
def main(_):
  # SETUP
  # Make the network and optimiser.
  network_parameters = load_model()
  opt = optax.adam(1e-3)
  opt_state = opt.init(network_parameters)
  # Colorama for code coloring
  colorama.init()

  # LOOP
  # We're back in the main loop
  # Start the training process
  iterations=0
  while True:
    iterations += 1
    print("Iteration %d" % iterations)
    network_parameters, opt_state = train_me(network_parameters, opt, opt_state, iterations%10 == 2)
    # Save the model for further analysis later
    save_model(network_parameters)


###################################  END MAIN   #########################################










if __name__ == "__main__":
  app.run(main)
