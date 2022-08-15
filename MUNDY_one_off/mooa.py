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



#jax.config.update('jax_platform_name', 'cpu')

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
  network_parameters = net.init(jax.random.PRNGKey(int(time())), hex.new_game_state())
  try:
    network_parameters = pickle.load(open(filename, 'rb'))
  except Exception as e:
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

  Use this for considering which move to make in a real game.
  '''
  predicted_probabilities = predict_raw_probability(network_parameters, current_board_state)
  # If red is playing, subtract from one so we're always trying to maximize the score
  predicted_probabilities = jnp.where(
    current_turn_color,
    jnp.subtract(1, predicted_probabilities),
    predicted_probabilities
    )

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
    i, j = jnp.unravel_index(index, (hex.board_size, hex.board_size))

    # Check for a winning state
    # Override anything the AI comes up with
    b0 = jnp.where(
      hex.check_win(game_state, color),
      hex.next_color(color),
      temp_s_predicted_probabilities
    )
    # Let the super AI think through the next move
    next_game_state = hex.place_piece(game_state, i, j, color)
    if level > 0:
      s_a_predicted_probabilities, s_s_predicted_probabilities = super_AI(current_network_parameters, next_game_state, hex.next_color(color), level-1)
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
    # Override everything if a player wins. 
    # If blue wins, force to a 1. Likewise, 0 for red
    # The crux of the training process!
    b0 = b0.at[i,j].set(
      jnp.where(
        hex.check_win(next_game_state, color),
        hex.next_color(color),
        b0[i,j]
      )
    )
    return b0
  #end superAI_prediction

  s_predicted_probabilities = jax.lax.fori_loop(
    0, hex.board_size**2,
    calculate_SuperAI_prediction,
    s_predicted_probabilities
  )
  # End iterating over all cells

  return a_predicted_probabilities, s_predicted_probabilities
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
# b = hex.new_game_state()
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

  @partial(jax.jit, static_argnums=(1,))
  def generate_turn_batch(random_key, batch_size = 500):
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
  if evaluate:
    random_key = jax.random.PRNGKey(int(time()))
    batch = generate_turn_batch(random_key, batch_size=50)
    print("Loss: %f" % (loss(current_network_parameters, batch)))
  # end evaluation

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
    save_model()


###################################  END MAIN   #########################################










if __name__ == "__main__":
  app.run(main)
