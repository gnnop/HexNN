# hex_game.py
import jax
import jax.numpy as jnp
import jax.scipy.signal as signal

def init_game_state(board_size=8):
    """Initialize the game state."""
    return jnp.ones([2, board_size, board_size], dtype=jnp.float32), 0, 1  # Game state, current color, is_first_move

def place_white_piece(game_state: jnp.ndarray, x: jnp.unsignedinteger, y: jnp.unsignedinteger):
  return game_state.at[0,y,x].set(0)

def place_black_piece(game_state: jnp.ndarray, x: jnp.unsignedinteger, y: jnp.unsignedinteger):
  return game_state.at[1,x,y].set(0)

def place_piece(game_state: jnp.ndarray, x: jnp.unsignedinteger, y: jnp.unsignedinteger, color: jnp.unsignedinteger):
  return jnp.where(
    color,
    place_black_piece(game_state, x, y),
    place_white_piece(game_state, x, y)
  )

@jax.jit
def check_free(game_state, x, y):
    """Check if the cell at (x, y) is free."""
    return jnp.logical_and(game_state[0, y, x] > 0, game_state[1, x, y] > 0)

@jax.jit
def check_win(game_state, current_color):
    """Check if the current player has won."""
    # Implement the win-checking logic here
    return False  # Placeholder

@jax.jit
def swap_sides(game_state):
    """Swap the sides of the game."""
    return game_state[::-1, :, :]

def get_network_input(game_state, current_color, is_first_move):
    """Prepare game state for neural network input."""
    flattened_state = game_state.flatten()
    current_player_vector = jnp.array([1, 0] if current_color == 0 else [0, 1])
    first_move_vector = jnp.array([is_first_move])
    return jnp.concatenate([flattened_state, current_player_vector, first_move_vector])
