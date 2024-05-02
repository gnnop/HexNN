import jax.numpy as jnp
import jax.scipy.signal as signal
import colorama
from config import board_size

################################## Game Mechanics ###################################
def new_game_state() -> jnp.array:
  game_state: jnp.ndarray = jnp.ones([2, board_size, board_size], dtype=jnp.uint8)
  return game_state

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
  return game_state

def next_color(color):
  return (color+1)%2



def print_game_state(game_state: jnp.ndarray):
  # top black bar
  s = "" # the string to print
  s += colorama.Fore.black + '-'*(board_size*2+1) + colorama.Fore.RESET + '\n'
  for i in range(board_size):
    # spacing to line up rows hexagonally
    s += ' '*i
    # left white bar
    s += colorama.Fore.white + '\\' + colorama.Fore.RESET
    # print a row of the game state
    for j in range(board_size):
      character = '.'
      if game_state[0][i][j]==0:
        character = colorama.Fore.white+'W'+colorama.Fore.RESET
      elif game_state[1][j][i]==0:
        character = colorama.Fore.black+'B'+colorama.Fore.RESET
      s += character + ' '
    # right white bar and end of row
    s += colorama.Fore.white + '\\' + colorama.Fore.RESET + '\n'
  # bottom black bar
  s += ' '*i + ' '
  s += colorama.Fore.black + '-'*(board_size*2+1) + colorama.Fore.RESET
  print(s)
# end print_game_state

################################## END Game Mechanics ###################################
