from typing import Iterator, Mapping, Tuple

from absl import app
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np




################################## Game Mechanics ###################################
def new_game_state(size: np.unsignedinteger) -> jnp.array:
  game_state: jnp.ndarray = jnp.ones([2, size, size])
  return game_state

def place_blue_piece(game_state: jnp.ndarray, x: np.unsignedinteger, y: np.unsignedinteger):
  game_state[0][y][x] = 0

def place_red_piece(game_state: jnp.ndarray, x: np.unsignedinteger, y: np.unsignedinteger):
  game_state[1][x][y] = 0

def check_free(game_state: jnp.ndarray, x: np.unsignedinteger, y: np.unsignedinteger):
  game_state[0][y][x] * game_state[1][x][y]

def check_win(game_state: jnp.ndarray, color: np.unsignedinteger) -> bool:
  tempState = game_state[color].copy()
  size = len(tempState)
  for i in range(1, size):
    for j in range(size-1):
      newTempState = tempState[i][j] + (tempState[i-1][j] + tempState[i-1][j+1])
      tempState.at[i,j].set(newTempState)
    tempState.at[i,size-1].set(tempState[i][size-1] * tempState[i-1][size-1])
  return jnp.sum(tempState[size-1])

def swap(game_state: jnp.ndarray):
  game_state[0], game_state[1] = game_state[1], game_state[0]
################################## END Game Mechanics ###################################


def main(_):
  game_size = 11
  initial_game_state = new_game_state(game_size)

  batch_size = 1000
  game_states    = jnp.tile(initial_game_state, (batch_size, 1, 1, 1))
  game_turnColor = jnp.ones(batch_size, dtype=jnp.uint8)*game_size
  
  auto_batch_checkwin = jax.vmap(check_win)
  print(auto_batch_checkwin(game_states, game_turnColor))

if __name__ == "__main__":
  app.run(main)
