from game import *
import jax.numpy as jnp

class TicTacToe(Game):
    def __init__(self):
        # Initialize with an empty game_state
        self.initial_state = GameState([[0, 0, 0], [0, 0, 0], [0, 0, 0]])

    def initial_state(self) -> GameState:
        return self.initial_state

    def take_turn(self, game_state: GameState, game_action: GameAction) -> tuple[GameState, GameReward]:
        row, col, current_player = game_action.at[0].get(), game_action.at[1].get(), game_action.at[2].get()
        new_board = game_state.at[row, col].set(current_player)
        reward = self.calculate_reward(new_board)
        return new_board, reward

    def end_condition_met(self, game_state: GameState) -> bool:
        return jnp.any(jnp.abs(game_state.sum(axis=0)) == 3) or \
               jnp.any(jnp.abs(game_state.sum(axis=1)) == 3) or \
               abs(game_state.trace()) == 3 or \
               abs(jnp.fliplr(game_state).trace()) == 3 or \
               jnp.all(game_state != 0)

    def is_valid_action(self, game_state: GameState, game_action: GameAction) -> bool:
        row, col = game_action.at[0], game_action.at[1]
        return game_state[row, col] == 0

    def calculate_reward(self, game_state: GameState) -> GameReward:
        return jnp.any(jnp.abs(game_state.sum(axis=0)) == 3) or \
               jnp.any(jnp.abs(game_state.sum(axis=1)) == 3) or \
               abs(game_state.trace()) == 3 or \
               abs(jnp.fliplr(game_state).trace()) == 3

