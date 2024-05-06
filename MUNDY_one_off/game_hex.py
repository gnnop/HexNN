import jax.numpy as jnp
import jax.scipy.signal as signal
import colorama
from game import *

class Hex(ZeroSumTwoPlayerTiledGame):

    def __init__(self, board_size):
        self.board_size = board_size

    def initial_state(self) -> GameState:
        return jnp.ones([2, self.board_size, self.board_size], dtype=jnp.int8)

    def place_piece(self, game_state: GameState, x: int, y: int, color: int) -> GameState:
        return game_state.at[color, jnp.where(color,x,y), jnp.where(color,y,x)].set(0)

    def take_turn(self, game_state: GameState, game_action: GameAction) -> tuple[GameState, GameReward]:
        y = game_action.at[1].get()
        x = game_action.at[2].get()
        color = game_action.at[0].get()
        new_state = self.place_piece(game_state, x, y, color)
        reward = self.end_condition_met(new_state)  # TODO inefficient
        return new_state, reward

    def check_free(self, game_state: GameState, x: int, y: int):
        return game_state[0][y][x] * game_state[1][x][y]

    def end_condition_met(self, game_state: GameState) -> GameDone:
        # TODO speed up with knowledge of the player
        return self.check_win(game_state, 0) | self.check_win(game_state, 1)

    def is_valid_action(self, game_state: GameState, game_action: GameAction) -> GameCondition:
        """Check if placing a piece at position (x, y) for the given color is a valid move."""    
        # Extract x, y coordinates and the color from the game_action data.
        y = game_action.at[1].get()
        x = game_action.at[2].get()
        color = game_action.at[0].get()

        # Use the check_free method to determine if the spot is unoccupied.
        # The check_free should return True if the spot is free.
        return GameCondition(self.check_free(game_state, x, y)).astype(jnp.bool_)


    def check_win(self, game_state: GameState, color: jnp.int8) -> GameDone:
        x = jnp.where(color, game_state[1][0], game_state[0][0])
        x = x.astype(jnp.int8)
        temp_state = jnp.zeros((self.board_size, self.board_size), dtype=jnp.int8)
        temp_state = temp_state.at[0].set(x)
        kernel = jnp.array([[0, 1, 1], [1, 1, 1], [1, 1, 0]])
        for _ in range(self.board_size * self.board_size // 2):
            temp_state = signal.convolve2d(temp_state, kernel, mode='same')
            temp_state = jnp.minimum(temp_state, 1)
            temp_state = jnp.multiply(temp_state, jnp.where(color, game_state[1], game_state[0]))
        return jnp.sum(temp_state[-1]) == 0

    def print_game_state(self, game_state: GameState):
        s = colorama.Fore.RED + '-' * (self.board_size * 2 + 1) + colorama.Fore.RESET + '\n'
        for i in range(self.board_size):
            s += ' ' * i
            s += colorama.Fore.BLUE + '\\' + colorama.Fore.RESET
            for j in range(self.board_size):
                character = '•'
                if game_state[0][i][j] == 0:
                    character = colorama.Fore.BLUE + 'B' + colorama.Fore.RESET
                elif game_state[1][j][i] == 0:
                    character = colorama.Fore.RED + 'R' + colorama.Fore.RESET
                s += character + ' '
            s += colorama.Fore.BLUE + '\\' + colorama.Fore.RESET + '\n'
        s += ' ' * i + ' '
        s += colorama.Fore.RED + '-' * (self.board_size * 2 + 1) + colorama.Fore.RESET
        print(s)

    def get_valid_action_mask(self, game_state: GameState) -> chex.Array:
        return jnp.multiply(game_state[0], jnp.transpose(game_state[1])).flatten().astype(jnp.uint8)

    def get_board_shape(self) -> tuple[int, int]:
        '''
        The dimensions of the board
        '''
        return (self.board_size, self.board_size)