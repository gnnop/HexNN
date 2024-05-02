import jax.numpy as jnp
import jax.scipy.signal as signal
import colorama
from game import *


# Game configuration constants
BOARD_SIZE = 5

class Hex(Game):

    def __init__(self, board_size=BOARD_SIZE):
        self.board_size = board_size

    def new_game_state(self) -> GameState:
        return jnp.ones([2, self.board_size, self.board_size], dtype=jnp.uint8)

    def place_piece(self, game_state: GameState, x: int, y: int, color: int) -> GameState:
        if color == 0:  # Blue
            return game_state.at[0, y, x].set(0)
        else:  # Red
            return game_state.at[1, x, y].set(0)

    def take_turn(self, game_state: GameState, game_action: GameAction) -> tuple[GameState, GameReward]:
        x = game_action.at[0].get()  # Assuming game_action is a structured array containing these elements
        y = game_action.at[1].get()
        color = game_action.at[2].get()
        new_state = self.place_piece(game_state, x, y, color)
        reward = self.check_win(new_state, color)  # Example: Define reward based on winning or not
        return new_state, reward

    def check_free(self, game_state: GameState, x: int, y: int):
        return game_state[0][y][x] * game_state[1][x][y]

    def end_condition_met(self, game_state: GameState) -> bool:
        for color in [0, 1]:
            if self.check_win(game_state, color):
                return True
        return False

    def is_valid_action(self, game_state: GameState, x: int, y: int, color: int) -> bool:
        """Check if placing a piece at position (x, y) for the given color is a valid move."""    
        # Extract x, y coordinates and the color from the game_action data.
        x = game_action.at[0].get()
        y = game_action.at[1].get()
        color = game_action.at[2].get()

        # Check if the coordinates are within the valid range for the board.
        if x < 0 or x >= self.board_size or y < 0 or y >= self.board_size:
            return False

        # Use the check_free method to determine if the spot is unoccupied.
        # The check_free should return True if the spot is free.
        return bool(self.check_free(game_state, x, y))


    def check_win(self, game_state: GameState, color: int) -> bool:
        x = game_state[1][0] if color else game_state[0][0]
        x = x.astype(jnp.float32)
        temp_state = jnp.zeros((self.board_size, self.board_size), dtype=jnp.float32)
        temp_state = temp_state.at[0].set(x)
        kernel = jnp.array([[0, 1, 1], [1, 1, 1], [1, 1, 0]])
        for _ in range(self.board_size * self.board_size // 2):
            temp_state = signal.convolve2d(temp_state, kernel, mode='same')
            temp_state = jnp.minimum(temp_state, 1)
            temp_state = jnp.multiply(temp_state, game_state[1] if color else game_state[0])
        return jnp.sum(temp_state[-1]) == 0

    def print_game_state(self, game_state: GameState):
        s = colorama.Fore.RED + '-' * (self.board_size * 2 + 1) + colorama.Fore.RESET + '\n'
        for i in range(self.board_size):
            s += ' ' * i
            s += colorama.Fore.BLUE + '\\' + colorama.Fore.RESET
            for j in range(self.board_size):
                character = '.'
                if game_state[0][i][j] == 0:
                    character = colorama.Fore.BLUE + 'B' + colorama.Fore.RESET
                elif game_state[1][j][i] == 0:
                    character = colorama.Fore.RED + 'R' + colorama.Fore.RESET
                s += character + ' '
            s += colorama.Fore.BLUE + '\\' + colorama.Fore.RESET + '\n'
        s += ' ' * i + ' '
        s += colorama.Fore.RED + '-' * (self.board_size * 2 + 1) + colorama.Fore.RESET
        print(s)
