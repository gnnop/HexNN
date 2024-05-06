import jax.numpy as jnp
from abc import ABC, abstractmethod
import chex

GameState = chex.Array
GameAction = chex.Array
GameReward = chex.Array
GameDone = jnp.bool_
GameCondition = jnp.bool_

class Game(ABC):

    @abstractmethod
    def initial_state(self) -> GameState:
        '''
        Creates a new game. That might look like an empty TicTacToe or Go board.
        '''
        pass

    @abstractmethod
    def take_turn(self, game_state: GameState, game_action: GameAction) -> tuple[GameState, GameReward]:
        '''
        Alters the game_state to reflect the changes done by game_action and evaluates a reward.
        An example of the GameReward might be scoring points in a video game or winning Tic Tac Toe.
        '''
        pass

    @abstractmethod
    def end_condition_met(self, game_state: GameState) -> GameDone:
        '''
        True if the game is terminated for some reason, 
        like if there's a checkmate in chess
        '''
        pass

    @abstractmethod
    def is_valid_action(self, game_state: GameState, game_action: GameAction) -> GameCondition:
        '''
        True if the action can be taken
        '''
        pass


class ZeroSumTwoPlayerTiledGame(Game):
    '''
    A two-player game where each player puts a piece on a tile until some win condition is satisfied.
    The first element in the GameAction must be the player ID, 0 for player 1 and 1 for player 2
    The second and third elements in the GameAction are the row and column of the piece to place
    '''
    @abstractmethod
    def get_board_shape(self) -> tuple[int, int]:
        '''
        The dimensions of the board
        '''
        pass
    @abstractmethod
    def get_valid_action_mask(self, game_state: GameState) -> chex.Array:
        '''
        A mask of the remaining valid positions on the board
        Flattened to 1D
        '''
        pass
