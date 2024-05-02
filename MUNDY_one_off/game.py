import jax.numpy as jnp
from abc import ABC, abstractmethod

class GameState:
    def __init__(self, data):
        self._data = jnp.array(data)
    def __getattr__(self, name):
        return getattr(self._data, name)
    def __getitem__(self, index):
        return self._data[index]

class GameAction:
    def __init__(self, data):
        self._data = jnp.array(data)
    def __getattr__(self, name):
        return getattr(self._data, name)
    def __getitem__(self, index):
        return self._data[index]

class GameReward:
    def __init__(self, data):
        self._data = jnp.array(data)
    def __getattr__(self, name):
        return getattr(self._data, name)
    def __getitem__(self, index):
        return self._data[index]

class Game(ABC):

    @abstractmethod
    def take_turn(self, game_state: GameState, game_action: GameAction) -> tuple[GameState, GameReward]:
        '''
        Alters the game_state to reflect the changes done by game_action and evaluates a reward.
        An example of the GameReward might be scoring points in a video game or winning Tic Tac Toe.
        '''
        pass

    @abstractmethod
    def end_condition_met(self, game_state: GameState) -> bool:
        '''
        True if the game is terminated for some reason, 
        like if there's a checkmate in chess
        '''
        pass

    @abstractmethod
    def is_valid_action(self, game_state: GameState, game_action: GameAction) -> bool:
        '''
        True if the action can be taken
        '''
        pass