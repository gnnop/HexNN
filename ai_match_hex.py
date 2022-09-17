from cProfile import label
from doctest import master
from math import gamma
from typing import Iterator, Mapping, Tuple

from absl import app
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import optax
import random
import pickle
import copy
from multiprocessing.dummy import Pool as ThreadPool
import colorama
import ai_one
import ai_two

#First I'm going to start with an 8 by 8 board:

hexDims = 8

class hexGame():
    global hexDims #Check if this is legal
    turnPos = hexDims ** 2

    """Creates a new game of Hex"""
    def __init__(self, hexgame=None):
        global hexDims

        if hexgame == None:
          #Note that this is currently optimized for the 8 by 8 board. Very specific
          self.won = 0
          self.gameSize = hexDims
          self.hexes = np.array([0 for i in range(self.gameSize**2 + 1)])
          self.hexes[self.turnPos] = 1
          self.hexNeighbors = [[-1, 1], [0, 1], [1, 0], [1, -1], [-1, 0], [0, -1]]
          self.winArray = [[[(-1, i) for i in range(hexDims)], [(i, -1) for i in range(hexDims)]], [ [0]*hexDims for i in range(hexDims)]]
          #Red starts. Currently, we have no PI rule, I'm going to introduce that later
        else:
          self.won = hexgame.won
          self.gameSize = hexgame.gameSize
          self.hexes = copy.deepcopy(hexgame.hexes)
          self.hexNeighbors = hexgame.hexNeighbors
          self.winArray = copy.deepcopy(hexgame.winArray)

    def takeTurn(self, x, y): 
      if self.hexes[self.hexToLine(x, y)] == 0:
        self.hexes[self.hexToLine(x, y)] = self.hexes[self.turnPos]
        self.hexes[self.turnPos] *= -1
        return True
      else:
        return False

    def displayGame(self):
      # top red bar
      s = "" # the string to print
      s += colorama.Fore.RED + '-'*(hexDims*2+1) + colorama.Fore.RESET + '\n'
      for i in range(hexDims):
        # spacing to line up rows hexagonally
        s += ' '*i
        # left blue bar
        s += colorama.Fore.BLUE + '\\' + colorama.Fore.RESET
        # print a row of the game state
        for j in range(hexDims):
          character = '.'
          if self.hexes[self.hexToLine(i, j)]==1:
            character = colorama.Fore.BLUE+'B'+colorama.Fore.RESET
          elif self.hexes[self.hexToLine(i, j)]==-1:
            character = colorama.Fore.RED+'R'+colorama.Fore.RESET
          s += character + ' '
        # right blue bar and end of row
        s += colorama.Fore.BLUE + '\\' + colorama.Fore.RESET + '\n'
      # bottom red bar
      s += ' '*i + ' '
      s += colorama.Fore.RED + '-'*(hexDims*2+1) + colorama.Fore.RESET
      print(s)

    def takeLinTurn(self, x):
      if self.hexes[x] == 0:
        self.hexes[x] = self.hexes[self.turnPos]
        self.hexes[self.turnPos] *= -1
        return True
      else:
        print("BAD!!!")
        return False

    def getHexTurn(self):
      return self.hexes[self.turnPos]

    def hexToLine(self, x, y):
      return self.gameSize * x + y

    def checkGameWin(self):
      if self.won == 0:
        for state in [-1, 1]:
          for loc in self.winArray[0][int((1+state) / 2)]:
            for k in self.hexNeighbors:
              if 0 <= loc[0] + k[0] < self.gameSize and 0 <= loc[1] + k[1] < self.gameSize:#add in something to stop checking when filled around
                if self.winArray[1][loc[0] + k[0]][loc[1] + k[1]] == 0 and self.hexes[self.hexToLine(loc[0] + k[0],loc[1] + k[1])] == state:
                  self.winArray[0][int((1+state) / 2)].append((loc[0] + k[0], loc[1] + k[1]))
                  self.winArray[1][loc[0] + k[0]][loc[1] + k[1]] = state
        for i in range(self.gameSize):
          if self.winArray[1][i][self.gameSize-1] == 1:
            self.won = 1
            return 1
          if self.winArray[1][self.gameSize - 1][i] == -1:
            self.won = -1
            return -1
        return 0
      else:
        return self.won


def compareAI(aiOne, aiTwo):
    global hexDims
    aiOneScore = 0
    aiTwoScore = 0
    print("evaluating AIs")
    for pp in range(100):
      hexgame = hexGame()
      if pp % 2 == 0:
        firstPlayer = aiOne
        negOnePlayer = aiTwo
      else:
        firstPlayer = aiTwo
        negOnePlayer = aiOne
      
      #Do the first turn so results aren't even
      hexgame.takeLinTurn(random.randrange(0, hexDims ** 2))
      
      
      while hexgame.checkGameWin() == 0:
        if pp == 0:
          hexgame.displayGame()
        boards = []
        gamestates = []
        for i in range(hexDims**2):
          if hexgame.hexes[i] == 0:
            gamestates.append(i)
            hexgame.hexes[i] = hexgame.getHexTurn()
            boards.append(copy.deepcopy(hexgame.hexes))
            hexgame.hexes[i] = 0
          
        if hexgame.getHexTurn() == 1:
          pos = firstPlayer()
        else:
            pos = negOnePlayer()
        
        hexgame.takeLinTurn(pos)
      

      if pp % 2 == 0:
        if hexgame.checkGameWin() == 1:
          aiOneScore += 1
        else:
          aiTwoScore += 1
      else:
        if hexgame.checkGameWin() == 1:
          aiTwoScore += 1
        else:
          aiOneScore += 1

    return (aiOneScore, aiTwoScore)


print("(First player wins, second player wins)", compareAI(ai_one.play_turn, ai_two.play_turn))
