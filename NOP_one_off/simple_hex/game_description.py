from dataclasses import dataclass
import colorama
from colorama import init
import numpy as np

init(convert=True)

hexDims = 8

@dataclass
class hexGame():
    global hexDims #Check if this is legal
    turnPos = hexDims ** 2

    """Creates a new game of Hex"""
    def __init__(self):
        global hexDims
        self.won = -2
        self.gameSize = hexDims
        self.board = np.array([0 for i in range(self.gameSize**2 + 1)])
        self.board[self.turnPos] = 1
        self.hexNeighbors = [[-1, 1], [0, 1], [1, 0], [1, -1], [-1, 0], [0, -1]]
        self.winArray = [[[(-1, i) for i in range(hexDims)], [(i, -1) for i in range(hexDims)]], [ [0]*hexDims for i in range(hexDims)]]

    def checkLegal(self, x, y):
      global hexDims
      if x >= 0 and y >= 0 and x < hexDims and y < hexDims:
        return self.board[self.hexToLine(x, y)] == 0
      else:
        return False

    def takeTurn(self, x, y): 
      if self.board[self.hexToLine(x, y)] == 0:
        self.board[self.hexToLine(x, y)] = self.board[self.turnPos]
        self.board[self.turnPos] *= -1
        return True
      else:
        return False

    def cNtC(self, num):
      if num == 0:
        return '.'
      elif num == -1:
        return colorama.Fore.RED+'R'+colorama.Fore.RESET
      elif num == 1:
        return colorama.Fore.BLUE+'B'+colorama.Fore.RESET
      else:
        return str(num)
    
    def displayInfo(self, refArray):
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
          s += self.cNtC(refArray[self.hexToLine(i, j)]) + ' '
        # right blue bar and end of row
        s += colorama.Fore.BLUE + '\\' + colorama.Fore.RESET + '\n'
      # bottom red bar
      s += ' '*i + ' '
      s += colorama.Fore.RED + '-'*(hexDims*2+1) + colorama.Fore.RESET
      print(s)

    def displayGame(self):
      self.displayInfo(self.board)
    
    def getPlayableArea(self):
      global hexDims
      return hexDims**2

    def takeLinTurn(self, x):
      if self.board[x] == 0:
        self.board[x] = self.board[self.turnPos]
        self.board[self.turnPos] *= -1
        return True
      else:
        print("BAD!!!")
        return False

    def getTurn(self):
      return self.board[self.turnPos]

    def hexToLine(self, x, y):
      return self.gameSize * x + y

    def checkGameWin(self):
      if self.won == -2:
        for state in [-1, 1]:
          for loc in self.winArray[0][int((1+state) / 2)]:
            for k in self.hexNeighbors:
              if 0 <= loc[0] + k[0] < self.gameSize and 0 <= loc[1] + k[1] < self.gameSize:#add in something to stop checking when filled around
                if self.winArray[1][loc[0] + k[0]][loc[1] + k[1]] == 0 and self.board[self.hexToLine(loc[0] + k[0],loc[1] + k[1])] == state:
                  self.winArray[0][int((1+state) / 2)].append((loc[0] + k[0], loc[1] + k[1]))
                  self.winArray[1][loc[0] + k[0]][loc[1] + k[1]] = state
        for i in range(self.gameSize):
          if self.winArray[1][i][self.gameSize-1] == 1:
            self.won = 1
            return 1
          if self.winArray[1][self.gameSize - 1][i] == -1:
            self.won = -1
            return -1
        return -2
      else:
        return self.won