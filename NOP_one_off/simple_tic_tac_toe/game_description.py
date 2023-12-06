from dataclasses import dataclass
import colorama
import numpy as np


@dataclass
class tictactoe:
  # board = np.array([0,0,0, 0,0,0, 0,0,0, 1])
  def __init__(self):
    self.board = np.array([0,0,0, 0,0,0, 0,0,0, 1])

  def clicked(self, i , j):
    self.takeTurn(3*i + j)

  def takeTurn(self, i):
    if self.board[i] == 0:
      self.board[i] = self.board[9]
      self.board[9] *= -1
      return True
    else:
      return False
  
  def cNtC(self, num):
    if num == 0:
      return " "
    elif num == -1:
      return "X"
    elif num == 1:
      return "O"
    else:
      return str(num)

  def displayInfo(self, stuff):
    s = "\n" + self.cNtC(stuff[0]) + "|" + self.cNtC(stuff[3]) + "|" + self.cNtC(stuff[6])
    s+= "\n------\n"
    s+= self.cNtC(stuff[1]) + "|" + self.cNtC(stuff[4]) + "|" + self.cNtC(stuff[7])
    s+= "\n------\n"
    s+= self.cNtC(stuff[2]) + "|" + self.cNtC(stuff[5]) + "|" + self.cNtC(stuff[8]) + "\n"
    print(s)

  def displayGame(self):
    s = "\n" + self.cNtC(self.board[0]) + "|" + self.cNtC(self.board[3]) + "|" + self.cNtC(self.board[6])
    s+= "\n------\n"
    s+= self.cNtC(self.board[1]) + "|" + self.cNtC(self.board[4]) + "|" + self.cNtC(self.board[7])
    s+= "\n------\n"
    s+= self.cNtC(self.board[2]) + "|" + self.cNtC(self.board[5]) + "|" + self.cNtC(self.board[8]) + "\n"
    print(s)
  
  def getTurn(self):
    return self.board[9]
  
  def checkGameWin(self):
    for i in [-1, 1]:
      if (i == self.board[0] == self.board[1] == self.board[2] or
         i == self.board[3] == self.board[4] == self.board[5] or
         i == self.board[6] == self.board[7] == self.board[8] or
         i == self.board[0] == self.board[3] == self.board[6] or
         i == self.board[1] == self.board[4] == self.board[7] or
         i == self.board[2] == self.board[5] == self.board[8] or
         i == self.board[0] == self.board[4] == self.board[8] or
         i == self.board[2] == self.board[4] == self.board[6]):
        return i
    for i in range(9):
      if self.board[i] == 0:
        return -2

    return 0