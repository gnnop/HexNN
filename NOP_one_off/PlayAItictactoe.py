#importing Packages from tkinter
from tkinter import *
from tkinter import messagebox
from dataclasses import dataclass
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import pickle
import copy

Player1 = 'X'
stop_game = False

#add an ai button to force the ai to make a move


def net_fn(batch):
  x = batch.astype(jnp.float32)
  h1 = hk.Sequential([
    hk.Flatten(),
    hk.Linear(100), jax.nn.relu,
    hk.Linear(300), jax.nn.relu])
  h2 = hk.Sequential([
    hk.Linear(300), jax.nn.relu,
    hk.Linear(300), jax.nn.relu,
    hk.Linear(300), jax.nn.relu])
  h3 = hk.Sequential([
    hk.Linear(300), jax.nn.relu,
    hk.Linear(100), jax.nn.relu,
    hk.Linear(20),  hk.Linear(1)])
  y1 = h1(x)
  y2 = y1 + h2(y1)
  y3 = h3(y2)

  return y3

net = hk.without_apply_rng(hk.transform(net_fn))

@dataclass
class tictactoe:
  # board = np.array([0,0,0, 0,0,0, 0,0,0, 1])
  def __init__(self):
    self.board = np.array([0,0,0, 0,0,0, 0,0,0, 1])

  def takeTurn(self, i):
    if self.board[i] == 0:
      self.board[i] = self.board[9]
      self.board[9] *= -1
      return True
    else:
      return False

  def clicked(self, i , j):
    self.takeTurn(3*i + j)
	#To maintain state, simply repopulate the thing
  
  def convertNumToChar(self, num):
    if num == 0:
      return " "
    elif num == -1:
      return "X"
    elif num == 1:
      return "O"
    else:
      return str(num)

  def displayInfo(self, stuff):
    s = "\n" + self.convertNumToChar(stuff[0]) + "|" + self.convertNumToChar(stuff[3]) + "|" + self.convertNumToChar(stuff[6])
    s+= "\n------\n"
    s+= self.convertNumToChar(stuff[1]) + "|" + self.convertNumToChar(stuff[4]) + "|" + self.convertNumToChar(stuff[7])
    s+= "\n------\n"
    s+= self.convertNumToChar(stuff[2]) + "|" + self.convertNumToChar(stuff[5]) + "|" + self.convertNumToChar(stuff[8]) + "\n"
    print(s)

  def displayGame(self):
    s = "\n" + self.convertNumToChar(self.board[0]) + "|" + self.convertNumToChar(self.board[3]) + "|" + self.convertNumToChar(self.board[6])
    s+= "\n------\n"
    s+= self.convertNumToChar(self.board[1]) + "|" + self.convertNumToChar(self.board[4]) + "|" + self.convertNumToChar(self.board[7])
    s+= "\n------\n"
    s+= self.convertNumToChar(self.board[2]) + "|" + self.convertNumToChar(self.board[5]) + "|" + self.convertNumToChar(self.board[8]) + "\n"
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


# Design window
#Creating the Canvas
root = Tk()
# Title of the window			
root.title("Tic Tac Toe")
root.resizable(0,0)

#Button
b = [
	[0,0,0],
	[0,0,0],
	[0,0,0]]

game = tictactoe()

try:
	params = pickle.load(open('partial_training.params', 'rb'))
	print('loaded AI')
except:
	print('no AI, abort!')



def display(game):
	global b
	for i in range(3):
		for j in range(3):
			n = game.board[3*i + j]
			if n == 1:
				b[i][j].configure(text='O')
			elif n == -1:
				b[i][j].configure(text='X')
			else:
				b[i][j].configure(text='')
	
	if game.checkGameWin() == 0:
		messagebox.showinfo("","Tie")
	if game.checkGameWin() == -1:
		messagebox.showinfo("","X win")
	if game.checkGameWin() == 1:
		messagebox.showinfo("","O won")

def resetBoard():
	global game
	game = tictactoe()
	display(game)

def makeAImove():
	global params
	global game

	boards = []
	gamestates = []
	for i in range(9):
		if game.board[i] == 0:
			gamestates.append(i)
			game.board[i] = game.getTurn()
			boards.append(copy.deepcopy(game.board))
			game.board[i] = 0
		
	preds = net.apply(params, jnp.array(boards))
	if game.getTurn() == 1:
		val = jnp.max(preds)
	else:
		val = jnp.min(preds)
	
	writeVals = ['NA']*9
	for i in range(len(gamestates)):
		writeVals[gamestates[i]] = preds[i][0]

	game.displayInfo(writeVals)
	game.displayGame()


	game.takeTurn(gamestates[jnp.where(preds == val)[0][0]])
	display(game)

def clicked(i, j):
	global game
	game.clicked(i,j)
	display(game)

Button(height=4, width=6, font = ("Helvetica","16"), command = resetBoard, text='reset').grid(column=0,row=0)
Button(height=4, width=6, font = ("Helvetica","16"), command = makeAImove, text='go AI').grid(column=2,row=0)

for i in range(3):
	for j in range(3):
		b[i][j] = Button(height=4, width=8, font = ("Helvetica","20"), command = lambda r = i, c = j : clicked(r,c))
		b[i][j].grid(column=2*i,row=2*(j+1))

mainloop()		
