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
from game_description import *
from AI import *

Player1 = 'X'
stop_game = False

#add an ai button to force the ai to make a move
net = hk.without_apply_rng(hk.transform(net_fn))
game = tictactoe()

root = Tk()			
root.title("Tic Tac Toe")
root.resizable(0,0)

b = [
	[0,0,0],
	[0,0,0],
	[0,0,0]]

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