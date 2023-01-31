from cProfile import label
from doctest import master
from math import gamma
from typing import Iterator, Mapping, Tuple
from os.path import exists
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
import nop_ai

from tkinter import *
import itertools
#import tensorflow as tf
#import keras from tensorflow

boardSize = 11

  def compareAI(aiOne, aiTwo):
    global hexDims
    aiOneScore = 0
    aiTwoScore = 0
    print("evaluating AIs")
    for pp in range(10):
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
          preds = net.apply(firstPlayer, jnp.array(boards))
          val = jnp.max(preds)
        else:
          preds = net.apply(negOnePlayer, jnp.array(boards))
          val = jnp.min(preds)
        
        hexgame.takeLinTurn(gamestates[jnp.where(preds == val)[0][0]])
      
      if pp == 0:
        print("This player won, blue went first: ", hexgame.checkGameWin())
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
          

def net_fn(batch: Batch) -> jnp.ndarray:
  """Standard LeNet-300-100 MLP network."""
  x = batch.astype(jnp.float32)

  #sequential unit
  #this code is the initial processing
  mlp = hk.Sequential([
      hk.Flatten(),
      hk.Linear(100), jax.nn.relu,
      hk.Linear(300), jax.nn.relu,
      hk.Linear(300), jax.nn.relu,
      hk.Linear(300), jax.nn.relu,
      hk.Linear(300), jax.nn.relu,
      hk.Linear(100), jax.nn.relu,
      hk.Linear(20), jax.nn.relu,
      hk.Linear(1)
  ])
  #convolutional network
  #this code convolces everything a couple of times
  """
  conv = hk.Sequential([
    hk.Flatten(),
    hk.Linear()
  ])"""

  #combination network. concatenate previous results
  #and combine to see what happens.
  #may switch to attention transformater


  return mlp(x)

class HexaCanvas(Canvas):
    """A canvas that provides a create-hexagone method"""
    def __init__(self, master, *args, **kwargs):
        Canvas.__init__(self, master, *args, **kwargs)
        self.hexaSize = 20

    def setHexaSize(self, number):
        self.hexaSize = number


    def create_hexagone(self, set, x, y, color = "black", fill="blue", color1=None, color2=None, color3=None, color4=None, color5=None, color6=None):
        size = self.hexaSize
        Δx = (size**2 - (size/2)**2)**0.5

        point1 = (x+Δx, y+size/2)
        point2 = (x+Δx, y-size/2)
        point3 = (x   , y-size  )
        point4 = (x-Δx, y-size/2)
        point5 = (x-Δx, y+size/2)
        point6 = (x   , y+size  )

        #this setting allow to specify a different color for each side.
        if color1 == None:
            color1 = color
        if color2 == None:
            color2 = color
        if color3 == None:
            color3 = color
        if color4 == None:
            color4 = color
        if color5 == None:
            color5 = color
        if color6 == None:
            color6 = color
        
        w = [0, 1, 1, 1, 1, 1, 1]
        if set[1]==1:
            w[5] = 2
            color5 = "red"
            w[6] = 2
            color6 = "red"
        
        if set[1]==-1:
            w[2] = 2
            color2 = "red"
            w[3] = 2
            color3 = "red"

        if set[0]==-1:
            w[4] = 2
            color4 = "blue"
            w[5] = 2
            color5 = "blue"

        if set[0]==1:
            w[2] = 2
            color2 = "blue"
            w[1] = 2
            color1 = "blue"

        if set[0]==1 and set[1]==-1:
            color2 = "purple"

        if set[1]==1 and set[0]==-1:
            color5 = "purple"

        if fill != None:
            self.create_polygon(point1, point2, point3, point4, point5, point6, fill=fill)

        self.create_line(point1, point2, fill=color1, width=6*w[1])
        self.create_line(point2, point3, fill=color2, width=6*w[2])
        self.create_line(point3, point4, fill=color3, width=6*w[3])
        self.create_line(point4, point5, fill=color4, width=6*w[4])
        self.create_line(point5, point6, fill=color5, width=6*w[5])
        self.create_line(point6, point1, fill=color6, width=6*w[6])



class HexagonalGrid(HexaCanvas):
    """ A grid whose each cell is hexagonal """
    def __init__(self, master, scale, grid_width, grid_height, *args, **kwargs):

        self.canDraw = True
        Δx     = (scale**2 - (scale/2.0)**2)**0.5
        width  = 2 * Δx * grid_width + Δx
        height = 1.5 * scale * grid_height + 0.5 * scale

        HexaCanvas.__init__(self, master, background='white', width=1.5*width, height=height, *args, **kwargs)
        self.setHexaSize(scale)

    def setCell(self, xCell, yCell, *args, **kwargs ):
        global boardSize
        """ Create a content in the cell of coordinates x and y. Could specify options throught keywords : color, fill, color1, color2, color3, color4; color5, color6"""
        if self.canDraw:
            size = self.hexaSize
            Δx = (size**2 - (size/2)**2)**0.5
            pix_x = Δx + 2*Δx*xCell + yCell * Δx
            pix_y = size + yCell*1.5*size
            set = [0,0]
            if xCell==0:
                set[0] = -1
            elif xCell==boardSize - 1:
                set[0] = 1
            
            if yCell==0:
                set[1] = -1
            elif yCell==boardSize - 1:
                set[1] = 1
            self.create_hexagone(set, pix_x, pix_y, *args, **kwargs)

    def setDraw(self, draw):
        self.canDraw = draw

    def getLinBoard():
        self.

    def convertToGrid(self, pix_x, pix_y):

        size = self.hexaSize
        Δx = (size**2 - (size/2)**2)**0.5
        yCell = (pix_y - size) / (1.5 * size)
        xCell = (pix_x - Δx - yCell*Δx) / (2*Δx)

        yCell = round(yCell)
        xCell = round(xCell)
        return xCell, yCell


prevxCell = 0
prevyCell = 0
gameState = [[ [0]*boardSize for i in range(boardSize)], 1] #This gives the array and the turn #, 0 for and 1 for
winState = [[[(-1, i) for i in range(boardSize)], [(i, -1) for i in range(boardSize)]], [ [0]*boardSize for i in range(boardSize)]]
hexNeighbors = [[-1, 1], [0, 1], [1, 0], [1, -1], [-1, 0], [0, -1]]
won = 0

#Next step: AI
def checkGameWin(game, winArray):
    global boardSize
    global hexNeighbors
    for state in [-1, 1]:
        for loc in winArray[0][int((1+state) / 2)]:
            for k in hexNeighbors:
                if 0 <= loc[0] + k[0] < boardSize and 0 <= loc[1] + k[1] < boardSize:#add in something to stop checking when filled around
                    if winArray[1][loc[0] + k[0]][loc[1] + k[1]] == 0 and game[loc[0] + k[0]][loc[1] + k[1]] == state:
                        winArray[0][int((1+state) / 2)].append((loc[0] + k[0], loc[1] + k[1]))
                        winArray[1][loc[0] + k[0]][loc[1] + k[1]] = state
    for i in range(boardSize):
        if winArray[1][i][boardSize-1] == 1:
            return (1, winArray)
        if winArray[1][boardSize - 1][i] == -1:
            return (-1, winArray)
    return (0, winArray)



def statetoColor(num):
    if num == 0:
        return 'white'
    elif num == 1:
        return 'red'
    else:
        return 'blue'

def checkLegal(xCell, yCell):
    global gameState
    return gameState[0][xCell][yCell] == 0

if __name__ == "__main__":
    tk = Tk()

    grid = HexagonalGrid(tk, scale = 50, grid_width=boardSize, grid_height=boardSize)
    grid.grid(row=0, column=0, padx=5, pady=5)

    def correct_quit(tk):
        tk.destroy()
        tk.quit()

    quit = Button(tk, text = "Quit", command = lambda :correct_quit(tk))
    quit.grid(row=1, column=0)

    for i in range(boardSize):
        for j in range(boardSize):
            grid.setCell(i, j, fill='white')

    def getClick(event):
        global gameState
        global winState
        global won
        xCell, yCell = grid.convertToGrid(event.x, event.y)
        if won==0 and (xCell in range(boardSize) and yCell in range(boardSize)):
            if(checkLegal(xCell, yCell)):
                gameState[1] = -1 * gameState[1]
                gameState[0][xCell][yCell] = gameState[1]
                grid.setCell(xCell, yCell, fill=statetoColor(gameState[1]))
                won, winState = checkGameWin(gameState[0], winState)
                if won != 0:
                    for loc in itertools.product(range(boardSize), range(boardSize)):
                        if winState[1][loc[0]][loc[1]] == won:
                            grid.setCell(loc[0], loc[1], fill='yellow')
                    grid.setDraw(False)



        
    
    def motion(event):
        global prevxCell
        global prevyCell
        global gameState
        grid.setCell(prevxCell, prevyCell, fill=statetoColor(gameState[0][prevxCell][prevyCell]))
        x, y = event.x, event.y
        xCell, yCell = grid.convertToGrid(x, y)
        if won == 0:
            if(xCell in range(boardSize) and yCell in range(boardSize) and checkLegal(xCell, yCell)):    
                grid.setCell(xCell, yCell, fill='gray')
                prevxCell = xCell
                prevyCell = yCell
            
            #Now do the AI stiff
            grid.setLinCell(nop_ai(grid.getLinBoard()))
    tk.bind('<Motion>', motion)
    tk.bind('<Button-1>', getClick)



    net = hk.without_apply_rng(hk.transform(net_fn))





    tk.mainloop()