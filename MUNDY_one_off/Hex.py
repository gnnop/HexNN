from unicodedata import numeric
import numpy as np
import enum
import math

from tkinter import *
from tkinter import ttk




################################## Mundy-Game Mechanics ###################################
class HexGame:

  def __init__(self, size:numeric):
    self.size = size
    '''
    The game state is stored in two rectangular arrays.
    This makes checking for wins easy.
    '''
    self.game_state: np.array = np.ones([2, self.size, self.size])

  def place_blue_piece(self, x, y):
    '''
    Place a blue piece at a given location
    '''
    self.game_state[0][y][x] = 0

  def place_red_piece(self, x, y):
    '''
    Place a red piece at a given location
    '''
    self.game_state[1][x][y] = 0

  def check_free(self, x, y):
    '''
    Returns 1 if a space is free.
    Returns 0 if a space is occupied by a color
    '''
    return self.game_state[0][y][x] * self.game_state[1][x][y]

  def check_not_win(self, color):
    '''
    Returns 0 if the color 'color' won
    Returns a natural number otherwise
    '''
    tempState = self.game_state[color].copy()
    for i in range(1, self.size):
      for j in range(self.size-1):
        tempState[i][j]           *= (tempState[i-1][j] + tempState[i-1][j+1])
        tempState[i][self.size-1] *= tempState[i-1][self.size-1]
    return np.sum(tempState[self.size-1])


  def swap(self):
    '''
    Makes the red pieces blue and the blue pieces red.
    Also rotates the game board 180 degrees
    Used for the swap rule
    '''
    self.game_state[0], self.game_state[1] = self.game_state[1], self.game_state[0]

################################## Mundy-Game Mechanics ###################################



# A futile attempt at recreating the GUI. I like the draw hexagon function though
# class TKHexGame(Canvas):

#   def draw_hexagon(self, x, y, fill_color="gray", stroke_color="black"):
#     '''
#     Shape generated from the following:
#     shape = [[math.sin(i*math.pi/3), math.cos(i*math.pi/3)] for i in range(6)]
#     '''
#     xt = x + 1 + 0.5*y
#     yt = y + 1
#     shape = [
#          1+xt,      0+yt  ,  
#         .5+xt,   .866+yt  ,  
#        -.5+xt,   .866+yt  ,  
#         -1+xt,      0+yt  , 
#        -.5+xt,  -.866+yt  ,  
#         .5+xt,  -.866+yt  
#     ]
#     shape = [i*self.scale for i in shape]
    
#     # Draw the hexagon
#     self.create_polygon(shape, fill=fill_color)
#     # Draw the outline
#     for i in range(5):
#        self.create_line(shape[i*2:i*2+4], fill=stroke_color)
#     self.create_line(shape[10:12]+shape[0:2], fill=stroke_color)
#   # End draw_hexagon

#   def draw_hex_grid(self):
#     gs = self.game_state
#     for i in range(gs.size):
#       for j in range(gs.size):
#         color = 'gray'
#         if gs.game_state[0][i][j] == 0:
#           color = 'blue'
#         elif gs.game_state[1][j][i] == 0:
#           color = 'red'
#         self.draw_hexagon(i, j, fill_color=color)

#   def __init__(self, tk_instance, size):
#     Canvas.__init__(self, tk_instance)
#     self.game_state = HexGame(size)
#     #self.scale = min(tk.winfo_width()/(1.5*self.size), tk.winfo_height()/self.size)
#     self.scale = 32










#########################################    GUI   ###################################################
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
    def __init__(self, master, scale, grid_size, *args, **kwargs):
        self.grid_size = grid_size
        self.canDraw = True
        Δx     = (scale**2 - (scale/2.0)**2)**0.5
        width  = 2 * Δx * self.grid_size + Δx
        height = 1.5 * scale * self.grid_size + 0.5 * scale

        HexaCanvas.__init__(self, master, background='white', width=1.5*width, height=height, *args, **kwargs)
        self.setHexaSize(scale)

    def setCell(self, xCell, yCell, *args, **kwargs ):
        """ Create a content in the cell of coordinates x and y. Could specify options throught keywords : color, fill, color1, color2, color3, color4; color5, color6"""
        if self.canDraw:
            size = self.hexaSize
            Δx = (size**2 - (size/2)**2)**0.5
            pix_x = Δx + 2*Δx*xCell + yCell * Δx
            pix_y = size + yCell*1.5*size
            set = [0,0]
            if xCell==0:
                set[0] = -1
            elif xCell==self.grid_size - 1:
                set[0] = 1
            
            if yCell==0:
                set[1] = -1
            elif yCell==self.grid_size - 1:
                set[1] = 1
            self.create_hexagone(set, pix_x, pix_y, *args, **kwargs)

    def setDraw(self, draw):
        self.canDraw = draw

    def convertToGrid(self, pix_x, pix_y):

        size = self.hexaSize
        Δx = (size**2 - (size/2)**2)**0.5
        yCell = (pix_y - size) / (1.5 * size)
        xCell = (pix_x - Δx - yCell*Δx) / (2*Δx)

        yCell = round(yCell)
        xCell = round(xCell)
        return xCell, yCell













class TKHex:


  def __init__(self, size):
    # The guts
    self.game_mechanics = HexGame(size)
    self.turn = 0
    # The skin
    self.tk = Tk()
    self.grid = HexagonalGrid(self.tk, scale = 50, grid_size=size)
    self.grid.grid(row=0, column=0, padx=5, pady=5)

    
    def correct_quit():
      self.tk.destroy()
      self.tk.quit()

    self.quit = Button(self.tk, text = "Quit", command = lambda: correct_quit())
    self.quit.grid(row=1, column=0)
    
    for i in range(size):
        for j in range(size):
            self.grid.setCell(i, j, fill='gray')

    def getClick(event):
      xCell, yCell = self.grid.convertToGrid(event.x, event.y)
      if xCell in range(self.game_mechanics.size) and yCell in range(self.game_mechanics.size):
        if self.game_mechanics.check_free(xCell, yCell):
          if self.turn == 0:
            self.turn = 1
            self.game_mechanics.place_blue_piece(xCell, yCell)
            self.grid.setCell(xCell, yCell, fill="blue")
            if self.game_mechanics.check_not_win(0) == 0:
              print("Blue won")
              self.grid.setDraw(False)
          else:
            self.turn = 0
            self.game_mechanics.place_red_piece(xCell, yCell)
            self.grid.setCell(xCell, yCell, fill="red")
            if self.game_mechanics.check_not_win(1) == 0:
              print("Red won")
              self.grid.setDraw(False)
        else:
          print("Invalid move")
            

    self.tk.bind('<Button-1>', getClick)
  # End constructor

  def spin(self):
    self.tk.mainloop()




if __name__ == "__main__":
  hex = TKHex(11)
  hex.spin()




