from tkinter import *
from tkinter import messagebox
import itertools
#import tensorflow as tf
#import keras from tensorflow

hexDims = 11

class HexaCanvas(Canvas):
    """A canvas that provides a create-hexagone method"""
    def __init__(self, master, scale, grid_width, grid_height, *args, **kwargs):
        self.hexaSize = scale
        self.canDraw = True
        Δx     = (scale**2 - (scale/2.0)**2)**0.5
        width  = 2 * Δx * grid_width + Δx
        height = 1.5 * scale * grid_height + 0.5 * scale
        Canvas.__init__(self, master, *args, **kwargs, background='white', width = 1.5*width, height=height)

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
    
    def setCell(self, xCell, yCell, *args, **kwargs ):
        global hexDims
        """ Create a content in the cell of coordinates x and y. Could specify options throught keywords : color, fill, color1, color2, color3, color4; color5, color6"""
        size = self.hexaSize
        Δx = (size**2 - (size/2)**2)**0.5
        pix_x = Δx + 2*Δx*xCell + yCell * Δx
        pix_y = size + yCell*1.5*size
        set = [0,0]
        if xCell==0:
            set[0] = -1
        elif xCell==hexDims - 1:
            set[0] = 1
        
        if yCell==0:
            set[1] = -1
        elif yCell==hexDims - 1:
            set[1] = 1
        self.create_hexagone(set, pix_x, pix_y, *args, **kwargs)

    def convertToGrid(self, pix_x, pix_y):

        size = self.hexaSize
        Δx = (size**2 - (size/2)**2)**0.5
        yCell = (pix_y - size) / (1.5 * size)
        xCell = (pix_x - Δx - yCell*Δx) / (2*Δx)

        yCell = round(yCell)
        xCell = round(xCell)
        return xCell, yCell
    
    def drawGame(self, boardFunc):
        global hexDims
        #expensive to draw to board every frame? Yes, however, 
        #all games are cheap, so just make this standard.
        for i in range(hexDims):
            for j in range(hexDims):
                self.setCell(i, j, fill=boardFunc(i, j))



def statetoColor(num):
    if num == 0:
        return 'white'
    elif num == 1:
        return 'red'
    else:
        return 'blue'

if __name__ == "__main__":
    tk = Tk()

    game = hexGame()
    grid = HexaCanvas(tk, scale = 50, grid_width=hexDims, grid_height=hexDims)
    grid.grid(row=0, column=0, padx=5, pady=5)

    def displayGame(i, j):
        return 0#return the color here


    def correct_quit(tk):
        tk.destroy()
        tk.quit()

    quit = Button(tk, text = "Quit", command = lambda :correct_quit(tk))
    quit.grid(row=1, column=0)

    for i in range(hexDims):
        for j in range(hexDims):
            grid.setCell(i, j, fill='white')


    def resetBoard():
        global game
        game = hexGame()
        grid.drawGame(displayGame)


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

    def getClick(event):
        xCell, yCell = grid.convertToGrid(event.x, event.y)
        if game.checkLegal(xCell, yCell):
            game.takeTurn(xCell, yCell)
        
        grid.drawGame(displayGame)

        won = game.checkGameWin()
        if won == -1:
            messagebox.showinfo("","Blue win")
        elif won == 1:
            messagebox.showinfo("","Red won")

    
    def motion(event):
        xCell, yCell = grid.convertToGrid(x, y)
        grid.drawGame()
        if game.checkLegal(xCell, yCell):
            grid.setCell(xCell, yCell, fill='gray')

    tk.bind('<Motion>', motion)
    tk.bind('<Button-1>', getClick)

    tk.mainloop()


    #In this case, it is probably faster to have the neural network generate a number of moves, then check for wins.
    #instead, i will implement it slow to make certain that we can train something