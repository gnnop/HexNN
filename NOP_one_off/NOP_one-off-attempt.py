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
          self.hexes = hexgame.hexes
          self.hexNeighbors = hexgame.hexNeighbors
          self.winArray = hexgame.winArray.copy()

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


Batch = Mapping[str, np.ndarray]



#This is the neural networkd, I want it to take in a representation
#with two parameters



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


def main(_):
  # Make the network and optimiser.
  net = hk.without_apply_rng(hk.transform(net_fn))
  opt = optax.adam(1e-3)


  #generating a tree to tranverse. First design is to assign a value to a given game state. What I do then is evaluate all the game states
  #this simply generates a value. I impose consistency requirements on everything. 

  #I believe what I'll do for now is to evaluate an entire game tree and look at all states along that tree.

  def compareAI(aiOne, aiTwo):
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
      
      while hexgame.checkGameWin() == 0:
        if pp == 0:
          hexgame.displayGame()
        boards = []
        gamestates = []
        for i in range(hexDims**2):
          if hexgame.hexes[i] == 0:
            gamestates.append(i)
            hexgame.hexes[i] = hexgame.getHexTurn()
            boards.append(hexgame.hexes.copy())
            hexgame.hexes[i] = 0
          
        if hexgame.getHexTurn() == 1:
          preds = net.apply(firstPlayer, jnp.array(boards))
          val = jnp.max(preds)
        else:
          preds = net.apply(negOnePlayer, jnp.array(boards))
          val = jnp.min(preds)
        
        hexgame.takeLinTurn(gamestates[jnp.where(preds == val)[0][0]])
      
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
          


  def generateGameBatch(hexgame,params):
    global hexDims

    boards = []
    labels = []

    ii = 0

    while hexgame.checkGameWin() == 0:
      alphaBetaBoards = []

      foundGameWin = False

      gameStates = []

      for i in range(hexDims**2):
        hexgame_ = copy.deepcopy(hexgame)

        if hexgame_.hexes[i] == 0 and not foundGameWin:
          gameStates.append(i)
          hexgame_.hexes[i] = hexgame_.getHexTurn()
          game_cond = hexgame_.checkGameWin()
          if hexgame_.getHexTurn() == game_cond == 1:
            foundGameWin = True
            labels.append(1)
            boards.append(copy.deepcopy(hexgame_.hexes))
          elif hexgame_.getHexTurn() == game_cond == -1:
            foundGameWin = True
            labels.append(-1)
            boards.append(copy.deepcopy(hexgame_.hexes))

          alphaBetaBoards.append(hexgame_.hexes.copy())
      #alpha beta value - the nn returns 

      #Now serialize the boards and apply everything:
      if not foundGameWin:
        ls = net.apply(params, np.array(alphaBetaBoards))
        if hexgame.getHexTurn() == 1: #simple alpha beta
          labels.append(jnp.max(ls))
        else:
          labels.append(jnp.min(ls))

        #Use some mix of exploration and the network
        if random.random() < 0.3:
          num = 0
          for i in hexgame.hexes:
            if i == 0:
              num+=1
          pos = random.randrange(0, num)
          num = 0
          absPos = 0
          for i in hexgame.hexes:
            if i == 0:
              if pos == num:
                hexgame.takeLinTurn(absPos)
              num+=1
            absPos+=1
        else:
          boards.append(copy.deepcopy(hexgame.hexes))
          hexgame.takeLinTurn(gameStates[np.where(ls == labels[-1])[0][0]])
      else:
        break
      
    return boards, labels

  # Training loss (cross-entropy).    pool = ThreadPool(40)
  def loss(params: hk.Params, batch_data, batch_labels) -> jnp.ndarray:
    """Compute the loss of the network, including L2."""
    logits = net.apply(params, jnp.array(batch_data))
    labels = jnp.array(batch_labels)

    l2_loss = 0.5 * sum(jnp.sum(jnp.square(p)) for p in jax.tree_leaves(params))
    softmax_xent = -jnp.sum(labels * jax.nn.log_softmax(logits))
    softmax_xent /= labels.shape[0]

    return softmax_xent + 1e-4 * l2_loss


  @jax.jit
  def update(
      params: hk.Params,
      opt_state: optax.OptState,
      batch_data, batch_labels
  ) -> Tuple[hk.Params, optax.OptState]:
    """Learning rule (stochastic gradient descent)."""
    val, grads = jax.value_and_grad(loss)(params, batch_data, batch_labels)
    updates, opt_state = opt.update(grads, opt_state)
    new_params = optax.apply_updates(params, updates)
    return val, new_params, opt_state

  # We maintain avg_params, the exponential moving average of the "live" params.
  # avg_params is used only for evaluation (cf. https://doi.org/10.1137/0330046)
  @jax.jit
  def ema_update(params, avg_params):
    return optax.incremental_update(params, avg_params, step_size=0.001)

  # Initialize network and optimiser; note we draw an input to get shapes.
  params = net.init(jax.random.PRNGKey(42), jnp.array([hexGame().hexes]))
  opt_state = opt.init(params)

  grabAI = params

  # Train/eval loop.
  for step in range(100001):
    if step % 100 == 0:
      print(step)
    if step % 300 == 0:
      # Periodically evaluate classification accuracy on train & test sets.
      oldScore, newScore = compareAI(grabAI, params)
      print("The old AI scored " + str(oldScore) + "and the new scored " + str(newScore))
      grabAI = copy.deepcopy(params)

    # Do SGD on a batch of training examples.

    pool = ThreadPool(20)
    master_list = pool.map(lambda a: generateGameBatch(hexGame(), params), range(20))

    if step < 2:
      print(master_list)

    flat_list_data = [item for sublist in master_list for item in sublist[0]]
    flat_list_label = [item for sublist in master_list for item in sublist[1]]
    
    val, params, opt_state = update(params, opt_state, flat_list_data, flat_list_label)
    print(val)

  file = open('trained-model.params', 'wb')
  pickle.dump(params, file)
  file.close()
  print("Bye")
  exit()

if __name__ == "__main__":
  app.run(main)