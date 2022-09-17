from cProfile import label
from doctest import master
from math import gamma
from operator import truediv
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
from dataclasses import dataclass
import colorama

#First I'm going to start with an 8 by 8 board:


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
  
  def convertNumToChar(self, num):
    if num == 0:
      return " "
    elif num == -1:
      return "X"
    elif num == 1:
      return "O"

  def displayGame(self):
    s = "\n" + self.convertNumToChar(self.board[0]) + "|" + self.convertNumToChar(self.board[1]) + "|" + self.convertNumToChar(self.board[2])
    s+= "\n------\n"
    s+= self.convertNumToChar(self.board[3]) + "|" + self.convertNumToChar(self.board[4]) + "|" + self.convertNumToChar(self.board[5])
    s+= "\n------\n"
    s+= self.convertNumToChar(self.board[6]) + "|" + self.convertNumToChar(self.board[7]) + "|" + self.convertNumToChar(self.board[8]) + "\n"
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
      hk.Linear(100), jax.nn.relu,
      hk.Linear(20),
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
      hexgame = tictactoe()
      if pp % 2 == 0:
        firstPlayer = aiOne
        negOnePlayer = aiTwo
      else:
        firstPlayer = aiTwo
        negOnePlayer = aiOne
      
      #Do the first turn so results aren't even
      hexgame.takeTurn(random.randrange(0, 9))
      

      while hexgame.checkGameWin() == -2:
        if pp == 0:
          print("displaying game")
          hexgame.displayGame()
        boards = []
        gamestates = []
        for i in range(9):
          if hexgame.board[i] == 0:
            gamestates.append(i)
            hexgame.board[i] = hexgame.getTurn()
            boards.append(copy.deepcopy(hexgame.board))
            hexgame.board[i] = 0
          
        if hexgame.getTurn() == 1:
          preds = net.apply(firstPlayer, jnp.array(boards))
          val = jnp.max(preds)
        else:
          preds = net.apply(negOnePlayer, jnp.array(boards))
          val = jnp.min(preds)
        
        hexgame.takeTurn(gamestates[jnp.where(preds == val)[0][0]])
      
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
          


  def generateGameBatch(hexgame,params):

    boards = []
    labels = []

    turns = 0
    while hexgame.checkGameWin() == -2:
      turns += 1
      alphaBetaBoards = []
      gameStates = []

      for i in range(9):
        if hexgame.board[i] == 0:
          hexgame.board[i] = hexgame.getTurn()#in place modification without changing state
          gameStates.append(i)
          alphaBetaBoards.append(copy.deepcopy(hexgame.board))
          hexgame.board[i] = 0
      #alpha beta value - the nn returns 

      #Now serialize the boards and apply everything:
      ls = net.apply(params, np.array(alphaBetaBoards))
      if hexgame.getTurn() == 1: #simple alpha beta
        boards.append(copy.deepcopy(hexgame.board))
        labels.append(jnp.max(ls))
      else:
        boards.append(copy.deepcopy(hexgame.board))
        labels.append(jnp.min(ls))
      
      #make the actual move with some probability:

      #Use some mix of exploration and the network
      if random.random() < 0.1 and turns < 20:
        num = 0
        for i in hexgame.board:
          if i == 0:
            num+=1
        pos = random.randrange(0, num)
        num = 0
        absPos = 0
        for i in hexgame.board:
          if i == 0:
            if pos == num:
              hexgame.takeTurn(absPos)
            num+=1
          absPos+=1
      else:
        hexgame.takeTurn(gameStates[np.where(ls == labels[-1])[0][0]])
    
    boards.append(copy.deepcopy(hexgame.board))
    labels.append(hexgame.checkGameWin())

    return boards, labels

  # Training loss (cross-entropy).    pool = ThreadPool(40)
  def loss(params: hk.Params, batch_data, batch_labels) -> jnp.ndarray:
    """Compute the loss of the network, including L2."""
    logits = net.apply(params, jnp.array(batch_data))
    labels = jnp.array(batch_labels)

    loss = jnp.sum(jnp.square(logits - labels)) / labels.shape[0]
    #l2_loss = 0.5 * sum(jnp.sum(jnp.square(p)) for p in jax.tree_leaves(params))
    #softmax_xent = -jnp.sum(labels * jnp.log(logits))
    #softmax_xent /= labels.shape[0]

    return loss


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

  # Initialize network and optimiser; note we draw an input to get shapes.
  params = net.init(jax.random.PRNGKey(42), jnp.array([tictactoe().board]))
  opt_state = opt.init(params)

  grabAI = params

  # Train/eval loop.
  try:
    for step in range(10001):
      if step % 100 == 0:
        print(step)
      if step % 30 == 0:
        # Periodically evaluate classification accuracy on train & test sets.
        oldScore, newScore = compareAI(grabAI, params)
        print("The old AI scored " + str(oldScore) + "and the new scored " + str(newScore))
        grabAI = copy.deepcopy(params)

      # Do SGD on a batch of training examples.

      pool = ThreadPool(20)
      master_list = pool.map(lambda a: generateGameBatch(tictactoe(), params), range(20))

      flat_list_data = [item for sublist in master_list for item in sublist[0]]
      flat_list_label = [item for sublist in master_list for item in sublist[1]]
      
      val, params, opt_state = update(params, opt_state, flat_list_data, flat_list_label)
      print(val)
  except KeyboardInterrupt:
    file = open('partial_training.params', 'wb')
    pickle.dump(params, file)
    file.close()
    exit()

  file = open('trained-model.params', 'wb')
  pickle.dump(params, file)
  file.close()
  print("Bye")
  exit()

if __name__ == "__main__":
  app.run(main)