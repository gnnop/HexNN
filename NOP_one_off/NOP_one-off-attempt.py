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
          self.gameSize = hexDims
          self.hexes = np.array([0 for i in range(self.gameSize**2 + 1)])
          self.hexes[self.turnPos] = 1
          self.hexNeighbors = [[-1, 1], [0, 1], [1, 0], [1, -1], [-1, 0], [0, -1]]
          self.winArray = [[[(-1, i) for i in range(hexDims)], [(i, -1) for i in range(hexDims)]], [ [0]*hexDims for i in range(hexDims)]]
          #Red starts. Currently, we have no PI rule, I'm going to introduce that later
        else:
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
        print("BAD!!!")
        return False

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
        for state in [-1, 1]:
            for loc in self.winArray[0][int((1+state) / 2)]:
                for k in self.hexNeighbors:
                    if 0 <= loc[0] + k[0] < self.gameSize and 0 <= loc[1] + k[1] < self.gameSize:#add in something to stop checking when filled around
                        if self.winArray[1][loc[0] + k[0]][loc[1] + k[1]] == 0 and self.hexes[self.hexToLine(loc[0] + k[0],loc[1] + k[1])] == state:
                            self.winArray[0][int((1+state) / 2)].append((loc[0] + k[0], loc[1] + k[1]))
                            self.winArray[1][loc[0] + k[0]][loc[1] + k[1]] = state
        for i in range(self.gameSize):
            if self.winArray[1][i][self.gameSize-1] == 1:
                return 1
            if self.winArray[1][self.gameSize - 1][i] == -1:
                return -1
        return 0


Batch = Mapping[str, np.ndarray]



#This is the neural networkd, I want it to take in a representation
#with two parameters



def net_fn(batch: Batch) -> jnp.ndarray:
  """Standard LeNet-300-100 MLP network."""
  x = batch["data"].astype(jnp.float32)

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

  def generateGameBatch(hexgame,params):
    global hexDims

    boards = {"data" : [], "label" : []}

    ii = 0

    while hexgame.checkGameWin() == 0:

      boards["data"].append(hexgame.hexes)

      alphaBetaBoards = []

      foundGameWin = False

      for i in range(hexDims**2):
        if hexgame.hexes[i] == 0 and not foundGameWin:
          hexgame.hexes[i] = hexgame.getHexTurn()
          if hexgame.getHexTurn() == hexGame(hexgame).checkGameWin() == 1:
            foundGameWin = True
            boards["label"].append(1)
          elif hexgame.getHexTurn() == hexGame(hexgame).checkGameWin() == -1:
            foundGameWin = True
            boards["label"].append(-1)

          alphaBetaBoards.append(hexgame.hexes)
          hexgame.hexes[i] = 0
      #alpha beta value - the nn returns 

      #Now serialize the boards and apply everything:
      if not foundGameWin:
        ls = net.apply(params, alphaBetaBoards)
        if hexgame.getHexTurn() == 1: #simple alpha beta
          boards["label"].append(max(ls))
        else:
          boards["label"].append(min(ls))

      #Use some mix of exploration and the network
      if random.random() < 0.3:
        hexgame.takeTurn(random.randint(0, hexDims - 1), random.randint(0, hexDims - 1))
      else:
        hexgame.takeLinTurn(boards["label"][-1])
      
    boards["label"] = jnp.array(boards["label"])
    boards["data"] = jnp.array(boards["data"])
    return boards

  # Training loss (cross-entropy).
  def loss(params: hk.Params, batch: Batch) -> jnp.ndarray:
    """Compute the loss of the network, including L2."""
    logits = net.apply(params, batch)
    labels = batch["label"]

    l2_loss = 0.5 * sum(jnp.sum(jnp.square(p)) for p in jax.tree_leaves(params))
    softmax_xent = -jnp.sum(labels * jax.nn.log_softmax(logits))
    softmax_xent /= labels.shape[0]

    return softmax_xent + 1e-4 * l2_loss

  # Evaluation metric (classification accuracy).
  @jax.jit
  def accuracy(params: hk.Params, batch: Batch) -> jnp.ndarray:
    predictions = net.apply(params, batch)
    return jnp.sum(jnp.square(predictions - batch["label"])) / predictions.size

  @jax.jit
  def update(
      params: hk.Params,
      opt_state: optax.OptState,
      batch: Batch,
  ) -> Tuple[hk.Params, optax.OptState]:
    """Learning rule (stochastic gradient descent)."""
    grads = jax.grad(loss)(params, batch)
    updates, opt_state = opt.update(grads, opt_state)
    new_params = optax.apply_updates(params, updates)
    return new_params, opt_state

  # We maintain avg_params, the exponential moving average of the "live" params.
  # avg_params is used only for evaluation (cf. https://doi.org/10.1137/0330046)
  @jax.jit
  def ema_update(params, avg_params):
    return optax.incremental_update(params, avg_params, step_size=0.001)

  # Initialize network and optimiser; note we draw an input to get shapes.
  params = avg_params = net.init(jax.random.PRNGKey(42), {"data" : jnp.array([hexGame().hexes]), "label" : jnp.array([0.0])})
  opt_state = opt.init(params)

  # Train/eval loop.
  for step in range(100001):
    if step % 1000 == 0:
      # Periodically evaluate classification accuracy on train & test sets.
      games = generateGameBatch(hexGame(), avg_params)
      train_accuracy = accuracy(avg_params, games)
      train_accuracy = jax.device_get(train_accuracy)
      print(f"[Step {step}] Game accuracy, may be inaccurate: "
            f"{train_accuracy:.3f}.")

    # Do SGD on a batch of training examples.
    params, opt_state = update(params, opt_state, generateGameBatch(hexGame(), params))
    avg_params = ema_update(params, avg_params)

  file = open('trained-model.params', 'wb')
  pickle.dump(avg_params, file)
  file.close()
  print("Bye")
  exit()

if __name__ == "__main__":
  app.run(main)