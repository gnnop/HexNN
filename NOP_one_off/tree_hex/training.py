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
import optax
import random
import pickle
import copy
from multiprocessing.dummy import Pool as ThreadPool
from game_description import *
from AI import *

"""

Now, I take the time to construct trees. Instead of the network being trained on its own evaluations,
instead we sample thousands of games to deteremine win percentages and train the networkdirectly on that. 
Eventually, with each epoch, we use the updated network on the tree to generate more choices.


In a sense, this should give a better idea for the net of everything.

Also, I should pre-zero the network maybe, since near zero predictions are expected.

Since the state in theory contains the entire game, I can also store collisions in a hash

"""


#For evaluating nodes
class Trainer():

  def __init__(self) -> None:
    pass



def main(_):
  # Make the network and optimiser.
  net = hk.without_apply_rng(hk.transform(net_fn))
  opt = optax.adam(2e-4)

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
      
      #Do the first turn so results aren't even
      hexgame.takeLinTurn(random.randrange(0, hexgame.getPlayableArea()))
      

      while hexgame.checkGameWin() == -2:
        if pp == 0:
          print("displaying game")
          hexgame.displayGame()
        boards = []
        gamestates = []
        for i in range(hexgame.getPlayableArea()):
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
          


  def generateGameBatch(hexgame,params):

    boards = []
    labels = []

    turns = 0
    while hexgame.checkGameWin() == -2:
      turns += 1
      alphaBetaBoards = []
      gameStates = []

      for i in range(hexgame.getPlayableArea()):
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
      #Use some mix of exploration and the network
      if random.random() < 0.2 and turns < 15:
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
              hexgame.takeLinTurn(absPos)
            num+=1
          absPos+=1
      else:
        hexgame.takeLinTurn(gameStates[np.where(ls == labels[-1])[0][0]])
    
    boards.append(copy.deepcopy(hexgame.board))
    labels.append(hexgame.checkGameWin())

    return boards, labels

  def loss(params: hk.Params, batch_data, batch_labels):
    logits = net.apply(params, jnp.array(batch_data))
    labels = jnp.array(batch_labels)
    tuning_loss = jnp.sum(jnp.square(jax.nn.relu(jnp.abs(logits) - 1)))
    loss = jnp.sum(jnp.square(logits - labels)) / labels.shape[0]
    weight_decay = 0.02 * 0.5 * sum(jnp.sum(jnp.square(p)) for p in jax.tree_leaves(params)) / sum(x.size for x in jax.tree_leaves(params))

    return loss + weight_decay + 5*tuning_loss


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
  try:
    params = pickle.load(open('partial_training.params', 'rb'))
    print('loaded previous AI')
  except:
    print('previous AI not found. New init being used')
    params = net.init(jax.random.PRNGKey(42), jnp.array([hexGame().board]))

  print(params)
  opt_state = opt.init(params)

  grabAI = params

  try:
    for step in range(10001):
      if step % 100 == 0:
        print(step)
      if step % 30 == 0 and step > 0:
        # Periodically evaluate classification accuracy on train & test sets.
        oldScore, newScore = compareAI(grabAI, params)
        print("The old AI scored " + str(oldScore) + "and the new scored " + str(newScore))
        grabAI = copy.deepcopy(params)
        with open('partial_training.params', 'wb') as file:
          pickle.dump(params, file)
      # Do SGD on a batch of training examples.

      pool = ThreadPool(20)
      master_list = pool.map(lambda a: generateGameBatch(hexGame(), params), range(20))

      flat_list_data = (np.array([item for sublist in master_list for item in sublist[0]]))
      flat_list_label = (np.transpose(np.array([[item for sublist in master_list for item in sublist[1]]])))
      
      val, params, opt_state = update(params, opt_state, flat_list_data, flat_list_label)
      print(val)
  except KeyboardInterrupt:
    with open('partial_training.params', 'wb') as file:
      pickle.dump(params, file)
    exit()

  with open('trained-model.params', 'wb') as file:
    pickle.dump(params, file)
  print("Bye")
  exit()

if __name__ == "__main__":
  app.run(main)