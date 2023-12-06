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
from multiprocessing.dummy import Pool, Manager
from game_description import *
from AI import *


import mglobals
mglobals.net = 0
mglobals.opt = 0

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
        preds = mglobals.net.apply(firstPlayer, jnp.array(boards))
        val = jnp.max(preds)
      else:
        preds = mglobals.net.apply(negOnePlayer, jnp.array(boards))
        val = jnp.min(preds)
      
      hexgame.takeTurn(gamestates[jnp.where(preds == val)[0][0]])
    
    wom = hexgame.checkGameWin()
    if pp == 0:
      print("This player won, blue went first: ", wom)
    if pp % 2 == 0:
      if wom == 1:
        aiOneScore += 1
      elif wom == -1:
        aiTwoScore += 1
    else:
      if wom == 1:
        aiTwoScore += 1
      elif wom == -1:
        aiOneScore += 1

  return (aiOneScore, aiTwoScore)

def loss(params: hk.Params, batch_data, batch_labels, weights):
  logits = mglobals.net.apply(params, jnp.array(batch_data))
  labels = jnp.array(batch_labels)
  tuning_loss = 5.0*jnp.sum(jnp.square(jax.nn.relu(jnp.abs(logits) - 1)))
  intermediate_loss = jnp.square(logits - labels)

  red_arr = jnp.stack((intermediate_loss[:, 0], weights), axis=1)
  def calc_red(cum, el):
      ret = el[1]*cum*(4.0 - el[0]) / 4.0 + (1-el[1])*(4.0 - el[0]) / 4.0#
      return ret, ret
  loss, tabl = jax.lax.scan(calc_red, 0.0,red_arr , reverse=True)

  inter_weights = jnp.stack((tabl, weights), axis=1)
  def shift_loss(cum, el):
      return el[0], el[1]*cum + (1-el[1])
  _, final_weights = jax.lax.scan(shift_loss, 0.0,inter_weights , reverse=True)
  loss = jnp.dot(jax.lax.stop_gradient(final_weights), jnp.square(intermediate_loss)) /  labels.shape[0]
  #loss = jnp.dot(jnp.array(weights), jnp.square(logits - labels)) / labels.shape[0]#the [0] is just for this line
  #loss = jnp.sum(jnp.square(logits - labels)) / labels.shape[0]
  weight_decay = 0.02 * 0.5 * sum(jnp.sum(jnp.square(p)) for p in jax.tree_leaves(params)) / sum(x.size for x in jax.tree_leaves(params))

  #print(loss.shape)
  #print(weight_decay.shape)
  return loss[0]  + weight_decay + tuning_loss


@jax.jit
def update(
    params: hk.Params,
    opt_state: optax.OptState,
    batch_data, batch_labels, weights
) -> Tuple[hk.Params, optax.OptState]:
  """Learning rule (stochastic gradient descent)."""
  val, grads = jax.value_and_grad(loss)(params, batch_data, batch_labels, weights)
  updates, opt_state = mglobals.opt.update(grads, opt_state)
  new_params = optax.apply_updates(params, updates)
  return val, new_params, opt_state

def generateGameBatch(params):

  boards = []
  labels = []

  turns = 0
  hexgame = tictactoe()

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
    ls = mglobals.net.apply(params, np.array(alphaBetaBoards))
    if hexgame.getTurn() == 1: #simple alpha beta
      boards.append(copy.deepcopy(hexgame.board))
      labels.append(jnp.max(ls))
    else:
      boards.append(copy.deepcopy(hexgame.board))
      labels.append(jnp.min(ls))
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


def main(_):
  # Make the network and optimiser.
  mglobals.net = hk.without_apply_rng(hk.transform(net_fn))
  mglobals.opt = optax.adam(8e-4)

  # Initialize network and optimiser; note we draw an input to get shapes.
  try:
    params = pickle.load(open('partial_training.params', 'rb'))
    print('loaded previous AI')
  except:
    print('previous AI not found. New init being used')
    params = mglobals.net.init(jax.random.PRNGKey(42), jnp.array([tictactoe().board]))

  opt_state = mglobals.opt.init(params)

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

      pool = Pool(processes=10)
      args_list = [(params) for _ in range(30)]
      master_list = pool.map(generateGameBatch, args_list)

      def cu(l, i):
        from_end = len(l) - i -1
        if from_end == 0:
          return 0
        else:
          return 1


      flat_list_distance = (np.array([cu(sublist[0], index) for sublist in master_list for index, _ in enumerate(sublist[0])]))
      flat_list_data = (np.array([item for sublist in master_list for item in sublist[0]]))
      flat_list_label = (np.transpose(np.array([[item for sublist in master_list for item in sublist[1]]])))
      
      val, params, opt_state = update(params, opt_state, flat_list_data, flat_list_label, flat_list_distance)
      print(val)
  except KeyboardInterrupt:
    pool.close()
    pool.join()
    with open('partial_training.params', 'wb') as file:
      pickle.dump(params, file)
    exit()

  pool.close()
  pool.join()
  with open('trained-model.params', 'wb') as file:
    pickle.dump(params, file)
  print("Bye")
  exit()

if __name__ == "__main__":
  app.run(main)