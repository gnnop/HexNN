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

hexDims = 8

def net_fn(batch) -> jnp.ndarray:
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

  #combination network. concatenate previous results
  #and combine to see what happens.
  #may switch to attention transformater


  return mlp(x)

params = None
net = None

def nop_ai(board_representation):
    global params
    global net
    global hexDims

    if params == None:
        net = hk.without_apply_rng(hk.transform(net_fn))
        if exists("partial_training.params"):
            with open('partial_training.params', 'rb') as file:
                params = pickle.load(file)

    gameStates = []
    alphaBetaBoards = []


    for i in range(hexDims**2):
        if board_representation[i] == 0:
            board_representation[i] = board_representation[-1]#in place modification without changing state
            gameStates.append(i)
            alphaBetaBoards.append(copy.deepcopy(board_representation))
            board_representation = 0
    #alpha beta value - the nn returns

    ls = net.apply(params, np.array(alphaBetaBoards))

    return gameStates[np.where(ls == ls)[0][0]]