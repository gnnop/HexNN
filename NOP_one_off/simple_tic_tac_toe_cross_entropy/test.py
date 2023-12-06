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

weights = jnp.array([1,0,1,1,1,0])
red_arr = jnp.stack((jnp.array([0.3,0.3,0.3,0.3,0.3,0.3]), weights), axis=1)

def calc_red(cum, el):
    ret = el[1]*cum*(4.0 - el[0]) / 4.0 + (1-el[1])*(4.0 - el[0]) / 4.0#
    return ret, ret
loss, tabl = jax.lax.scan(calc_red, 0.0,red_arr , reverse=True)

dfvb = jnp.stack((tabl, weights), axis=1)
def shift_loss(cum, el):
    return el[0], el[1]*cum + (1-el[1])
_, wegh = jax.lax.scan(shift_loss, 0.0,dfvb , reverse=True)
print(loss)
print(tabl)

print(wegh)

input()