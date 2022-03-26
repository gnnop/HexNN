


#First I'm going to start with an 8 by 8 board:

class hexGame():

    """Creates a new game of Hex"""
    def __init__(self, gameSize):
        #Note that this is currently optimized for the 8 by 8 board. Very specific
        self.gameSize = gameSize
        self.hexes = np.array([0 for i in range(gameSize**2)])
        self.winState = [[[(-1, i) for i in range(gameSize)], [(i, -1) for i in range(gameSize)]], [ [0]*gameSize for i in range(gameSize)]]
        self.posTurn = 1
        self.hexNeighbors = [[-1, 1], [0, 1], [1, 0], [1, -1], [-1, 0], [0, -1]]
        #Red starts. Currently, we have no PI rule, I'm going to introduce that later
    
    #We check whose turn it is, and determine if they are playing on something already known.
    def hexToLine(x, y):
      return 0

    def lineToHex(i):
      return [0, 0]
    
    def takeTurn(self, x, y):
      if self.hexes[self.hexToLine[x, y]] == 0:
        self.hexes = self.posTurn
        self.posTurn *= -1


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



# Copyright 2020 DeepMind Technologies Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""MNIST classifier example."""

from typing import Iterator, Mapping, Tuple

from absl import app
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import optax
import tensorflow_datasets as tfds

Batch = Mapping[str, np.ndarray]


def net_fn(batch: Batch) -> jnp.ndarray:
  """Standard LeNet-300-100 MLP network."""
  x = batch["image"].astype(jnp.float32) / 255.
  mlp = hk.Sequential([
      hk.Flatten(),
      hk.Linear(300), jax.nn.relu,
      hk.Linear(100), jax.nn.relu,
      hk.Linear(10),
  ])
  return mlp(x)


def load_dataset(
    split: str,
    *,
    is_training: bool,
    batch_size: int,
) -> Iterator[Batch]:
  """Loads the dataset as a generator of batches."""
  ds = tfds.load("mnist:3.*.*", split=split).cache().repeat()
  if is_training:
    ds = ds.shuffle(10 * batch_size, seed=0)
  ds = ds.batch(batch_size)
  return iter(tfds.as_numpy(ds))


def main(_):
  # Make the network and optimiser.
  net = hk.without_apply_rng(hk.transform(net_fn))
  opt = optax.adam(1e-3)

  # Training loss (cross-entropy).
  def loss(params: hk.Params, batch: Batch) -> jnp.ndarray:
    """Compute the loss of the network, including L2."""
    logits = net.apply(params, batch)
    labels = jax.nn.one_hot(batch["label"], 10)

    l2_loss = 0.5 * sum(jnp.sum(jnp.square(p)) for p in jax.tree_leaves(params))
    softmax_xent = -jnp.sum(labels * jax.nn.log_softmax(logits))
    softmax_xent /= labels.shape[0]

    return softmax_xent + 1e-4 * l2_loss

  # Evaluation metric (classification accuracy).
  @jax.jit
  def accuracy(params: hk.Params, batch: Batch) -> jnp.ndarray:
    predictions = net.apply(params, batch)
    return jnp.mean(jnp.argmax(predictions, axis=-1) == batch["label"])

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

  # Make datasets.
  train = load_dataset("train", is_training=True, batch_size=1000)
  train_eval = load_dataset("train", is_training=False, batch_size=10000)
  test_eval = load_dataset("test", is_training=False, batch_size=10000)

  # Initialize network and optimiser; note we draw an input to get shapes.
  params = avg_params = net.init(jax.random.PRNGKey(42), next(train))
  opt_state = opt.init(params)

  # Train/eval loop.
  for step in range(10001):
    if step % 1000 == 0:
      # Periodically evaluate classification accuracy on train & test sets.
      train_accuracy = accuracy(avg_params, next(train_eval))
      test_accuracy = accuracy(avg_params, next(test_eval))
      train_accuracy, test_accuracy = jax.device_get(
          (train_accuracy, test_accuracy))
      print(f"[Step {step}] Train / Test accuracy: "
            f"{train_accuracy:.3f} / {test_accuracy:.3f}.")

    # Do SGD on a batch of training examples.
    params, opt_state = update(params, opt_state, next(train))
    avg_params = ema_update(params, avg_params)

if __name__ == "__main__":
  app.run(main)