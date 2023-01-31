import haiku as hk
import jax
import jax.numpy as jnp
import optax

def net_fn(batch):
  x = batch.astype(jnp.float32)
  h1 = hk.Sequential([
    hk.Flatten(),
    hk.Linear(100), jax.nn.relu,
    hk.Linear(300), jax.nn.relu])
  h2 = hk.Sequential([
    hk.Linear(300), jax.nn.relu,
    hk.Linear(300), jax.nn.relu,
    hk.Linear(300), jax.nn.relu])
  h3 = hk.Sequential([
    hk.Linear(300), jax.nn.relu,
    hk.Linear(100), jax.nn.relu,
    hk.Linear(20),  hk.Linear(1)])
  y1 = h1(x)
  y2 = y1 + h2(y1)
  y3 = h3(y2)
  return y3