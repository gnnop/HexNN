import haiku as hk
import jax
import jax.numpy as jnp
import optax

def net_fn(batch):
  x = batch.astype(jnp.float32)
  h1 = hk.Sequential([
    hk.Flatten(),
    hk.Linear(20), jax.nn.leaky_relu,
    hk.Linear(30), jax.nn.leaky_relu,
    hk.Linear(30), jax.nn.leaky_relu])
  h2 = hk.Sequential([
    hk.Linear(100), jax.nn.leaky_relu,
    hk.Linear(100), jax.nn.leaky_relu,
    hk.Linear(200), jax.nn.leaky_relu,
    hk.Linear(100), jax.nn.leaky_relu,
    hk.Linear(30), jax.nn.leaky_relu])
  h3 = hk.Sequential([
    hk.Linear(30), jax.nn.leaky_relu,
    hk.Linear(20), jax.nn.leaky_relu,
    hk.Linear(10), jax.nn.leaky_relu,
    hk.Linear(5),  jax.nn.leaky_relu, hk.Linear(1)])
  y1 = h1(x)
  y2 = y1 + h2(y1)
  y3 = h3(y2)

  #Note that I want to normalize to the proper range:



  return (y3 * 2.0) - 1.0