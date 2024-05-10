import jax
import jax.numpy as jnp
import chex
import mctx
import functools
from game_tictactoe import *
from game_hex import *

import jax.profiler
jax.profiler.start_server(9999)

'''
A 2xnxn array where n is the board size.
The axes are: color, row, column.
An element on the board is zero if a piece has been placed there. They are 1 by default.
For example: If (0,2,3) is zero, player 0 (the 1st player) placed a piece on the 2nd row 
and 3rd column sometime during the game.

If that was the first action in a game, the game board might be 
visualized like this:

    0 1 2 3 4
    -----------
  0 \. . . . . \
  1  \. . . . . \
  2   \. . . W . \
  3    \. . . . . \
  4     \. . . . . \
         -----------
'''


Player = chex.Array

game = Hex(5)

@chex.dataclass
class Environment:
    state: GameState
    player: Player
    done: GameDone
    reward: GameReward


def environment_reset(_):
    return Environment(
        state=game.initial_state(),
        player=jnp.int8(1),
        done=jnp.bool_(False),
        reward=jnp.int8(0)
    )

def environment_step(environment: Environment, action_id: GameAction) -> tuple[Environment, GameReward, GameDone]:

    player = (1-environment.player) >> 1 # Map from 1,-1 players to 0,1 players 
    y, x = jnp.divmod(action_id, game.get_board_shape()[1])
    action = jnp.zeros(3, dtype=jnp.int8)
    action = action.at[0].set(player)
    action = action.at[1].set(y)
    action = action.at[2].set(x)

    c_valid = game.is_valid_action(environment.state, action)
    c_state, c_reward = game.take_turn(environment.state, action)
    can_continue = (~environment.done) & c_valid

    # The reward:
    # * 0 if the game is already over. This is to ignore nodes below terminal nodes.
    # * -1 if the move is invalid
    # * Fallback to game reward
    out_reward = jnp.where(
        environment.done,
        0,             # Don't reward games that are already over
        jnp.where(
            c_valid,
            c_reward,  # Fallback to the game reward if valid move
            -1         # Override game reward to penalize invalid moves
        )
    ).astype(jnp.int8)

    out_done = (~can_continue) | (c_reward != 0)
    # out_done = c_done | environment.done

    

    out_environment = Environment(
         state = jnp.where(can_continue, c_state, environment.state),
        player = jnp.where(~out_done & can_continue, -environment.player, environment.player),
          done = out_done,
        reward = out_reward
    )

    return out_environment, out_reward, out_done
# end environment_step

def valid_action_mask(environment: Environment) -> chex.Array:
    '''
    Computes which actions are valid in the current state.
    Returns an array of booleans, indicating which indices are empty
    '''
    c_mask = game.get_valid_action_mask(environment.state).astype(jnp.int8)
    return jnp.multiply(c_mask, 1 - environment.done).astype(jnp.int8)

def action_result(environment: Environment, action_id: int):
    """
    Computes the new game state and reward for a given action.
    """
    new_state, reward = game.take_turn(environment.state, action_id)
    return reward

def winning_action_mask(environment: Environment) -> chex.Array:
    def expand_action(environment: Environment, action_id):
        player = environment.player
        y, x = jnp.divmod(action_id, game.get_board_shape()[1])
        action = jnp.zeros(3, dtype=jnp.int8)
        action = action.at[0].set(player)
        action = action.at[1].set(y)
        action = action.at[2].set(x)
        return action
    # Get all valid actions mask
    valid_mask = valid_action_mask(environment)
    # Create an array of action indices
    action_ids = jnp.arange(valid_mask.size)
    # Expand actions from these indices
    actions = jax.vmap(expand_action, in_axes=(None, 0))(environment, action_ids)
    # Convert player IDs from 1, -1 to 0, 1
    player_ids = (1 - actions[:, 0]) >> 1
    actions = actions.at[:, 0].set(player_ids)
    # Apply actions to the game state and get new states and rewards
    new_states, rewards = jax.vmap(lambda st, act: game.take_turn(st, act), in_axes=(None, 0))(environment.state, actions)
    # Mask rewards where action was valid and the reward indicates a win
    winning_mask = (rewards > 0) & valid_mask
    return winning_mask

def policy_function(environment: Environment) -> chex.Array:
    antagonist_environment = Environment(
         state = environment.state,
        player = -environment.player,
          done = environment.done,
        reward = environment.reward
    )

    # return valid_action_mask(environment=environment).astype(jnp.float32) * 100 \
    #     + environment.reward.astype(jnp.float32) * -900

    return valid_action_mask(environment=environment).astype(jnp.float32) * 100 \
        + winning_action_mask(environment=environment).astype(jnp.float32) * 900

    # return valid_action_mask(environment=environment).astype(jnp.float32) * 100 \
    #     + winning_action_mask(environment=antagonist_environment).astype(jnp.float32) * 400 \
    #     + winning_action_mask(environment=environment).astype(jnp.float32) * 900


def rollout(environment: Environment, rng_key: chex.PRNGKey) -> GameReward:
    '''
    Plays a game until the end and returns the reward from the perspective of the initial player.
    '''
    def cond(a):
        environment, key = a
        return ~environment.done
    def step(a):
        environment, key = a
        key, subkey = jax.random.split(key)
        action_id = jax.random.categorical(subkey, policy_function(environment)).astype(jnp.int8)
        environment, reward, done = environment_step(environment, action_id)
        return environment, key
    leaf, key = jax.lax.while_loop(cond, step, (environment, rng_key))
    # The leaf reward is from the perspective of the last player.
    # We negate it if the last player is not the initial player.
    return leaf.reward * leaf.player * environment.player

def value_function(environment: Environment, rng_key: chex.PRNGKey) -> chex.Array:
    return rollout(environment, rng_key).astype(jnp.float32)

def root_fn(environment: Environment, rng_key: chex.PRNGKey) -> mctx.RootFnOutput:
    return mctx.RootFnOutput(
        prior_logits=policy_function(environment),
        value=value_function(environment, rng_key),
        # We will use the `embedding` field to store the environment.
        embedding=environment,
    )

def recurrent_fn(params, rng_key, action_id, embedding):
    # Extract the environment from the embedding.
    environment = embedding

    # Play the action.
    environment, reward, done = environment_step(environment, action_id)

    # Create the new MCTS node.
    recurrent_fn_output = mctx.RecurrentFnOutput(
        # reward for playing `action`
        reward=reward,
        # discount explained in the next section
        discount=jnp.where(done, 0, -1).astype(jnp.float32),
        # prior for the new state
        prior_logits=policy_function(environment),
        # value for the new state
        value=jnp.where(done, 0, value_function(environment, rng_key)).astype(jnp.float32),
    )

    # Return the new node and the new environment.
    return recurrent_fn_output, environment


@functools.partial(jax.jit, static_argnums=(2,))
def run_mcts(rng_key: chex.PRNGKey, environment: Environment, num_simulations: int) -> chex.Array:
    batch_size = 1
    key1, key2 = jax.random.split(rng_key)
    policy_output = mctx.muzero_policy(
        # params can be used to pass additional data to the recurrent_fn like neural network weights
        params=None,

        rng_key=key1,

        # create a batch of environments (in this case, a batch of size 1)
        root=jax.vmap(root_fn, (None, 0))(environment, jax.random.split(key2, batch_size)),

        # automatically vectorize the recurrent_fn
        recurrent_fn=jax.vmap(recurrent_fn, (None, None, 0, 0)),

        num_simulations=num_simulations,

        # we limit the depth of the search tree to the size of the board
        max_depth=game.get_board_shape()[0] * game.get_board_shape()[1],

        # our value is in the range [-1, 1], so we can use the min_max qtransform to map it to [0, 1]
        qtransform=functools.partial(mctx.qtransform_by_min_max, min_value=-1, max_value=1),

        # Dirichlet noise is used for exploration which we don't need in this example (we aren't training)
        dirichlet_fraction=0.0,
    )
    return policy_output

environment = environment_reset(0)



if __name__ == "__main__":
    print(value_function(environment=environment, rng_key=jax.random.PRNGKey(42)))
    policy_output = run_mcts(jax.random.PRNGKey(0), environment, 2)
    print(policy_output.action_weights)

    # set to False to enable human input
    player_1_ai = True
    player_2_ai = False
    ai_level = 25000

    key = jax.random.PRNGKey(43)
    game.print_game_state(environment.state)
    while True:
        if player_1_ai:
            action_weights = jnp.sum(run_mcts(key, environment, ai_level).action_weights, axis=0)
            print(action_weights.reshape(game.get_board_shape()))
            action_id = action_weights.argmax().item()
        else:
            action_id = jnp.array(int(input()), dtype=jnp.int8)

        environment, reward, done = environment_step(environment, action_id)
        game.print_game_state(environment.state)
        # Temporary: debug
        if(game.check_win(environment.state, 0)):
            print("Win condition met for player 1")
        #end debug
        if done: break


        if player_2_ai:
            action_weights = jnp.sum(run_mcts(key, environment, ai_level).action_weights, axis=0)
            print(action_weights.reshape(game.get_board_shape()))
            action_id = action_weights.argmax().item()
        else:
            action_id = jnp.array(int(input()), dtype=jnp.int8)

        environment, reward, done = environment_step(environment, action_id)
        game.print_game_state(environment.state)
        # Temporary: debug
        if(game.check_win(environment.state, 1)):
            print("Win condition met for player 2")
        #end debug
        if done: break

    players = {
        1: "Blue",
        0: "Error",
        -1: "Red",
    }
    print(f"Winner: {players[environment.player.item()]}")
    print(environment)