import hex
import mooa
import haiku as hk
import colorama
import jax.numpy as jnp
from time import sleep


def print_predicted_game_state(game_state: jnp.ndarray, color, highlight = None):
  # predicted = mooa.super_AI(m_t, game_state, color)
  predicted = mooa.predict_raw_probability(m_t, game_state, color)
  # top red bar
  s = "" # the string to print
  s += colorama.Fore.RED + '--'*(hex.board_size*2+1) + colorama.Fore.RESET + '\n'
  for i in range(hex.board_size):
    # spacing to line up rows hexagonally
    s += ' '*i
    # left blue bar
    s += colorama.Fore.BLUE + '\\' + colorama.Fore.RESET
    # print a row of the game state
    for j in range(hex.board_size):
      character = "{:02d}".format(int(predicted[i][j]*100))
      if highlight and highlight == (j,i):
        character = (colorama.Back.BLUE if color == 0 else colorama.Back.RED) + character + colorama.Back.RESET
      if game_state[0][i][j]==0:
        character = colorama.Fore.BLUE+character+colorama.Fore.RESET
      elif game_state[1][j][i]==0:
        character = colorama.Fore.RED+character+colorama.Fore.RESET
      s += character + '  '
    # right blue bar and end of row
    s += colorama.Fore.BLUE + '\\' + colorama.Fore.RESET + '\n'
  # bottom red bar
  s += ' '*i + ' '
  s += colorama.Fore.RED + '--'*(hex.board_size*2+1) + colorama.Fore.RESET
  print(s)
# end print_game_state


# Try it out by playing a test game
@mooa.timer_func
def play_a_game(
  blue_network_parameters: hk.Params,
  red_network_parameters: hk.Params,
  print_game_state: bool = False,
  print_game_outcome: bool  = False
):
  '''
  Get a feel for the AI's playing ability by pitting it against itself (or a variant of itself)
  Returns true if blue wins, false if red wins
  '''
  current_board_state = hex.new_game_state()
  current_turn_count = 0
  while True:
    current_turn_count += 1

    # Blue's turn
    next_board_state, turn = mooa.make_best_move(blue_network_parameters, current_board_state, 0)
    #print("-----------------------| Turn %d |---------------------" % current_turn_count)
    if hex.check_win(next_board_state, 0):
      if print_game_outcome:
        print(colorama.Fore.YELLOW + "Blue WINS" + colorama.Fore.RESET)
        print_predicted_game_state(current_board_state, 0, turn)
      return (0, current_turn_count, current_board_state)
    elif print_game_state: 
      print_predicted_game_state(current_board_state, 0, turn)
    
    current_board_state = next_board_state

    # Red's turn
    next_board_state, turn  = mooa.make_best_move(red_network_parameters, current_board_state, 1)
    if hex.check_win(next_board_state, 1):
      if print_game_outcome:
        print(colorama.Fore.YELLOW + "Red WINS" + colorama.Fore.RESET)
        print_predicted_game_state(current_board_state, 1, turn)
      return (1, current_turn_count, current_board_state)
    elif print_game_state: 
      print_predicted_game_state(current_board_state, 1, turn)

    current_board_state = next_board_state
  # end while loop
# end play_a_game



# while True:
#   network_parameters = net.init(jax.random.PRNGKey(int(time())), b)
#   final_board_turn_color, final_turn_count, final_board_state = play_a_game(network_parameters)
#   # Check for a win
#   if final_board_turn_color:
#     print("Blue wins!")
#   else:
#     print("Red wins!")

# # Print the results
# print("-------------- Turn %d --------------" % (final_turn_count))
# hex.print_game_state(final_board_state)
# print()




if __name__ == "__main__":
  num_red_wins = 0
  num_blue_wins = 0
  colorama.init()


  # play_a_game(blue_network_parameters = m_t, red_network_parameters = m_u, print_game_state = False, print_game_outcome = True)
  # TODO replace with the code block below when the AI becomes nondeterministic
  while True: # Can fake nondeterminism by training in the background
    m_t = mooa.load_model(  'trained-model.dat')
    m_u = mooa.load_model('untrained-model.dat')
    final_game_state = play_a_game(blue_network_parameters = m_t, red_network_parameters = m_u, print_game_state = False, print_game_outcome = True)
    if final_game_state[0] == 0: # Blue wins
      num_blue_wins += 1
    else:
      num_red_wins += 1
    print("Score: " + colorama.Fore.BLUE + "%d"%(num_blue_wins) + colorama.Fore.RESET + "/" + colorama.Fore.RED + "%d"%(num_red_wins) + colorama.Fore.RESET)
    sleep(1)
    

    
