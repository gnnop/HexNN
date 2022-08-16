import hex
import mooa
import haiku as hk
import colorama


m_t = mooa.load_model()
m_u = mooa.load_model('untrained-model.dat')



# Try it out by playing a test game
@mooa.timer_func
def play_a_game(
  blue_network_parameters: hk.Params,
  red_network_parameters: hk.Params
):
  '''
  Get a feel for the AI's playing ability by pitting it against itself (or a variant of itself)
  '''
  current_board_state = hex.new_game_state()
  current_turn_count = 0
  while True:
    current_turn_count += 1

    # Blue's turn
    current_board_state = mooa.make_best_move(blue_network_parameters, current_board_state, 0)
    print("-----------------------| Turn %d |---------------------" % current_turn_count)
    mooa.hex.print_game_state(current_board_state)
    if hex.check_win(current_board_state, 0):
      print(colorama.Fore.YELLOW + "Blue WINS" + colorama.Fore.RESET)
      return (0, current_turn_count, current_board_state)

    # Red's turn
    current_board_state = mooa.make_best_move(red_network_parameters, current_board_state, 1)
    hex.print_game_state(current_board_state)
    if hex.check_win(current_board_state, 1):
      print(colorama.Fore.YELLOW + "RED WINS" + colorama.Fore.RESET)
      return (1, current_turn_count, current_board_state)
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
  colorama.init()
  play_a_game(m_t, m_u)
