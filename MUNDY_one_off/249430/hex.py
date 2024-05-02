import hex_engine

class Hex:
    def __init__(self):
        self.board_size = 8
        self.game_state = hex_engine.new_game_state()
        self.current_color = 0  # White's turn

    def place_piece(self, x: int, y: int) -> None:
        if self.game_state[0][y][x] * self.game_state[1][x][y]:
            if self.current_color == 0:
                self.game_state = hex_engine.place_white_piece(self.game_state, x, y)
            else:
                self.game_state = hex_engine.place_black_piece(self.game_state, x, y)
            self.current_color = next_color(self.current_color)

    def check_win(self) -> bool:
        return hex_engine.check_win(self.game_state, self.current_color)

    def swap_sides(self) -> None:
        self.game_state = hex_engine.swap(self.game_state)
        self.current_color = 1 - self.current_color

    def print_game_state(self) -> None:
        print(hex_engine.print_game_state(self.game_state))

    def free_cells(self) -> jnp.ndarray:
        return hex_engine.free_cells(self.game_state)

    def check_free(self, x: int, y: int) -> bool:
        return hex_engine.check_free(self.game_state, x, y)
