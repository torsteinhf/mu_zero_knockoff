from typing import Any

from game_interface import Game_Interface

# Two Thousand and Fourty Eight
class TTFE(Game_Interface):
    
    def __init__(self, grid_size) -> None:
        self.grid_size = grid_size

    def transition(self, state, action) -> Any:
        pass
    
