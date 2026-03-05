import config
from game_interface import Game_Interface

# Game State Manager
# Hint: GMS can help the MCTS expand leaf nodes

class GMS():

    def __init__(self) -> None:
        game_chosen, game_params = config.game_chosen, config.game_params
        self.GAME : Game_Interface = config.game_classes[game_chosen](**game_params)

    def get_legal_actions(self, state):
        pass

    def get_action_or_dist(self, state): #actor
        pass 
        # Given a game state, return one good action to perform, or a probability distribution over the actions to perform. 
        # Some GSM implementations may employ a neural network to do this mapping, while others use a lookup table.

    def is_goal_state(self, state):
        pass 
        # if in goal state
    
    def get_eval(self, state): #critic
        pass 
        # Given a game state, return an evaluation of that state. 
        # In some implementations, the GSM may use a neural network to perform this evaluation.