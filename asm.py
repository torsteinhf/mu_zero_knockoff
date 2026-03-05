
class ASM():

    def __init__(self) -> None:
        pass

    def map_to_abstract(self, g_states):
        pass
        # map sequence of game states to a single abstract state using NN_r

    def get_legal_actions(self, a_state):
        pass
        # normally all actions, except a_states used as root

    def get_policy(self):
        pass
        # return distribution over actions & predicted value using NN_p
    
    def a_step(self, a_state, action):
        pass
        # return next a_state & reward from a_state and action

    

