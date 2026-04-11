
# serve as a distributer between representation, dynamics and prediction

import pickle


class NNManager():

    def __init__(self) -> None:
        pass

    ####### FROM ASSIGNMENT TEXT. UNSURE WHERE TO PUT. THINK HERE #######
    def episode_loop(self):
        pass
    def do_rollout(self, node, depth, NN_d, NN_p):
        pass
    def do_backpropagation(self, node, goal_node, rewards):
        pass
    def do_bptt_training(self, trinet, epsiode_history, mbs):
        pass
    

    def save_params(self, params, filename):
        with open(filename + ".pkl", "wb") as f:
            pickle.dump(params, f)

    def load_params(self, filename):
        with open(filename + ".pkl", 'rb') as f:
            loaded_data = pickle.load(f)
        return loaded_data