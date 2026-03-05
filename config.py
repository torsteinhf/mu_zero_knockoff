from video_games.ttfe import TTFE

I_t = ... #training interval for the three neural networks: Ψ = (NNr,NNd,NNp) 
N_e = ... #number of episodes; 
N_es = ... #number of steps in each episode 
M_s = ... #number of searches in the u-MCTS tree (u-Tree) 
d_max = ... #maximum search depth in a u-Tree 
A = ... #complete set of actions 
mbs = ... #minibatch size
    
LR = 0.01
EPOCHS = 10
INIT_W_RANGE = [-0.01, 0.01]


# For games
game_chosen = "ttfe"

game_params = {
    "ttfe" : {
        "grid_size" : 2
    }
}

game_classes = {
    "ttfe" : TTFE
}