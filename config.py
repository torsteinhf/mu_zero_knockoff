
### 2048 params ###
BOARD_SIZE = 4
NUM_ACTIONS = 4
ACTIONS = ['UP', 'DOWN', 'LEFT', 'RIGHT']
ACTION_TO_IDX = {
    'UP' : 0,
    'DOWN' : 1, 
    'LEFT' : 2, 
    'RIGHT' : 3
}
INPUT_DIM = BOARD_SIZE * BOARD_SIZE
###################

### Representation input ###
LOOKBACK = 0 # let lookback be 0 as the current obs is markov (ca)
SEQ_DIM = (LOOKBACK + 1) * INPUT_DIM
# Youll get lookback + 1 consecutive concatenated states for reference
### Representation output ###
ABSTRACT_DIM = 64

### Neural net training ###
LEARNING_RATE = 3e-4
NUM_EPISODES = 500
EPISODE_STEPS = 400     # max steps per episode
TRAIN_INTERVAL = 5      # number of episodes accumalated before training on nn
MINIBATCH_SIZE = 32 
ROLL_AHEAD = 3          # number of steps in BPTT
BUFFER_SIZE = 200 # max episodes stored in buffer

### u-MCTS ###
N_SIMULATIONS = 100
MAX_ROLLOUT_DEPTH = 10
PUCT_C = 1.25 # exploration constant

### RL & Normalization ###
DISCOUNT = 0.997
MAX_LOG_TILE = 11.0 # used for normalizing input to NNr
MAX_REWARD = 2048.0 # used for normalizing output from NNr to have weighted contribution to loss
