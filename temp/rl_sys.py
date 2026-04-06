#################################################################
#       TEMPORARY CLASS UNTIL ABSTRACT STATES IMPLEMENTET       #
#################################################################

from game_2048 import Game_2048
from mcts import MCTS

class RL_SYS():

    def __init__(self) -> None:
        pass

    def run(self, game, max_iter, mcts: MCTS):
        state = game.get_init_state()
        game.visualize(state)
        tot_reward = 0

        for i in range(max_iter):
            if not game.get_actions(state): 
                print("no more moves")
                break
            
            action = mcts.search(state)
            state, r, fin = game.transition(state, action) 
            tot_reward += r
            game.visualize(state)
            print(i, "- action:", action, "- reward:", r)
        print("Total reward:", tot_reward)


game : Game_2048 = Game_2048(4)
mcts : MCTS = MCTS(game, 100, 100)
rl_sys : RL_SYS = RL_SYS()
rl_sys.run(game, 200, mcts)