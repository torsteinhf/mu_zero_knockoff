#################################################################
#       TEMPORARY CLASS UNTIL ABSTRACT STATES IMPLEMENTET       #
#################################################################

import numpy as np
from game_2048 import Game_2048

class Node:

    def __init__(self, state, parent, unexplored_actions, incoming_action):
        self.state = state
        self.parent = parent
        self.incoming_action = incoming_action

        self.rewards = 0
        self.visits = 0
        
        self.unexplored_actions = unexplored_actions
        self.children = []
    
    def is_fully_expanded(self):
        return len(self.unexplored_actions) == 0
 
    def is_terminal(self):
        return len(self.children) == 0 and self.is_fully_expanded()
    
    def U(self, c = 1):
        if self.visits == 0: return float('inf')
        Q = self.rewards / self.visits if self.visits > 0 else 0.0 # Q-val
        u = c * np.sqrt( np.log(self.parent.visits) / (1 + self.visits)) #exploration bonus
        return Q + u

    def best_child(self, c=1):
        return max(self.children, key=lambda n: n.U(c))
    
class MCTS():

    def __init__(self, game: Game_2048, n_simulations=100, max_rollout_depth=50):
        self.game = game
        self.n_simulations = n_simulations
        self.max_rollout_depth = max_rollout_depth

    def search(self, state):
        root = Node(state, None, self.game.get_actions(state), None)
        root.visits = 1
        for _ in range(self.n_simulations):
            node = self.select_node(root) # Select
            if node.unexplored_actions: # Expand.   VURDERE Å LEGGE TIL SJEKK OM "FERDIG" 
                node = self.expand(node)
            reward = self.rollout(node) # rollout
            self.backprop(node, reward)
        best = max(root.children, key=lambda n: n.visits)
        return best.incoming_action

    def select_node(self, root: Node):
        node = root
        while not node.is_fully_expanded() and node.children: # let fram til man finner ny å expande, eller ikke finner 
            node = node.best_child()
        return node 
    
    def expand(self, node: Node):
        action = node.unexplored_actions.pop(0)
        new_state, _, _ = self.game.transition(node.state, action)
        child = Node(new_state, node, self.game.get_actions(new_state), action)
        node.children.append(child)
        return child

    def rollout(self, node: Node):
        state = self.game.deep_copy(node.state)
        total_reward = 0

        for _ in range(self.max_rollout_depth):
            actions = self.game.get_actions(state)
            if not actions:
                break
            action = actions[np.random.randint(len(actions))]
            state, r, fin = self.game.transition(state, action)
            total_reward += r
            if fin:
                break

        return total_reward

    def backprop(self, node: Node, reward):
        while node is not None:
            node.visits += 1
            node.rewards += reward # may involve discounting
            node = node.parent
