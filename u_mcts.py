
import numpy as np

from config import NUM_ACTIONS, N_SIMULATIONS, MAX_ROLLOUT_DEPTH, PUCT_C

class UNode:

    def __init__(self, abstract_state, prior: float, parent, incoming_action, incoming_reward: float) -> None:
        self.abstract_state = abstract_state
        self.prior = prior          # P(parent_state, this_action) from NNp
        self.parent = parent
        self.incoming_action = incoming_action
        self.incoming_reward = incoming_reward
        self.visits = 0
        self.total_value = 0.0
        self.children = {}          # action_idx : UNode

    def is_fully_expanded(self) -> bool:
        return len(self.children) > 0
    
    def puct_score(self, c: float = PUCT_C) -> float:
        parent_visits = self.parent.visits if self.parent else 1
        q_value = self.total_value / self.visits if self.visits > 0 else 0.0
        exploration = c * self.prior * np.sqrt(parent_visits) / (1 + self.visits)
        return q_value + exploration



class UMCTS():

    def __init__(self, nn_manager) -> None:
        self.nn_m = nn_manager

    def search(self, state_seq, legal_actions: list):
        #Building root
        abstract = self.nn_m.represent(state_seq)
        root_policy, root_value = self.nn_m.predict(abstract)
        root_policy = np.array(root_policy)
        root = UNode(abstract, prior=1.0, parent=None,
                     incoming_action=None, incoming_reward=0.0)
        root.visits = 1
        # Expand root
        self._expand(root, root_policy, legal_actions)
        # Loop sim
        for _ in range(N_SIMULATIONS):
            # find best child
            node, depth = self._select(root)
            # make sure its expanded
            if not node.is_fully_expanded():
                node_policy, _ = self.nn_m.predict(node.abstract_state)
                self._expand(node, np.array(node_policy), list(range(NUM_ACTIONS))) # expand node with
            # chose random child to rollout
            if node.children:
                child = np.random.choice(list(node.children.values()))
            else:
                child = node # wont happen, just a safe guard
            #rollout from child
            rollout_depth = max(0, MAX_ROLLOUT_DEPTH - depth)
            accum_rewards = self._rollout(child, rollout_depth)
            self._backprop(child, root, accum_rewards)
            
    # traverse from root to deepest expanded node then return best PUCT child as leaf + depth reached
    def _select(self, root: UNode):
        node = root
        depth = 0
        while node.is_fully_expanded(): # ... go to next depth with best puct score
            node = max(node.children.values(), key=lambda c: c.puct_score())
            depth += 1
        return node, depth
        
    def _expand(self, node: UNode, policy_prior: np.ndarray, actions: list):
        for a in actions: # expand all nodes
            next_abstract_state, pred_r = self.nn_m.dynamics(node.abstract_state, a)
            prior = float(policy_prior[a]) if a < len(policy_prior) else 1.0 / NUM_ACTIONS # P(a) from NNd
            child = UNode(next_abstract_state, prior, node, a, float(pred_r))
            node.children[a] = child # adds child node in dict with key = actions

    def _rollout(self, node: UNode, depth: int) -> list:
        abstract = node.abstract_state
        rewards = []
        for _ in range(depth):
            policy, _ = self.nn_m.predict(abstract) # action sampling + value bootstrap
            a = int(np.random.choice(NUM_ACTIONS, p = np.array(policy)))
            abstract, reward = self.nn_m.dynamics(abstract, a) 
            rewards.append(float(reward))
        _, value = self.nn_m.predict(abstract)
        rewards.append(float(value)) # adds last estimate of bottom node
        return rewards

    def _backprop(self, node: UNode, root: UNode, rewards: list):
        current = node
        accum = list(rewards)
        while current is not None: # until root node reached
            current.visits += 1
            current.total_value += sum(accum)
            if current is root:
                break
            accum.append(current.incoming_reward)
            current = current.parent