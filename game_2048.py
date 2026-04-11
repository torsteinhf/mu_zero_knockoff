from random import randint
import numpy as np

class Game_2048():

    rotations = {"LEFT":0, "DOWN":1, "RIGHT":2, "UP":3}

    def __init__(self, size) -> None:
        self.size = size
    
    def get_init_state(self):
        board = [[0] * self.size for _ in range(self.size)]
        board[randint(0,self.size - 1)][randint(0,self.size - 1)] = int(2)
        return board
    
    def deep_copy(self, state):
        return [row[:] for row in state]
    
    def get_actions(self, state):
        old_board = self.deep_copy(state)
        a = []
        actions = []
        for a in ['LEFT', 'RIGHT', 'UP', 'DOWN']:
            new_board, _ = self.update_board(old_board, a)
            if new_board != old_board:
                actions.append(a)
        return actions

    def left_compress(self, row):
        l_comped_row = [n for n in row if n != 0]
        l_comped_row += [0] * (len(row) - len(l_comped_row))
        return l_comped_row
    
    def merge(self, row):
        score = 0
        for i in range(self.size - 1):
            if row[i] != 0 and row[i] == row[i+1]:
                row[i] *= 2
                row[i+1] = 0
                row.pop(i+1)
                row.append(0)
                score += row[i]
        return row, score
    
    def rotate_board(self, state, n):
        for _ in range(n):
            state = [list(row) for row in zip(*state[::-1])] # [claude] (what)
        return state
    
    def update_board(self, state, action):
        n = self.rotations[action] # number of rotations
        copy = self.deep_copy(state)
        rotated_board = self.rotate_board(copy, n) # rotate to reuse code for moving left

        cum_score = 0
        for row in range(self.size):
            l_comp_row = self.left_compress(rotated_board[row])
            merged_row, score = self.merge(l_comp_row)
            rotated_board[row] = merged_row
            cum_score += score
        
        rotated_back = self.rotate_board(rotated_board, (4 - n)) # rotate back again

        return rotated_back, cum_score

    def spawn_new_num(self, state):
        empties = [(r, c) for r in range(self.size) for c in range(self.size) if state[r][c] == 0]
        if not empties:
            return state
        r, c = empties[randint(0, len(empties) - 1)]
        new = 2 if randint(0,9) < 9 else 4
        state[r][c] = new
        return state

    def transition(self, state, action):
        if action not in self.get_actions(state):
            raise ValueError()
        state, cum_score = self.update_board(state, action)
        state = self.spawn_new_num(state)
        fin = self.won(state)
        if fin:
            cum_score += 10_000 # bare endre 
        return state, cum_score, fin
    
    def won(self, state):
        return any(state[r][c]==2048 for r in range(self.size) for c in range(self.size))
    
    def visualize(self, state):
        print("\n\n".join(" ".join(f"{str(int(v)).center(4)}" for v in row) for row in state))