
import random
from config import ROLL_AHEAD, BUFFER_SIZE

class EpisodeBuffer():

    def __init__(self, max_size: int = BUFFER_SIZE) -> None:
        self.episodes = []
        self.max_size = self.max_size

    def add_episode(self, episode):
        if episode:
            self.episodes.append(episode)
        if len(self.episodes) > self.max_size:
            self.episodes.pop(0)

    def sample(self, batch_size: int) -> list:
        batch = []
        for _ in range(batch_size):
            ep = random.choice(self.episodes) # select ep
            k = random.randint(0, len(ep) - 1) # select obs in ep
            batch.append({
                "state_seq": ep[k]["state_seq"],
                "actions":  [ep[k + t]["action"] for t in range(ROLL_AHEAD)],
                "policies": [ep[k + t]["action"] for t in range(ROLL_AHEAD + 1)],
                "values":   [ep[k + t]["action"] for t in range(ROLL_AHEAD + 1)],
                "rewards":  [ep[k + t]["action"] for t in range(ROLL_AHEAD)],
            })
        return batch
    
    def __len__(self):
        return len(self.episodes)