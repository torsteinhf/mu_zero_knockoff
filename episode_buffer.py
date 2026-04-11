
# episode buffer
class EpisodeBuffer():

    def __init__(self) -> None:
        self.episodes = []

    # anticipating list of 5 tuples
    def add_episode(self, episode):
        self.episodes.append(episode)

    def get_portion_of_ep(self, mbs, num = -1):
        ep = self.episodes[num]
        if mbs > len(ep): 
            mbs = len(ep)
        train = ep[0:mbs][:2]
        target = ep[num][0:mbs][2:]
        return train, target