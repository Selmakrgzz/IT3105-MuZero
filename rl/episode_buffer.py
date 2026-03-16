import random

class EpisodeBuffer:
    def __init__(self):
        self.episodes = []

    def save_episode(self, episode):
        """Save the finished episode"""
        self.episodes.append(episode)
    
    def random_episode(self):
        episode = random.choice(self.episodes)
        return episode

    def fetch_chunk(self, episode_idx, start, length):
        """
        Grab a chunk from an episode for training.
        episode_idx: which episode
        start: which step we start from
        length: how many steps (w = roll-ahead)
        """
        episode = self.episodes[episode_idx]
        return episode[start : start + length]

    def sample_random_chunk(self, length):
        """Grab a random chunk from a random episode"""
        episode = random.choice(self.episodes)
        start = random.randint(0, max(0, len(episode) - length))
        return episode[start : start + length]
    
    def __len__(self):
        return len(self.episodes)