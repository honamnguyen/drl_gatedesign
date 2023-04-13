import numpy as np, pickle
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.utils.annotations import override

class SaveBestEpisodesCallback(DefaultCallbacks):
    def __init__(self, num_episodes_to_save=10):
        super().__init__()
        print('\nInitialize SaveBestEpisodesCallback\n')
        self.num_episodes_to_save = num_episodes_to_save
        self.episode_rewards = []
        self.saved_episodes = []
        self.past_dones = []
        self.past_rewards = []
        self.past_actions = []
        
    @override(DefaultCallbacks)
    def on_sample_end(self, worker, samples, **kwargs):
        if 'actions' in samples:
            dones = np.hstack([self.past_dones,samples['dones']]).astype(int)
            rewards = np.hstack([self.past_rewards,samples['rewards']])
            actions = np.hstack([self.past_actions,samples['actions']]).astype(int)
            done_inds = np.where(dones==True)[0]
            i = 0
            for done_ind in done_inds:
                episode_reward = rewards[i:done_ind+1].sum()            
                if len(self.saved_episodes) < self.num_episodes_to_save:
                    self.saved_episodes.append(actions[i:done_ind+1])
                    self.episode_rewards.append(episode_reward)
                elif episode_reward > min(self.episode_rewards):
                    worst_episode_index = self.episode_rewards.index(min(self.episode_rewards))
                    self.saved_episodes.pop(worst_episode_index)
                    self.saved_episodes.append(actions[i:done_ind+1])
                    self.episode_rewards.pop(worst_episode_index)
                    self.episode_rewards.append(episode_reward)
                i = done_ind+1
            self.past_dones = dones[i:]
            self.past_rewards = rewards[i:]
            self.past_actions = actions[i:]
            data = {
                'rewards': self.episode_rewards,
                'episodes': self.saved_episodes,
            }
            pickle.dump(data, open(worker.io_context.log_dir+f'best_episodes_worker{worker.worker_index}.pkl', 'wb') )
