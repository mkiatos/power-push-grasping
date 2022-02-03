from random import Random
import numpy as np
import pickle
import os
import cv2


class ReplayBuffer:
    def __init__(self, log_dir, buffer_size=100000):
        self.buffer_size = buffer_size
        self.random = Random()
        self.count = 0
        self.log_dir = log_dir
        self.buffer_ids = []

        if not os.path.exists(os.path.join(log_dir, 'replay_buffer')):
            os.mkdir(os.path.join(log_dir, 'replay_buffer'))

        self.z_fill = 6

    def __call__(self, index):
        return self.buffer[index]

    def store(self, transition):
        folder_name = os.path.join(self.log_dir, 'replay_buffer/transition_' + str(self.count).zfill(5))
        if os.path.exists(folder_name):
            raise Exception
        os.mkdir(folder_name)

        cv2.imwrite(os.path.join(folder_name, 'heightmap.exr'), transition['state'])
        pickle.dump(transition['action'], open(os.path.join(folder_name, 'action'), 'wb'))

        self.buffer_ids.append(self.count)
        if self.count < self.buffer_size:
            self.count += 1

    def sample(self, given_batch_size=1):
        if self.count < given_batch_size:
            batch_size = self.count
        else:
            batch_size = given_batch_size

        batch_id = self.random.sample(self.buffer_ids, 1)[0]

        folder_name = os.path.join(self.log_dir, 'replay_buffer/transition_' + str(batch_id).zfill(5))
        state = cv2.imread(os.path.join(folder_name, 'heightmap.exr'), -1)
        action = pickle.load(open(os.path.join(folder_name, 'action'), 'rb'))

        return state, action

    def clear(self):
        self.buffer_ids.clear()
        self.count = 0

    def size(self):
        return self.count

    def seed(self, random_seed):
        self.random.seed(random_seed)


class PrioritizedReplayBuffer(ReplayBuffer):
    def __init__(self, log_dir, buffer_size=100000, alpha=2):
        super(PrioritizedReplayBuffer, self).__init__(log_dir, buffer_size)
        self.alpha = alpha
        self.surprise_values = []
        self.max_priority = 0.5

        if not os.path.exists(os.path.join(log_dir, 'replay_buffer')):
            os.mkdir(os.path.join(log_dir, 'replay_buffer'))

        self.z_fill = 6

    def __call__(self, index):
        return self.buffer[index]

    def store(self, error, transition):
        self.surprise_values.append(error)
        super(PrioritizedReplayBuffer, self).store(transition)

    def sample_proportional(self, batch_size):
        sorted_surprise_ids = np.argsort(self.surprise_values)
        rand_sample_ids = np.round(np.random.power(self.alpha, batch_size) * (self.count - 1)).astype(int)
        ids = sorted_surprise_ids[rand_sample_ids]
        return ids

    def sample(self, given_batch_size=1):
        if self.count < given_batch_size:
            batch_size = self.count
        else:
            batch_size = given_batch_size

        batch_id = self.sample_proportional(batch_size)[0]
        print(batch_id)

        folder_name = os.path.join(self.log_dir, 'replay_buffer/transition_' + str(batch_id).zfill(5))
        print(folder_name)
        state = cv2.imread(os.path.join(folder_name, 'heightmap.exr'), -1)
        action = pickle.load(open(os.path.join(folder_name, 'action'), 'rb'))

        return state, action, batch_id

    def update_priorities(self, idxs, priorities):
        self.surprise_values[idxs] = priorities
        self.max_priority = np.max(priorities)