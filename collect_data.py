import yaml
import numpy as np
import copy
import argparse
import os

from ppg.environment import Environment
from ppg.agent import PushGrasping
from ppg.utils.memory import ReplayBuffer


def collect_random_dataset(args):
    log_folder = 'logs_tmp'
    # Create a folder logs if it does not exist
    if not os.path.exists(log_folder):
        os.mkdir(log_folder)

    # Create a buffer to store the data
    memory = ReplayBuffer(os.path.join(log_folder, 'ppg-dataset'))

    env = Environment(assets_root='assets/', objects_set='seen')
    env.singulation_condition = args.singulation_condition

    with open('yaml/bhand.yml', 'r') as stream:
        params = yaml.safe_load(stream)

    policy = PushGrasping(params)
    policy.seed(args.seed)

    rng = np.random.RandomState()
    rng.seed(args.seed)

    for j in range(args.n_samples):
        episode_seed = rng.randint(0, pow(2, 32) - 1)
        env.seed(episode_seed)
        obs = env.reset()
        print('Episode seed:', episode_seed)

        while not policy.init_state_is_valid(obs):
            obs = env.reset()

        for i in range(15):
            state = policy.state_representation(obs)

            # Select action
            action = policy.guided_exploration(state)
            env_action = policy.action(action)

            # Step environment.
            next_obs, grasp_info = env.step(env_action)

            if grasp_info['stable']:
                transition = {'obs': obs, 'state': state, 'action': action, 'label': grasp_info['stable']}
                memory.store(transition)

            print(action)
            print(grasp_info)
            print('---------')

            if policy.terminal(obs, next_obs):
                break

            obs = copy.deepcopy(next_obs)


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--n_samples', default=10000, type=int, help='')
    parser.add_argument('--seed', default=1, type=int, help='')
    parser.add_argument('--singulation_condition', action='store_true', default=False, help='')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    collect_random_dataset(args)
