import pickle

from ppg.train import train_fcn_net, train_classifier

import yaml
import numpy as np
import copy
import os
import shutil
from tabulate import tabulate
import matplotlib.pyplot as plt

from ppg.environment import Environment
from ppg.agent import PushGrasping, HeuristicPushGrasping, PushGrasping2
from ppg.utils import utils
from ppg import cameras


def run_episode(policy, env, train=True):
    obs = env.reset()

    while not policy.init_state_is_valid(obs):
        obs = env.reset()

    episode_data = {'successes': 0,
                    'fails': 0,
                    'attempts': 0,
                    'collisions': 0,
                    'objects_removed': 0,
                    'objects_in_scene': len(obs['full_state'])}
    while True:
        state = policy.state_representation(obs)

        # Select action
        if train:
            action = policy.explore(state)
        else:
            action = policy.predict(state)

        env_action = policy.action(action)

        # Step environment.
        next_obs, grasp_info = env.step(env_action)

        # Update logger
        episode_data['attempts'] += 1
        if grasp_info['stable']:
            episode_data['successes'] += 1
            episode_data['objects_removed'] += 1
        else:
            episode_data['fails'] += 1
            if grasp_info['collision']:
                episode_data['collisions'] += 1

        if train:
            transition = {'state': state, 'action': action, 'label': grasp_info['stable']}
            policy.learn(transition)

        if policy.terminal(obs, next_obs):
            break

        obs = copy.deepcopy(next_obs)

    return episode_data


def train_agent(log_path, n_scenes, save_every=100, seed=0):
    with open('../yaml/bhand.yml', 'r') as stream:
        params = yaml.safe_load(stream)

    # Create a logger
    if os.path.exists(log_path):
        print('Directory ', log_path, 'exists, do you want to remove it? (y/n)')
        answer = input('')
        if answer == 'y':
            shutil.rmtree(log_path)
            os.mkdir(log_path)
        else:
            exit()
    else:
        os.mkdir(log_path)

    params['log_dir'] = log_path

    env = Environment(assets_root='../assets/', workspace_pos=[0.0, 0.0, 0.0])

    policy = PushGrasping(params)
    policy.seed(seed)

    rng = np.random.RandomState()
    rng.seed(seed)

    for i in range(n_scenes):
        episode_seed = rng.randint(0, pow(2, 32) - 1)
        env.seed(episode_seed)
        episode_data = run_episode(policy, env, train=True)
        if i % save_every == 0:
            policy.save(epoch=i)


def eval(n_scenes, log_path, seed=0):
    # Create a logger
    if os.path.exists(log_path):
        print('Directory ', log_path, 'exists, do you want to remove it? (y/n)')
        answer = input('')
        if answer == 'y':
            shutil.rmtree(log_path)
            os.mkdir(log_path)
        else:
            exit()
    else:
        os.mkdir(log_path)

    with open('../yaml/bhand.yml', 'r') as stream:
        params = yaml.safe_load(stream)

    params['log_dir'] = log_path

    pxl_size = 0.005
    bounds = np.array([[-0.25, 0.25], [-0.25, 0.25], [0.01, 0.3]])
    env = Environment(assets_root='../assets/', workspace_pos=[0.0, 0.0, 0.0])
    # policy = PushGrasping(params)
    # policy.load(weights_fcn='../logs/fcn_model/model_15.pt',
    #             weights_cls='../logs/classifier/model_5.pt')
    # policy = HeuristicPushGrasping(params)
    # policy.seed(seed)

    policy = PushGrasping(params)
    policy.load('../logs/train_self_supervised/model_200')
    # policy.seed(seed)

    rng = np.random.RandomState()
    rng.seed(seed)

    eval_data = []
    for i in range(n_scenes):

        episode_seed = rng.randint(0, pow(2, 32) - 1)
        env.seed(episode_seed)

        print('Episode {} seed{}:'.format(i, episode_seed))

        obs = env.reset()
        while not policy.init_state_is_valid(obs):
            obs = env.reset()

        episode_data = {'successes': 0,
                        'fails': 0,
                        'attempts': 0,
                        'objects_removed': 0,
                        'objects_in_scene': len(obs['full_state'])}
        while True:
            state = policy.state_representation(obs)
            if len(np.argwhere(state > 0.1)) == 0:
                break

            action = policy.predict(state)
            if action is None:
                break
                # TODO: why none?

            env_action = policy.action(action)

            next_obs, grasp_info = env.step(env_action)
            print('Success:', grasp_info['stable'])

            if grasp_info['collision']:
                continue

            episode_data['attempts'] += 1
            if grasp_info['stable']:
                episode_data['successes'] += 1
                episode_data['objects_removed'] += 1
            else:
                episode_data['fails'] += 1

            if policy.terminal(obs, next_obs):
                break

            obs = copy.deepcopy(next_obs)

        eval_data.append(episode_data)

    pickle.dump(eval_data, open(os.path.join(log_path, 'eval_data'), 'wb'))


def analyze(log_dir):
    eval_data = pickle.load(open(os.path.join(log_dir, 'eval_data'), 'rb'))

    success_rate = 0
    attempts = 0
    objects_removed = 0
    for episode_data in eval_data:
        success_rate += episode_data['successes']
        attempts += episode_data['attempts']
        objects_removed += episode_data['objects_removed'] / float(episode_data['objects_in_scene'])

    print('---------------------------------------------------------------------------------------')
    print(tabulate([['Heuristic', success_rate / len(eval_data),
                                  attempts / len(eval_data),
                                  objects_removed / len(eval_data)]],
                   headers=['Policy', 'Grasp success', 'Grasp attempts', 'Scene clearance']))
    print('---------------------------------------------------------------------------------------')


def analyze_replay_buffer(replay_buffer_dir):
    transition_dirs = next(os.walk(replay_buffer_dir))[1]

    apertures = []
    for transition_dir in transition_dirs:
        action = pickle.load(open(os.path.join(replay_buffer_dir, transition_dir, 'action'), 'rb'))
        apertures.append(action[3])

    bins = 5
    limits = [0.6, 1.1]
    step = (limits[1] - limits[0]) / bins
    discrete_apertures = np.zeros((bins, ))
    for i in range(bins):
        discrete_apertures[i] = i * step

    heights = np.zeros((bins, ))
    for aperture in apertures:
        # find the bin
        # print(aperture, step)
        bin_id = int((aperture - limits[0]) / step)
        # print(bin_id)
        heights[bin_id] += 1

    heights /= len(apertures)

    plt.bar(discrete_apertures, heights, width=0.5)
    plt.show()

if __name__ == "__main__":

    # params = {'dataset_dir': '../logs/dataset_no_flats',
    #           'split_ratio': 0.9,
    #           'epochs': 100,
    #           'batch_size': 1,
    #           'learning_rate': 0.0001,
    #           'log_path': '../logs/fcn_model'}
    # train_fcn_net(params)

    # params = {'dataset_dir': '../logs/dataset_no_flats',
    #           'split_ratio': 0.9,
    #           'epochs': 100,
    #           'batch_size': 1,
    #           'learning_rate': 0.0001,
    #           'log_path': '../logs/classifier'}
    # train_classifier(params)

    train_agent(log_path='../logs/train_self_supervised', n_scenes=10000, seed=0)

    # eval(n_scenes=100, log_path='../logs/eval_self_supervised', seed=1)

    # analyze(log_dir='../logs/eval_heuristic_policy')
