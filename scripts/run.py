import pickle

from ppg.train import train_fcn_net, train_classifier

import yaml
import numpy as np
import copy
import os
import shutil
from tabulate import tabulate

from ppg.environment import Environment
from ppg.agent import PushGrasping, HeuristicPushGrasping, PushGrasping2
from ppg.utils import utils
from ppg import cameras


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

    pxl_size = 0.005
    bounds = np.array([[-0.25, 0.25], [-0.25, 0.25], [0.01, 0.3]])
    env = Environment(assets_root='../assets/', workspace_pos=[0.0, 0.0, 0.0])
    # policy = PushGrasping(params)
    # policy.load(weights_fcn='../logs/fcn_model/model_15.pt',
    #             weights_cls='../logs/classifier/model_5.pt')
    # policy = HeuristicPushGrasping(params)
    # policy.seed(seed)

    policy = PushGrasping2(params)
    policy.seed(seed)

    rng = np.random.RandomState()
    rng.seed(seed)

    eval_data = []
    for i in range(n_scenes):

        episode_seed = rng.randint(0, pow(2, 32) - 1)
        env.seed(episode_seed)

        print('Episode seed:', episode_seed)

        obs = env.reset()
        while not policy.init_state_is_valid(obs):
            obs = env.reset()

        episode_data = {'successes': 0,
                        'fails': 0,
                        'attempts': 0,
                        'objects_removed': 0,
                        'objects_in_scene': len(obs['full_state'])}
        while True:
            state = utils.get_fused_heightmap(obs, cameras.RealSense.CONFIG, bounds, pxl_size)
            if len(np.argwhere(state > 0.1)) == 0:
                break

            action = policy.predict(state)
            if action is None:
                break
                # TODO: why none?

            print(action)
            env_action = policy.action(action, bounds, pxl_size)

            next_obs, grasp_info = env.step(env_action)

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

    eval(n_scenes=100, log_path='../logs/tmp', seed=1)

    # analyze(log_dir='../logs/eval_heuristic_policy')
