import pickle
import yaml
import numpy as np
import copy
import os
import shutil
from tabulate import tabulate
import matplotlib.pyplot as plt

from ppg.environment import Environment
from ppg.agent import PushGrasping, HeuristicPushGrasping, PushGrasping2
from ppg.train import train_fcn_net, train_classifier, eval_aperture_net
from ppg.utils.utils import Logger


def run_episode(policy, env, episode_seed, max_steps=15, train=True):
    env.seed(episode_seed)
    obs = env.reset()
    print('Episode seed:', episode_seed)

    while not policy.init_state_is_valid(obs):
        obs = env.reset()

    episode_data = {'successes': 0,
                    'fails': 0,
                    'attempts': 0,
                    'collisions': 0,
                    'objects_removed': 0,
                    'objects_in_scene': len(obs['full_state'])}
    for i in range(max_steps):
        print('---Step:', i)
        state = policy.state_representation(obs)

        # Select action
        if train:
            action = policy.explore(state)
        else:
            action = policy.predict(state)

        env_action = policy.action(action)

        # Step environment.
        next_obs, grasp_info = env.step(env_action)

        if grasp_info['collision']:
            episode_data['collisions'] += 1

        episode_data['attempts'] += 1
        if grasp_info['stable']:
            episode_data['successes'] += 1
            episode_data['objects_removed'] += 1
        else:
            episode_data['fails'] += 1

        if train:
            transition = {'state': state, 'action': action, 'label': grasp_info['stable']}
            policy.learn(transition)

        print(grasp_info)

        # if policy.terminal(obs, next_obs) or ((not grasp_info['stable']) and grasp_info['num_contacts'] > 0):
        if policy.terminal(obs, next_obs):
            break

        obs = copy.deepcopy(next_obs)

    print('--------')
    return episode_data


def train_agent(log_path, n_scenes, save_every=100, seed=0):
    with open('../yaml/bhand.yml', 'r') as stream:
        params = yaml.safe_load(stream)
    params['log_dir'] = log_path

    logger = Logger(log_path)

    env = Environment(assets_root='../assets/', workspace_pos=[0.0, 0.0, 0.0])

    policy = PushGrasping(params)
    policy.seed(seed)

    rng = np.random.RandomState()
    rng.seed(seed)

    train_data = []
    for i in range(n_scenes):
        episode_seed = rng.randint(0, pow(2, 32) - 1)
        episode_data = run_episode(policy, env, episode_seed, train=True)
        train_data.append(episode_data)

        logger.log_data(train_data, 'train_data')

        if i % save_every == 0:
            policy.save(epoch=i)


def eval_agent(n_scenes, log_path, seed=0):
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

    env = Environment(assets_root='../assets/', workspace_pos=[0.0, 0.0, 0.0])
    # policy = PushGrasping(params)
    # policy.load(weights_fcn='../logs/fcn_model/model_15.pt',
    #             weights_cls='../logs/classifier/model_5.pt')
    # policy = HeuristicPushGrasping(params)
    # policy.seed(seed)

    policy = PushGrasping(params)
    # policy.load('../logs/supervised')
    policy.load_seperately(fcn_model='../logs/fcn_model/model_5.pt',
                           reg_model='../logs/regressor/model_10.pt')
    # policy.seed(seed)

    rng = np.random.RandomState()
    rng.seed(seed)

    eval_data = []
    success_rate = 0
    attempts = 0
    objects_removed = 0
    for i in range(n_scenes):
        print('Episode ', i)
        episode_seed = rng.randint(0, pow(2, 32) - 1)
        episode_data = run_episode(policy, env, episode_seed, train=False)
        eval_data.append(episode_data)

        success_rate += episode_data['successes']
        attempts += episode_data['attempts'] - episode_data['collisions']
        objects_removed += episode_data['objects_removed'] / float(episode_data['objects_in_scene'] - 1)
        print('Success_rate: {}, Scene Clearance: {}'.format(success_rate / attempts, objects_removed / len(eval_data)))

    pickle.dump(eval_data, open(os.path.join(log_path, 'eval_data'), 'wb'))


def collect_random_dataset(n_scenes, log_path, seed=1):
    with open('../yaml/bhand.yml', 'r') as stream:
        params = yaml.safe_load(stream)
    params['log_dir'] = log_path
    logger = Logger(log_path)

    env = Environment(assets_root='../assets/', workspace_pos=[0.0, 0.0, 0.0])

    policy = PushGrasping(params)
    policy.seed(seed)

    rng = np.random.RandomState()
    rng.seed(seed)

    for j in range(n_scenes):
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
                transition = {'state': state, 'action': action, 'label': grasp_info['stable']}
                policy.replay_buffer.store(transition)

            print(action)
            print(grasp_info)
            print('---------')

            if policy.terminal(obs, next_obs):
                break

            obs = copy.deepcopy(next_obs)


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

    # params = {'dataset_dir': '../logs/train_self_supervised_per/replay_buffer',
    #           'split_ratio': 0.9,
    #           'epochs': 100,
    #           'batch_size': 1,
    #           'learning_rate': 0.0001,
    #           'log_path': '../logs/fcn_model'}
    # train_fcn_net(params)
    #
    # params = {'dataset_dir': '../logs/self_supervised_large_dist/replay_buffer',
    #           'split_ratio': 0.9,
    #           'epochs': 100,
    #           'batch_size': 4,
    #           'learning_rate': 0.0001,
    #           'log_path': '../logs/regressor'}
    # train_classifier(params)

    # train_agent(log_path='../logs/self-supervised', n_scenes=10000, seed=0)

    # eval(n_scenes=100, log_path='../logs/eval_supervised', seed=1)

    # analyze(log_dir='../logs/eval_heuristic_policy')

    # analyze_replay_buffer('../logs/train_self_supervised/replay_buffer')

    # eval_aperture_net(params)

    collect_random_dataset(n_scenes=10000, log_path='../logs/random-dataset-2', seed=2)
    