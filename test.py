import pickle
import yaml
import numpy as np
import argparse
import copy

from ppg.environment import Environment
from ppg.agent import PushGrasping


def run_episode(policy, env, episode_seed, max_steps=15, train=True):
    env.seed(episode_seed)
    obs = env.reset()

    while not policy.init_state_is_valid(obs):
        obs = env.reset()

    episode_data = {'sr-1': 0,
                    'sr-n': 0,
                    'fails': 0,
                    'attempts': 0,
                    'collisions': 0,
                    'objects_removed': 0,
                    'objects_in_scene': len(obs['full_state'])}

    i = 0
    while episode_data['attempts'] < max_steps:
        # print('---Step:', i)
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
            continue

        if grasp_info['stable'] and i == 0:
            episode_data['sr-1'] += 1

        episode_data['attempts'] += 1
        if grasp_info['stable']:
            episode_data['sr-n'] += 1
            episode_data['objects_removed'] += 1
        else:
            episode_data['fails'] += 1

        if policy.terminal(obs, next_obs):
            break

        obs = copy.deepcopy(next_obs)

        i += 1

    print('--------')
    return episode_data


def eval_agent(args):
    with open('yaml/bhand.yml', 'r') as stream:
        params = yaml.safe_load(stream)

    env = Environment(assets_root='assets/', objects_set=args.object_set)

    policy = PushGrasping(params)
    policy.load(fcn_model=args.fcn_model, reg_model=args.reg_model)

    rng = np.random.RandomState()
    rng.seed(args.seed)

    eval_data = []
    sr_n = 0
    sr_1 = 0
    attempts = 0
    objects_removed = 0
    for i in range(args.n_scenes):
        episode_seed = rng.randint(0, pow(2, 32) - 1)
        print('Episode:{}, seed:{}'.format(i, episode_seed))
        episode_data = run_episode(policy, env, episode_seed, train=False)
        eval_data.append(episode_data)

        sr_1 += episode_data['sr-1']
        sr_n += episode_data['sr-n']
        attempts += episode_data['attempts']
        objects_removed += (episode_data['objects_removed'] + 1) / float(episode_data['objects_in_scene'])
    print('SR-1:{}, SR-N: {}, Scene Clearance: {}'.format(sr_1 / args.n_scenes,
                                                          sr_n / attempts,
                                                          objects_removed / len(eval_data)))


def eval_challenging(args):
    pass


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--fcn_model', default='', type=str, help='')
    parser.add_argument('--reg_model', default='', type=str, help='')
    parser.add_argument('--seed', default=0, type=int, help='')
    parser.add_argument('--n_scenes', default=100, type=int, help='')
    parser.add_argument('--object_set', default='seen', type=str, help='')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.object_set == 'challenging':
        eval_challenging(args)
    else:
        eval_agent(args)
