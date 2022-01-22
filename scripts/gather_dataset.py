import os
import shutil
import numpy as np
import yaml
import copy

from ppg.environment import Environment
from ppg.agent import PushGrasping


def gather_dataset(out_dir, n_samples, seed=0):
    # Create the folder out_dir to save the dataset
    if os.path.exists(out_dir):
        answer = input('The folder ' + out_dir + ' exists. Do you want to remove it permanently? (y/n)')
        if answer:
            shutil.rmtree(out_dir)
        else:
            exit()
    os.mkdir(out_dir)

    # Load params
    with open('../yaml/params_clutter.yml', 'r') as stream:
        params = yaml.safe_load(stream)

    bounds = np.array([[-0.25, 0.25], [-0.25, 0.25], [0.01, 0.3]])
    pix_size = 0.005
    env = Environment(assets_root='../assets/', workspace_pos=params['env']['workspace']['pos'])
    policy = PushGrasping(params)
    policy.seed(seed)

    rng = np.random.RandomState()
    rng.seed(seed)

    i = 0
    while True:
        episode_seed = rng.randint(0, pow(2, 32) - 1)
        env.seed(episode_seed)
        obs = env.reset()

        while not policy.init_state_is_valid(obs):
            obs = env.reset()

        while True:

            # Compute state
            state = utils.get_fused_heightmap(obs, cameras.RealSense.CONFIG, bounds, pix_size)
            state_id = env.save_state()

            action = policy.random_sample(state)
            if action is None:
                break

            # Perform grasp with every opening and keep the best one
            grasp_metrics = []
            next_observations = []
            next_state_ids = []
            for j in range(len(policy.widths)):
                action[3] = policy.widths[j]
                env_action = policy.action(action, bounds, pix_size)

                next_obs, grasp, is_in_contact = env.step(env_action)
                # If the first attempted grasp collides with the objects, sample a new grasp
                if is_in_contact and j==0:
                    break

                if grasp['stable']:
                    grasp_metrics.append(grasp['num_contacts'])
                else:
                    grasp_metrics.append(0)

                # Save next states
                next_state_ids.append(env.save_state())
                next_observations.append(copy.deepcopy(next_obs))

                # Restore prev state and execute the same grasp with different opening
                env.restore_state(state_id)

            if len(grasp_metrics) == 0:
                continue

            # If the best grasp has 0 contacts is failed, continue
            best_grasp_id = np.argmax(grasp_metrics)
            if grasp_metrics[best_grasp_id] == 0.0:
                continue

            # Continue from the state produced from the best grasp
            env.restore_state(next_state_ids[best_grasp_id])
            next_obs = next_observations[best_grasp_id]

            # Save transition
            i += 1
            folder_name = os.path.join(out_dir, 'transition_' + str(i).zfill(5))
            if os.path.exists(folder_name):
                raise Exception
            os.mkdir(folder_name)

            pickle.dump(obs['full_state'], open(os.path.join(folder_name, 'full_state'), 'wb'))

            cv2.imwrite(os.path.join(folder_name, 'heightmap.exr'), state)

            os.mkdir(os.path.join(folder_name, 'rgb'))
            os.mkdir(os.path.join(folder_name, 'depth'))
            os.mkdir(os.path.join(folder_name, 'seg'))
            for j in range(len(obs['color'])):
                cv2.imwrite(os.path.join(folder_name, 'rgb/rgb_' + str(j) + '.jpg'), cv2.cvtColor(obs['color'][j],
                                                                                              cv2.COLOR_BGR2RGB))
                cv2.imwrite(os.path.join(folder_name, 'depth/depth_' + str(j) + '.exr'), obs['depth'][j])
                cv2.imwrite(os.path.join(folder_name, 'seg/seg' + str(j) + '.jpg'), obs['seg'][j])

            # Save the opening with the max contacts
            opening_max_contacts = policy.widths[np.argmax(grasp_metrics)]

            # Save the smallest successful opening
            valid_grasps_ids = np.argwhere(np.array(grasp_metrics) > 0)
            opening_min_spread = policy.widths[max(valid_grasps_ids)[0]]

            action_dict = {'pxl': np.array([action[0], action[1]]),
                           'theta': action[2],
                           'opening': {'max_contacts':opening_max_contacts, 'min_width':opening_min_spread}}
            pickle.dump(action_dict, open(os.path.join(folder_name, 'action'), 'wb'))

            if policy.terminal(obs, next_obs):
                break

            obs = copy.deepcopy(next_obs)