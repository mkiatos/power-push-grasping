import numpy as np
import pybullet as p
import matplotlib.pyplot as plt
import cv2
import random
import os
import pickle

import torch
import torch.optim as optim
import torch.nn as nn

from sklearn.cluster import DBSCAN
from sklearn.linear_model import LinearRegression

from ppg.utils import orientation as ori
from ppg.utils import utils
from ppg.utils.memory import ReplayBuffer, PrioritizedReplayBuffer
from ppg.models import ResFCN, Classifier, Regressor
from ppg import cameras


def compute_aperture(opening_in_cm):
    # 0.6 rad -> 0.2 m, 1.1 rad -> 0.09 m, 1.5 rad -> 0.01 m
    y = np.array([[0.6], [0.7], [0.8], [0.9], [1.0], [1.1], [1.2], [1.3], [1.4], [1.5]])
    x = np.array([[0.2], [0.179], [0.157], [0.135], [0.112], [0.09], [0.069], [0.048], [0.029], [0.01]])
    reg = LinearRegression().fit(x, y)
    data = np.array([opening_in_cm])
    return reg.predict(data.reshape(1, -1))


class Policy:
    def __init__(self, params):
        self.params = params
        self.rng = np.random.RandomState()

    def seed(self, seed):
        self.rng.seed(seed)

    def random_sample(self, state):
        pass

    def predict(self, state):
        pass

    def init_state_is_valid(self, obs):
        flat_objs = 0
        for obj in obs['full_state']:
            obj_pos, obj_quat = p.getBasePositionAndOrientation(obj.body_id)
            rot_mat = ori.Quaternion(x=obj_quat[0],
                                     y=obj_quat[1],
                                     z=obj_quat[2],
                                     w=obj_quat[3]).rotation_matrix()
            angle_z = np.arccos(np.dot(np.array([0, 0, 1]),
                                       rot_mat[0:3, 2]))
            if np.abs(angle_z) > 0.1:
                flat_objs += 1

        if flat_objs == len(obs['full_state']):
            return False
        else:
            return True

    def terminal(self, obs, next_obs):
        objects = next_obs['full_state']

        for obj in objects:
            obj_pos, obj_quat = p.getBasePositionAndOrientation(obj.body_id)
            if obj_pos[2] < 0:
                continue

            # Check if there is at least one object in the scene with the axis parallel to world z.
            rot_mat = ori.Quaternion(x=obj_quat[0], y=obj_quat[1], z=obj_quat[2], w=obj_quat[3]).rotation_matrix()
            angle_z = np.arccos(np.dot(np.array([0, 0, 1]), rot_mat[0:3, 2]))
            if np.abs(angle_z) < 0.1:
                return False

        return False


class PushGrasping2:
    def __init__(self, params):
        self.params = params

        self.rotations = params['rotations']
        self.aperture_limits = params['aperture_limits']
        self.aperture_limits = [0.6, 0.8, 1.1]
        self.crop_size = 32
        self.push_distance = 0.1

        self.fcn = ResFCN().to('cuda')
        self.classifier = Classifier(n_classes=3).to('cuda')

    def seed(self, seed):
        random.seed(seed)

    def pre_process(self, heightmap):
        """
        Pre-process heightmap (padding and normalization)
        """
        # Pad heightmap.
        diagonal_length = float(heightmap.shape[0]) * np.sqrt(2)
        diagonal_length = np.ceil(diagonal_length / 16) * 16
        self.padding_width = int((diagonal_length - heightmap.shape[0]) / 2)
        padded_heightmap = np.pad(heightmap, self.padding_width, 'constant', constant_values=-0.01)

        # Normalize maps ( ToDo: find mean and std)
        image_mean = 0.01
        image_std = 0.03
        padded_heightmap = (padded_heightmap - image_mean) / image_std

        # Add extra channel.
        padded_heightmap = np.expand_dims(padded_heightmap, axis=0)
        return padded_heightmap

    def post_process(self, q_maps):
        """
        Remove extra padding.
        """

        w = int(q_maps.shape[2] - 2 * self.padding_width)
        h = int(q_maps.shape[3] - 2 * self.padding_width)
        remove_pad = np.zeros((q_maps.shape[0], q_maps.shape[1], w, h))

        for i in range(q_maps.shape[0]):
            for j in range(q_maps.shape[1]):
                # remove extra padding
                q_map = q_maps[i, j, self.padding_width:int(q_maps.shape[2] - self.padding_width),
                               self.padding_width:int(q_maps.shape[3] - self.padding_width)]

                remove_pad[i][j] = q_map.detach().cpu().numpy()

        return remove_pad

    def pre_process_aperture_img(self, heightmap, p1, theta, plot=True):
        """
        Add extra padding, rotate image so as the push always points to the right, crop around the initial push
        position (something like attention) and finally normalize the cropped image.
        """
        # Add extra padding (to handle rotations inside network)
        diag_length = float(heightmap.shape[0]) * np.sqrt(2)
        diag_length = np.ceil(diag_length / 16) * 16
        padding_width = int((diag_length - heightmap.shape[0]) / 2)
        depth_heightmap = np.pad(heightmap, padding_width, 'constant')
        padded_shape = depth_heightmap.shape

        p1 += padding_width
        action_theta = -(theta + (2 * np.pi))

        # Rotate image (push always on the right)
        rot = cv2.getRotationMatrix2D((int(padded_shape[0] / 2), int(padded_shape[1] / 2)),
                                      action_theta * 180 / np.pi, 1.0)
        rotated_heightmap = cv2.warpAffine(depth_heightmap, rot, (padded_shape[0], padded_shape[1]))

        # Compute the position of p1 on rotated heightmap
        rotated_pt = np.dot(rot, (p1[0], p1[1], 1.0))
        rotated_pt = (int(rotated_pt[0]), int(rotated_pt[1]))

        # Crop heightmap
        cropped_map = np.zeros((2 * self.crop_size, 2 * self.crop_size), dtype=np.float32)
        y_start = max(0, rotated_pt[1] - self.crop_size)
        y_end = min(padded_shape[0], rotated_pt[1] + self.crop_size)
        x_start = rotated_pt[0]
        x_end = min(padded_shape[0], rotated_pt[0] + 2 * self.crop_size)
        cropped_map[0:y_end - y_start, 0:x_end - x_start] = rotated_heightmap[y_start: y_end, x_start: x_end]

        # print( action['opening']['min_width'])
        if plot:
            p2 = np.array([0, 0])
            p2[0] = p1[0] + 20 * np.cos(theta)
            p2[1] = p1[1] - 20 * np.sin(theta)

            fig, ax = plt.subplots(1, 3)
            ax[0].imshow(depth_heightmap)
            ax[0].plot(p1[0], p1[1], 'o', 2)
            ax[0].plot(p2[0], p2[1], 'x', 2)
            ax[0].arrow(p1[0], p1[1], p2[0] - p1[0], p2[1] - p1[1], width=1)

            rotated_p2 = np.array([0, 0])
            rotated_p2[0] = rotated_pt[0] + 20 * np.cos(0)
            rotated_p2[1] = rotated_pt[1] - 20 * np.sin(0)
            ax[1].imshow(rotated_heightmap)
            ax[1].plot(rotated_pt[0], rotated_pt[1], 'o', 2)
            ax[1].plot(rotated_p2[0], rotated_p2[1], 'x', 2)
            ax[1].arrow(rotated_pt[0], rotated_pt[1], rotated_p2[0] - rotated_pt[0], rotated_p2[1] - rotated_pt[1],
                        width=1)

            ax[2].imshow(cropped_map)
            plt.show()

        # Normalize maps ( ToDo: find mean and std) # Todo
        image_mean = 0.01
        image_std = 0.03
        cropped_map = (cropped_map - image_mean) / image_std
        cropped_map = np.expand_dims(cropped_map, axis=0)

        p1 -= padding_width
        return cropped_map

    def random_sample(self, state):

        threshold_height = 0.1

        # Sample position.
        obj_ids = np.argwhere(state > threshold_height)
        if len(obj_ids) == 0:
            return None

        valid_pxl_map = np.zeros(state.shape)
        for x in range(state.shape[0]):
            for y in range(state.shape[1]):
                if state[y, x] > threshold_height:
                    continue

                dists = np.linalg.norm(np.array([y, x]) - obj_ids, axis=1)
                if np.min(dists) > 30 or np.min(dists) < 10:  # Todo fix the hardcoded values
                    continue
                valid_pxl_map[y, x] = 255

        if (valid_pxl_map == 0).all():
            return None

        # mask = np.zeros(state.shape)
        # mask[state > threshold_height] = 127
        # plt.imshow(valid_pxl_map + mask)
        # plt.show()

        valid_pxls = np.argwhere(valid_pxl_map == 255)
        valid_ids = np.arange(0, valid_pxls.shape[0])
        pxl = valid_pxls[random.choice(valid_ids)]
        p1 = np.array([pxl[1], pxl[0]])

        # Sample pushing direction. Push directions point always towards the objects.
        objects_mask = np.zeros(state.shape)
        objects_mask[state > threshold_height] = 255

        # Compute contours
        _, thresh = cv2.threshold(objects_mask.astype(np.uint8), 127, 255, 0)
        contours, _ = cv2.findContours(thresh,
                                       cv2.RETR_TREE,
                                       cv2.CHAIN_APPROX_SIMPLE)

        # Keep only contour points that are around the sample pixel position.
        pushing_area = 30
        points = []
        for cnt in contours:
            for pnt in cnt:
                if (p1[0] - pushing_area < pnt[0, 0] < p1[0] + pushing_area) and \
                        (p1[1] - pushing_area < pnt[0, 1] < p1[1] + pushing_area):
                    points.append(pnt[0])

        if len(points) == 0:
            return None

        p2 = random.choice(points)
        push_dir = p2 - p1
        theta = -np.arctan2(push_dir[1], push_dir[0])

        # Discretize theta.
        step_angle = 2 * np.pi / self.rotations
        discrete_theta = round(theta / step_angle) * step_angle

        # Sample gripper opening.
        width = np.random.choice(self.widths)

        # plt.imshow(state)
        # plt.plot(p1[0], p1[1], 'o', 2)
        # plt.plot(p2[0], p2[1], 'x', 2)
        # plt.arrow(p1[0], p1[1], p2[0]-p1[0], p2[1]-p1[1], width=1)
        # plt.show()
        #
        return [pxl[1], pxl[0], discrete_theta, 0.6]

    def predict(self, heightmap):
        input_heightmap = self.pre_process(heightmap)

        # Find optimal position and orientation
        x = torch.FloatTensor(input_heightmap).unsqueeze(0).to('cuda')
        out_prob = self.fcn(x, is_volatile=True)
        out_prob = self.post_process(out_prob)

        best_action = np.unravel_index(np.argmax(out_prob), out_prob.shape)
        p1 = np.array([best_action[3], best_action[2]])
        theta = best_action[0] * 2 * np.pi / self.rotations

        # p2 = np.array([0, 0])
        # p2[0] = p1[0] + 20 * np.cos(theta)
        # p2[1] = p1[1] - 20 * np.sin(theta)
        #
        # plt.imshow(heightmap)
        # plt.plot(p1[0], p1[1], 'o', 2)
        # plt.plot(p2[0], p2[1], 'x', 2)
        # plt.arrow(p1[0], p1[1], p2[0] - p1[0], p2[1] - p1[1], width=1)
        # plt.show()

        # Find optimal aperture
        aperture_img = self.pre_process_aperture_img(heightmap, p1, theta)
        x = torch.FloatTensor(aperture_img).unsqueeze(0).to('cuda')
        pred = self.classifier(x)
        max_id = torch.argmax(pred).detach().cpu().numpy()

        return p1[0], p1[1], theta, self.aperture_limits[max_id]

    def action(self, action, bounds, pxl_size):
        # Convert from pixels to 3d coordinates.
        x = -(pxl_size * action[0] - bounds[0][1])
        y = pxl_size * action[1] - bounds[1][1]
        quat = ori.Quaternion.from_rotation_matrix(np.matmul(ori.rot_y(-np.pi / 2), ori.rot_x(action[2])))

        return {'pos': np.array([x, y, 0.1]), 'quat': quat, 'aperture': action[3], 'push_distance': self.push_distance}

    def load(self, weights_fcn, weights_cls):
        self.fcn.load_state_dict(torch.load(weights_fcn))
        self.classifier.load_state_dict(torch.load(weights_cls))

    def init_state_is_valid(self, obs):
        flat_objs = 0
        for obj in obs['full_state']:
            obj_pos, obj_quat = p.getBasePositionAndOrientation(obj.body_id)
            rot_mat = ori.Quaternion(x=obj_quat[0],
                                     y=obj_quat[1],
                                     z=obj_quat[2],
                                     w=obj_quat[3]).rotation_matrix()
            angle_z = np.arccos(np.dot(np.array([0, 0, 1]),
                                       rot_mat[0:3, 2]))
            if np.abs(angle_z) > 0.1:
                flat_objs += 1

        if flat_objs == len(obs['full_state']):
            return False
        else:
            return True

    def terminal(self, obs, next_obs):
        objects = next_obs['full_state']

        is_terminal = True
        for obj in objects:
            obj_pos, obj_quat = p.getBasePositionAndOrientation(obj.body_id)
            if obj_pos[2] < 0:
                continue

            # Check if there is at least one object in the scene with the axis parallel to world z.
            rot_mat = ori.Quaternion(x=obj_quat[0], y=obj_quat[1], z=obj_quat[2], w=obj_quat[3]).rotation_matrix()
            angle_z = np.arccos(np.dot(np.array([0, 0, 1]), rot_mat[0:3, 2]))
            if np.abs(angle_z) < 0.1:
                is_terminal = False

        return is_terminal


class HeuristicPushGrasping(PushGrasping2):
    def __init__(self, params):
        super(HeuristicPushGrasping, self).__init__(params)
        self.params = params
        self.robot = []

    def seed(self, seed):
        random.seed(seed)

    def clustering(self, heigtmap, plot=False):

        ids = np.argwhere(heigtmap > 0)
        db = DBSCAN(eps=2, min_samples=10).fit(ids)

        seg = np.zeros(heigtmap.shape)
        for i in range(len(ids)):
            seg[ids[i, 0], ids[i, 1]] = db.labels_[i] + 1

        if plot:
            fig, ax = plt.subplots(1, 2)
            ax[0].imshow(heigtmap)
            ax[1].imshow(seg)
            plt.show()

        return seg

    def predict(self, heightmap):
        # Perform clustering.
        seg = self.clustering(heightmap)

        # Find the smallest cluster.
        obj_ids = np.unique(seg)
        len_clusters = np.zeros(len(obj_ids),)
        for i in range(len(obj_ids)):
            len_clusters[i] = len(np.argwhere(seg == obj_ids[i]))
        target_ids = np.argwhere(seg == np.argmin(len_clusters))

        # Compute opening
        y_min = np.min(target_ids[:, 0])
        y_max = np.max(target_ids[:, 0])
        x_min = np.min(target_ids[:, 1])
        x_max = np.max(target_ids[:, 1])
        largest_side = max(y_max - y_min, x_max - x_min)
        aperture = compute_aperture(largest_side * 0.005) - 0.05

        # Compute the valid pxls
        valid_pxl_map = np.zeros(heightmap.shape)
        for x in range(heightmap.shape[0]):
            for y in range(heightmap.shape[1]):
                dists = np.linalg.norm(np.array([y, x]) - target_ids, axis=1)
                if np.min(dists) > 30 or np.min(dists) < 10:  # Todo fix the hardcoded values
                    continue
                valid_pxl_map[y, x] = 255

        if (valid_pxl_map == 0).all():
            return None

        # Remove pxls that belongs to other objects.
        obstacle_maps = np.ones(heightmap.shape)
        obstacle_maps[heightmap > 0] = 0.0
        valid_pxl_map *= obstacle_maps

        # Pick a collision free path (ToDo: now we sample an action and check in simulation if it's collision-free)

        # Compute initial position.
        valid_pxls = np.argwhere(valid_pxl_map == 255)
        valid_ids = np.arange(0, valid_pxls.shape[0])
        pxl = valid_pxls[random.choice(valid_ids)]

        target_centroid = np.mean(target_ids, axis=0)

        # Compute direction.
        p1 = np.array([pxl[1], pxl[0]])
        p2 = np.array([target_centroid[1], target_centroid[0]])
        push_dir = p2 - p1
        theta = -np.arctan2(push_dir[1], push_dir[0])

        # Compute distance
        self.dist = np.linalg.norm(push_dir) * 0.005
        # print(self.dist)

        target_mask = np.zeros(heightmap.shape)
        target_mask[seg == np.argmin(len_clusters)] = 122
        # plt.imshow(valid_pxl_map + target_mask)
        # plt.plot(pxl[1], pxl[0], 'o')
        # plt.arrow(p1[0], p1[1], p2[0]-p1[0], p2[1]-p1[1], width=1)
        # plt.show()

        return [pxl[1], pxl[0], theta, aperture]

    def action(self, action, bounds, pxl_size):
        env_action = super(HeuristicPushGrasping, self).action(action, bounds, pxl_size)
        env_action['push_distance'] = self.dist
        return env_action


class PushGrasping(Policy):
    def __init__(self, params):
        super(PushGrasping, self).__init__(params)

        self.rotations = params['agent']['fcn']['rotations']
        self.aperture_limits = params['agent']['regressor']['aperture_limits']
        self.pxl_size = params['env']['pixel_size']
        self.bounds = np.array(params['env']['workspace']['bounds'])

        self.crop_size = 32
        self.push_distance = 0.15
        self.z = 0.1

        self.fcn = ResFCN().to('cuda')
        self.fcn_optimizer = optim.Adam(self.fcn.parameters(), lr=params['agent']['fcn']['learning_rate'])
        self.fcn_criterion = nn.BCELoss(reduction='none')
        # self.fcn_criterion = nn.SmoothL1Loss(reduction='none')

        self.reg = Regressor().to('cuda')
        self.reg_optimizer = optim.Adam(self.reg.parameters(), lr=params['agent']['regressor']['learning_rate'])
        self.reg_criterion = nn.L1Loss()

        if self.params['agent']['per']:
            self.replay_buffer = PrioritizedReplayBuffer(log_dir=params['log_dir'])
        else:
            self.replay_buffer = ReplayBuffer(log_dir=params['log_dir'])

        self.learn_step_counter = 0
        self.info = {'fcn_loss': [], 'reg_loss': [], 'learn_step_counter': 0}

        os.mkdir(os.path.join(params['log_dir'], 'maps'))
        np.set_printoptions(formatter={'float': lambda x: "{0:0.2f}".format(x)})

    def state_representation(self, obs):
        state = utils.get_fused_heightmap(obs, cameras.RealSense.CONFIG, self.bounds, self.pxl_size)
        return state

    def pre_process(self, state):
        """
        Pre-process heightmap (padding and normalization)
        """
        # Pad heightmap.
        diagonal_length = float(state.shape[0]) * np.sqrt(2)
        diagonal_length = np.ceil(diagonal_length / 16) * 16
        self.padding_width = int((diagonal_length - state.shape[0]) / 2)
        padded_heightmap = np.pad(state, self.padding_width, 'constant', constant_values=-0.01)

        # Normalize heightmap.
        image_mean = 0.01
        image_std = 0.03
        padded_heightmap = (padded_heightmap - image_mean) / image_std

        # Add extra channel.
        padded_heightmap = np.expand_dims(padded_heightmap, axis=0)
        return padded_heightmap

    def post_process(self, q_maps):
        """
        Remove extra padding.
        """

        w = int(q_maps.shape[2] - 2 * self.padding_width)
        h = int(q_maps.shape[3] - 2 * self.padding_width)
        remove_pad = np.zeros((q_maps.shape[0], q_maps.shape[1], w, h))

        for i in range(q_maps.shape[0]):
            for j in range(q_maps.shape[1]):
                # remove extra padding
                q_map = q_maps[i, j, self.padding_width:int(q_maps.shape[2] - self.padding_width),
                               self.padding_width:int(q_maps.shape[3] - self.padding_width)]

                remove_pad[i][j] = q_map.detach().cpu().numpy()

        return remove_pad

    def pre_process_aperture_img(self, state, p1, theta, plot=False):
        """
        Add extra padding, rotate image so as the push always points to the right, crop around the initial push
        position (something like attention) and finally normalize the cropped image.
        """
        # Add extra padding (to handle rotations inside network)
        diag_length = float(state.shape[0]) * np.sqrt(2)
        diag_length = np.ceil(diag_length / 16) * 16
        padding_width = int((diag_length - state.shape[0]) / 2)
        depth_heightmap = np.pad(state, padding_width, 'constant')
        padded_shape = depth_heightmap.shape

        p1 += padding_width
        action_theta = -((theta + (2 * np.pi)) % (2 * np.pi))

        # Rotate image (push always on the right)
        rot = cv2.getRotationMatrix2D((int(padded_shape[0] / 2), int(padded_shape[1] / 2)),
                                      action_theta * 180 / np.pi, 1.0)
        rotated_heightmap = cv2.warpAffine(depth_heightmap, rot, (padded_shape[0], padded_shape[1]))

        # Compute the position of p1 on rotated heightmap
        rotated_pt = np.dot(rot, (p1[0], p1[1], 1.0))
        rotated_pt = (int(rotated_pt[0]), int(rotated_pt[1]))

        # Crop heightmap
        cropped_map = np.zeros((2 * self.crop_size, 2 * self.crop_size), dtype=np.float32)
        y_start = max(0, rotated_pt[1] - self.crop_size)
        y_end = min(padded_shape[0], rotated_pt[1] + self.crop_size)
        x_start = rotated_pt[0]
        x_end = min(padded_shape[0], rotated_pt[0] + 2 * self.crop_size)
        cropped_map[0:y_end - y_start, 0:x_end - x_start] = rotated_heightmap[y_start: y_end, x_start: x_end]

        # print( action['opening']['min_width'])
        if plot:
            p2 = np.array([0, 0])
            p2[0] = p1[0] + 20 * np.cos(theta)
            p2[1] = p1[1] - 20 * np.sin(theta)

            fig, ax = plt.subplots(1, 3)
            ax[0].imshow(depth_heightmap)
            ax[0].plot(p1[0], p1[1], 'o', 2)
            ax[0].plot(p2[0], p2[1], 'x', 2)
            ax[0].arrow(p1[0], p1[1], p2[0] - p1[0], p2[1] - p1[1], width=1)

            rotated_p2 = np.array([0, 0])
            rotated_p2[0] = rotated_pt[0] + 20 * np.cos(0)
            rotated_p2[1] = rotated_pt[1] - 20 * np.sin(0)
            ax[1].imshow(rotated_heightmap)
            ax[1].plot(rotated_pt[0], rotated_pt[1], 'o', 2)
            ax[1].plot(rotated_p2[0], rotated_p2[1], 'x', 2)
            ax[1].arrow(rotated_pt[0], rotated_pt[1], rotated_p2[0] - rotated_pt[0], rotated_p2[1] - rotated_pt[1],
                        width=1)

            ax[2].imshow(cropped_map)
            plt.show()

        # Normalize maps ( ToDo: find mean and std) # Todo
        image_mean = 0.01
        image_std = 0.03
        cropped_map = (cropped_map - image_mean) / image_std
        cropped_map = np.expand_dims(cropped_map, axis=0)

        three_channel_img = np.zeros((3, cropped_map.shape[1], cropped_map.shape[2]))
        three_channel_img[0] = cropped_map
        three_channel_img[1] = cropped_map
        three_channel_img[2] = cropped_map

        p1 -= padding_width
        return three_channel_img

    def random_sample(self, state):
        action = np.zeros((4,))
        action[0] = self.rng.randint(0, state.shape[0])
        action[1] = self.rng.randint(0, state.shape[1])
        action[2] = self.rng.randint(0, 16) * 2 * np.pi / self.rotations
        action[3] = self.rng.uniform(self.aperture_limits[0], self.aperture_limits[1])
        return action

    def guided_exploration(self, state, sample_limits=[0.1, 0.15]):
        obj_ids = np.argwhere(state > self.z)

        # Sample initial position.
        valid_pxl_map = np.zeros(state.shape)
        for x in range(state.shape[0]):
            for y in range(state.shape[1]):
                dists = np.linalg.norm(np.array([y, x]) - obj_ids, axis=1)
                if sample_limits[0] / self.pxl_size < np.min(dists) < sample_limits[1] / self.pxl_size:
                    valid_pxl_map[y, x] = 255

        # fig, ax = plt.subplots(1, 2)
        # ax[0].imshow(state)
        # ax[1].imshow(valid_pxl_map)
        # plt.show()

        valid_pxls = np.argwhere(valid_pxl_map == 255)
        valid_ids = np.arange(0, valid_pxls.shape[0])

        objects_mask = np.zeros(state.shape)
        objects_mask[state > self.z] = 255
        _, thresh = cv2.threshold(objects_mask.astype(np.uint8), 127, 255, 0)
        contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        while True:
            pxl = valid_pxls[self.rng.choice(valid_ids, 1)[0]]
            p1 = np.array([pxl[1], pxl[0]])

            # Sample pushing direction. Push directions point always towards the objects.
            # Keep only contour points that are around the sample pixel position.
            pushing_area = self.push_distance / self.pxl_size
            points = []
            for cnt in contours:
                for pnt in cnt:
                    if (p1[0] - pushing_area < pnt[0, 0] < p1[0] + pushing_area) and \
                       (p1[1] - pushing_area < pnt[0, 1] < p1[1] + pushing_area):
                        points.append(pnt[0])
            if len(points) > 0:
                break

        ids = np.arange(len(points))
        random_id = self.rng.choice(ids, 1)[0]
        p2 = points[random_id]
        push_dir = p2 - p1
        theta = -np.arctan2(push_dir[1], push_dir[0])
        step_angle = 2 * np.pi / self.rotations
        discrete_theta = round(theta / step_angle) * step_angle

        # print(theta, discrete_theta)
        #
        # p2[0] = p1[0] + 20 * np.cos(discrete_theta)
        # p2[1] = p1[1] - 20 * np.sin(discrete_theta)
        #
        # plt.imshow(state)
        # plt.plot(p1[0], p1[1], 'o', 2)
        # plt.plot(p2[0], p2[1], 'x', 2)
        # plt.arrow(p1[0], p1[1], p2[0] - p1[0], p2[1] - p1[1], width=1)
        # plt.show()

        # Sample aperture uniformly
        aperture = self.rng.uniform(self.aperture_limits[0], self.aperture_limits[1])

        action = np.zeros((4,))
        action[0] = p1[0]
        action[1] = p1[1]
        action[2] = discrete_theta
        action[3] = aperture

        return action

    def explore(self, state):
        explore_prob = max(0.8 * np.power(0.9998, self.learn_step_counter), 0.1)

        if self.rng.rand() < explore_prob:
            action = self.guided_exploration(state)
        else:
            action = self.predict(state)
        print('explore action:', action)
        return action

    def predict(self, state):

        # Find optimal position and orientation
        x = torch.FloatTensor(self.pre_process(state)). unsqueeze(0).to('cuda')
        out_prob = self.fcn(x, is_volatile=True)
        out_prob = self.post_process(out_prob)

        self.plot_maps(state, out_prob)

        best_action = np.unravel_index(np.argmax(out_prob), out_prob.shape)
        p1 = np.array([best_action[3], best_action[2]])
        theta = best_action[0] * 2 * np.pi / self.rotations

        p2 = np.array([0, 0])
        p2[0] = p1[0] + 20 * np.cos(theta)
        p2[1] = p1[1] - 20 * np.sin(theta)
        #
        # plt.imshow(state)
        # plt.plot(p1[0], p1[1], 'o', 2)
        # plt.plot(p2[0], p2[1], 'x', 2)
        # plt.arrow(p1[0], p1[1], p2[0] - p1[0], p2[1] - p1[1], width=1)
        # plt.show()

        # Find optimal aperture
        aperture_img = self.pre_process_aperture_img(state, p1, theta)

        x = torch.FloatTensor(aperture_img).unsqueeze(0).to('cuda')
        aperture = self.reg(x).detach().cpu().numpy()[0, 0]
        # print('aperture', aperture.detach().cpu().numpy()[0, 0])

        # Undo normalization
        aperture = utils.min_max_scale(aperture,
                                       range=[0, 1],
                                       target_range=[self.aperture_limits[0], self.aperture_limits[1]])

        action = np.zeros((4,))
        action[0] = p1[0]
        action[1] = p1[1]
        action[2] = theta
        action[3] = aperture
        print('Predicted action:', action)
        return action

    def get_fcn_labels(self, state, action):
        heightmap = self.pre_process(state)

        angle = (action[2] + (2 * np.pi)) % (2 * np.pi)
        rot_id = round(angle / (2 * np.pi / 16))

        action_area = np.zeros((state.shape[0], state.shape[1]))
        action_area[int(action[1]), int(action[0])] = 1.0
        label = np.zeros((1, heightmap.shape[1], heightmap.shape[2]))
        label[0, self.padding_width:heightmap.shape[1] - self.padding_width,
              self.padding_width:heightmap.shape[2] - self.padding_width] = action_area

        return heightmap, rot_id, label

    def learn(self, transition):
        if not transition['label']:
            return

        self.replay_buffer.store(transition)

        # if self.replay_buffer.size() < 64:
        #     return

        # Sample from replay buffer
        state, action = self.replay_buffer.sample()

        # Update FCN
        heightmap, rot_id, label = self.get_fcn_labels(state, action)
        x = torch.FloatTensor(heightmap).unsqueeze(0).to('cuda')
        label = torch.FloatTensor(label).unsqueeze(0).to('cuda')
        rotations = np.array([rot_id])
        q_maps = self.fcn(x, specific_rotation=rotations)

        # Compute loss in the whole scene
        loss = self.fcn_criterion(q_maps, label)
        loss = torch.sum(loss)
        print('fcn_loss:', loss.detach().cpu().numpy())
        self.info['fcn_loss'].append(loss.detach().cpu().numpy())

        self.fcn_optimizer.zero_grad()
        loss.backward()
        self.fcn_optimizer.step()

        # Update regression network
        x = torch.FloatTensor(self.pre_process_aperture_img(state,
                                                            p1=np.array([action[0], action[1]]),
                                                            theta=action[2])).unsqueeze(0).to('cuda')
        # Normalize aperture to range 0-1
        normalized_aperture = utils.min_max_scale(action[3],
                                                  range=[self.aperture_limits[0], self.aperture_limits[1]],
                                                  target_range=[0, 1])
        gt_aperture = torch.FloatTensor(np.array([normalized_aperture])).unsqueeze(0).to('cuda')
        pred_aperture = self.reg(x)

        print('APERTURES')
        print(gt_aperture, pred_aperture)

        loss = self.reg_criterion(pred_aperture, gt_aperture)
        print('reg_loss:', loss.detach().cpu().numpy())
        self.info['reg_loss'].append(loss.detach().cpu().numpy())

        self.reg_optimizer.zero_grad()
        loss.backward()
        self.reg_optimizer.step()

        self.learn_step_counter += 1

    def learn2(self, transition):

        # Store state and label to replay buffer
        if transition['label']:
            self.replay_buffer.store(transition)
        return

        # if self.params['agent']['per']:
        #         x = torch.FloatTensor(self.pre_process_aperture_img(transition['state'],
        #                                                             p1=np.array([transition['action'][0],
        #                                                                          transition['action'][1]]),
        #                                                             theta=transition['action'][2])).unsqueeze(0).to('cuda')
        #         # Normalize aperture to range 0-1
        #         normalized_aperture = utils.min_max_scale(transition['action'][3],
        #                                                   range=[self.aperture_limits[0], self.aperture_limits[1]],
        #                                                   target_range=[0, 1])
        #         gt_aperture = torch.FloatTensor(np.array([normalized_aperture])).unsqueeze(0).to('cuda')
        #         pred_aperture = self.reg(x)
        #         error = self.reg_criterion(pred_aperture, gt_aperture).detach().cpu().numpy()
        #         self.replay_buffer.store(error, transition)
        #     else:
        #         self.replay_buffer.store(transition)
        #
        #     # if self.replay_buffer.size() < 64:
        #     #     return
        #
        #     # Sample from replay buffer
        #     # state, action = self.replay_buffer.sample()
        #     state, action, sample_id = self.replay_buffer.sample()
        #
        #     # Update FCN
        #     heightmap, rot_id, label = self.get_fcn_labels(state, action)
        #     x = torch.FloatTensor(heightmap).unsqueeze(0).to('cuda')
        #     label = torch.FloatTensor(label).unsqueeze(0).to('cuda')
        #     rotations = np.array([rot_id])
        #     q_maps = self.fcn(x, specific_rotation=rotations)
        #
        #     # Compute loss in the whole scene
        #     loss = self.fcn_criterion(q_maps, label)
        #     loss = torch.sum(loss)
        #     print('fcn_loss:', loss.detach().cpu().numpy())
        #     self.info['fcn_loss'].append(loss.detach().cpu().numpy())
        #
        #     self.fcn_optimizer.zero_grad()
        #     loss.backward()
        #     self.fcn_optimizer.step()
        #
        #     # Update regression network
        #     x = torch.FloatTensor(self.pre_process_aperture_img(state,
        #                                                         p1=np.array([action[0], action[1]]),
        #                                                         theta=action[2])).unsqueeze(0).to('cuda')
        #     # Normalize aperture to range 0-1
        #     normalized_aperture = utils.min_max_scale(action[3],
        #                                               range=[self.aperture_limits[0], self.aperture_limits[1]],
        #                                               target_range=[0, 1])
        #     gt_aperture = torch.FloatTensor(np.array([normalized_aperture])).unsqueeze(0).to('cuda')
        #     pred_aperture = self.reg(x)
        #
        #     print('APERTURES')
        #     print(gt_aperture, pred_aperture)
        #
        #     loss = self.reg_criterion(pred_aperture, gt_aperture)
        #     print('reg_loss:', loss.detach().cpu().numpy())
        #     self.info['reg_loss'].append(loss.detach().cpu().numpy())
        #
        #     self.reg_optimizer.zero_grad()
        #     loss.backward()
        #     self.reg_optimizer.step()
        #
        #     self.learn_step_counter += 1
        #
        #     # Update priorities
        #     self.replay_buffer.update_priorities(sample_id, loss.detach().cpu().numpy())

            # aperture_error =
            # self.replay_buffer.store()

    def action(self, action):
        # Convert from pixels to 3d coordinates.
        x = -(self.pxl_size * action[0] - self.bounds[0][1])
        y = self.pxl_size * action[1] - self.bounds[1][1]
        quat = ori.Quaternion.from_rotation_matrix(np.matmul(ori.rot_y(-np.pi / 2), ori.rot_x(action[2])))

        return {'pos': np.array([x, y, self.z]),
                'quat': quat,
                'aperture': action[3],
                'push_distance': self.push_distance}

    def plot_maps(self, state, out_prob):

        glob_max_prob = np.max(out_prob)
        fig, ax = plt.subplots(4, 4)
        for i in range(16):
            x = int(i / 4)
            y = i % 4

            min_prob = np.min(out_prob[i][0])
            max_prob = np.max(out_prob[i][0])

            prediction_vis = utils.min_max_scale(out_prob[i][0],
                                                 range=(min_prob, max_prob),
                                                 target_range=(0, 1))
            best_pt = np.unravel_index(prediction_vis.argmax(), prediction_vis.shape)
            maximum_prob = np.max(out_prob[i][0])

            ax[x, y].imshow(state, cmap='gray')
            ax[x, y].imshow(prediction_vis, alpha=0.5)
            ax[x, y].set_title(str(i) + ', ' + str(format(maximum_prob, ".3f")))

            if glob_max_prob == max_prob:
                ax[x, y].plot(best_pt[1], best_pt[0], 'rx')
            else:
                ax[x, y].plot(best_pt[1], best_pt[0], 'ro')
            dx = 20 * np.cos((i / 16) * 2 * np.pi)
            dy = -20 * np.sin((i / 16) * 2 * np.pi)
            ax[x, y].arrow(best_pt[1], best_pt[0], dx, dy, width=2, color='g')

        plt.savefig(os.path.join(self.params['log_dir'], 'maps', 'map_' + str(self.learn_step_counter) + '.png'),
                    dpi=720)
        plt.close()

    def save(self, epoch):
        folder_name = os.path.join(self.params['log_dir'], 'model_' + str(epoch))
        os.mkdir(folder_name)
        torch.save(self.fcn.state_dict(), os.path.join(folder_name, 'fcn.pt'))
        torch.save(self.reg.state_dict(), os.path.join(folder_name, 'reg.pt'))

        log_data = {'params': self.params.copy()}
        pickle.dump(log_data, open(os.path.join(folder_name, 'log_data'), 'wb'))

        self.info['learn_step_counter'] = self.learn_step_counter
        pickle.dump(self.info, open(os.path.join(folder_name, 'info'), 'wb'))

    @ classmethod
    def load(cls, log_dir):
        params = pickle.load(open(os.path.join(log_dir, 'log_data'), 'rb'))
        self = cls(params)

        self.fcn.load_state_dict(torch.load(os.path.join(log_dir, 'fcn.pt')))
        self.fcn.eval()

        self.reg.load_state_dict(torch.load(os.path.join(log_dir, 'reg.pt')))
        self.reg.eval()

        return self

    def load_seperately(self, fcn_model, reg_model):
        self.fcn.load_state_dict(torch.load(fcn_model))
        self.fcn.eval()

        self.reg.load_state_dict(torch.load(reg_model))
        self.reg.eval()

    def resume_training(self, log_dir):
        # First, you have to call the function load...!
        self.info = pickle.load(open(os.path.join(log_dir, 'info'), 'rb'))
        self.learn_step_counter = self.info['learn_step_counter']

    def terminal(self, obs, next_obs):
        # Check if there are only flat objects in the scene
        state = self.state_representation(next_obs)
        obj_ids = np.argwhere(state > self.z)
        if len(obj_ids) == 0:
            return True

        # Check if there is only one object in the scene
        objects_above = 0
        for obj in next_obs['full_state']:
            if obj.pos[2] > 0:
                objects_above += 1
        if objects_above == 1:
            return True

        return super(PushGrasping, self).terminal(obs, next_obs)