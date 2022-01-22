import numpy as np
import pybullet as p
import matplotlib.pyplot as plt
import cv2
import random
import torch

from ppg.utils import orientation as ori
from ppg.utils.utils import min_max_scale
from ppg.models import ResFCN, Classifier, Regressor


class PushGrasping:
    def __init__(self, params):
        self.params = params

        self.rotations = params['rotations']
        self.aperture_limits = params['aperture_limits']
        self.aperture_limits = [0.6, 0.8, 1.1]
        self.crop_size = 32

        self.fcn = ResFCN().to('cuda')
        self.classifier = Classifier(n_classes=3).to('cuda')

    def pre_process(self, heightmap):
        diag_length = float(heightmap.shape[0]) * np.sqrt(2)
        diag_length = np.ceil(diag_length / 16) * 16
        self.padding_width = int((diag_length - heightmap.shape[0]) / 2)
        padded_heightmap = np.pad(heightmap, self.padding_width, 'constant', constant_values=-0.01)

        # Normalize maps ( ToDo: find mean and std)
        image_mean = 0.01
        image_std = 0.03
        padded_heightmap = (padded_heightmap - image_mean) / image_std

        # Add extra channel
        padded_heightmap = np.expand_dims(padded_heightmap, axis=0)
        return padded_heightmap

    def post_process(self, q_maps):
        """
        Remove extra padding

        Params:
            shape: the output shape of preprocess

        Returns rotations x openings x w x h
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

    def seed(self, seed):
        random.seed(seed)

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

    def pre_process_aperture_img(self, heightmap, p1, theta, plot=True):
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

    def predict(self, heightmap):
        input_heightmap = self.pre_process(heightmap)

        # Find optimal position and orientation
        x = torch.FloatTensor(input_heightmap).unsqueeze(0).to('cuda')
        out_prob = self.fcn(x, is_volatile=True)
        out_prob = self.post_process(out_prob)

        best_action = np.unravel_index(np.argmax(out_prob), out_prob.shape)

        p1 = np.array([best_action[3], best_action[2]])

        theta = best_action[0] * 2 * np.pi / self.rotations

        p2 = np.array([0, 0])
        p2[0] = p1[0] + 20 * np.cos(theta)
        p2[1] = p1[1] - 20 * np.sin(theta)

        plt.imshow(heightmap)
        plt.plot(p1[0], p1[1], 'o', 2)
        plt.plot(p2[0], p2[1], 'x', 2)
        plt.arrow(p1[0], p1[1], p2[0] - p1[0], p2[1] - p1[1], width=1)
        plt.show()

        # Find optimal aperture
        aperture_img = self.pre_process_aperture_img(heightmap, p1, theta)
        x = torch.FloatTensor(aperture_img).unsqueeze(0).to('cuda')
        pred = self.classifier(x)
        max_id = torch.argmax(pred).detach().cpu().numpy()

        return p1[0], p1[1], theta, self.aperture_limits[max_id]

    def action(self, action, bounds, pxl_size):
        print(action)
        x = -(pxl_size * action[0] - bounds[0][1])
        y = pxl_size * action[1] - bounds[1][1]
        print(x, y)

        quat = ori.Quaternion.from_rotation_matrix(np.matmul(ori.rot_y(-np.pi / 2), ori.rot_x(action[2])))

        return {'pos': np.array([x, y, 0.1]), 'quat': quat, 'width': action[3]}

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

