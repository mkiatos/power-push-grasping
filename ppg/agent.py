import numpy as np
import pybullet as p
import matplotlib.pyplot as plt
import cv2
import random
import torch

from ppg.utils import orientation as ori
from ppg.utils.utils import min_max_scale
from ppg.models import ResFCN, Classifier


class PushGrasping:
    def __init__(self,
                 robot_hand):
        self.robot_hand = robot_hand

        self.init_distance = 0.15
        self.rotations = 16
        self.widths = np.array([0.6, 0.7, 0.8, 0.9, 1.0, 1.1])

        self.model = ResFCN().to('cuda')

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

    def action(self, action, bounds, pxl_size):
        x = -(pxl_size * action[0] - bounds[0][1])
        y = pxl_size * action[1] - bounds[1][1]

        quat = ori.Quaternion.from_rotation_matrix(np.matmul(ori.rot_y(-np.pi / 2), ori.rot_x(action[2])))

        return {'pos': np.array([x, y, 0.1]), 'quat': quat, 'width': action[3]}

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

    def load(self, model_weights):
        checkpoint_model = torch.load(model_weights)
        self.model.load_state_dict(checkpoint_model)

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

    def predict(self, heightmap):
        diag_length = float(heightmap.shape[0]) * np.sqrt(2)
        diag_length = np.ceil(diag_length / 16) * 16
        self.padding_width = int((diag_length - heightmap.shape[0]) / 2)
        depth_heightmap = np.pad(heightmap, self.padding_width, 'constant', constant_values=-0.01)
        padded_shape = (depth_heightmap.shape[0], depth_heightmap.shape[1])

        # Normalize maps ( ToDo: find mean and std) # Todo
        image_mean = 0.01
        image_std = 0.03
        depth_heightmap = (depth_heightmap - image_mean) / image_std

        # Add extra channel
        depth_heightmap = np.expand_dims(depth_heightmap, axis=0)

        x = torch.FloatTensor(depth_heightmap).unsqueeze(0).to('cuda')
        out_prob = self.model(x, is_volatile=True)
        out_prob = self.post_process(out_prob)

        glob_max_prob = np.max(out_prob)
        for j in range(6):
            fig, ax = plt.subplots(4, 4)
            for i in range(16):
                x = int(i / 4)
                y = i % 4

                min_prob = np.min(out_prob[i][j])
                max_prob = np.max(out_prob[i][j])

                prediction_vis = min_max_scale(out_prob[i][j],
                                               range=(min_prob, max_prob),
                                               target_range=(0, 1))
                best_pt = np.unravel_index(prediction_vis.argmax(), prediction_vis.shape)
                maximum_prob = np.max(out_prob[i][j])

                prediction_vis = cv2.applyColorMap((prediction_vis * 255).astype(np.uint8), cv2.COLORMAP_JET)
                prediction_vis = cv2.cvtColor(prediction_vis, cv2.COLOR_BGR2RGB)
                prediction_vis = (0.5 * cv2.cvtColor(heightmap, cv2.COLOR_RGB2BGR) + 0.5 * prediction_vis).astype(
                    np.uint8)
                ax[x, y].imshow(prediction_vis)
                ax[x, y].set_title(str(i) + ', ' + str(format(maximum_prob, ".3f")))

                if glob_max_prob == max_prob:
                    ax[x, y].plot(best_pt[1], best_pt[0], 'rx')
                else:
                    ax[x, y].plot(best_pt[1], best_pt[0], 'ro')
                dx = 20 * np.cos((i / 16) * 2 * np.pi)
                dy = -20 * np.sin((i / 16) * 2 * np.pi)
                ax[x, y].arrow(best_pt[1], best_pt[0], dx, dy, width=2, color='g')
            plt.show()

        best_action = np.unravel_index(np.argmax(out_prob), out_prob.shape)

        p1 = np.array([best_action[2], best_action[3]])
        theta = best_action[0] * 2 * np.pi / self.rotations

        p2 = np.array([0, 0])
        p2[0] = p1[0] + 20 * np.cos(theta)
        p2[1] = p1[1] - 20 * np.sin(theta)

        plt.imshow(heightmap)
        plt.plot(p1[0], p1[1], 'o', 2)
        plt.plot(p2[0], p2[1], 'x', 2)
        plt.arrow(p1[0], p1[1], p2[0] - p1[0], p2[1] - p1[1], width=1)
        plt.show()

        return p1[0], p1[1], theta, self.widths[best_action[1]]


