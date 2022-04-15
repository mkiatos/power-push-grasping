import pickle
import matplotlib.pyplot as plt
import yaml
import numpy as np
import cv2
import open3d as o3d

from ppg.agent import PushGrasping
from ppg.utils import utils
from ppg.utils.orientation import Quaternion


def get_point_cloud(depth, intrinsics):
    depth = depth.astype(np.float32)
    depth /= 10000.0
    width, height = depth.shape
    c, r = np.meshgrid(np.arange(height), np.arange(width), sparse=True)
    valid = (depth > 0)
    z = np.where(valid, depth, 0)
    x = np.where(valid, z * (c - intrinsics[0, 2]) / intrinsics[0, 0], 0)
    y = np.where(valid, z * (r - intrinsics[1, 2]) / intrinsics[1, 1], 0)
    pcd = np.dstack((x, y, z))
    point_cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pcd.reshape(-1, 3)))
    return point_cloud


def get_aligned_point_cloud(depth, camera, bounds, plot=False):
    point_cloud = get_point_cloud(depth, camera.intrinsics)
    point_cloud = point_cloud.transform(camera.pose)

    # Crop w.r.t. bounds
    crop_box = o3d.geometry.AxisAlignedBoundingBox(min_bound=np.array([bounds[0, 0], bounds[1, 0], bounds[2, 0]]),
                                                   max_bound=np.array([bounds[0, 1], bounds[1, 1], bounds[2, 1]]))
    point_cloud = point_cloud.crop(crop_box)

    if plot:
        mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
        o3d.visualization.draw_geometries([point_cloud, mesh_frame])
    return point_cloud


def get_fused_heightmap(obs, camera_config, bounds, pix_size):
    point_cloud = get_aligned_point_cloud(obs['depth'], camera_config, bounds, plot=True)
    xyz = np.asarray(point_cloud.points)

    # Compute heightmap size
    heightmap_size = np.round(((bounds[1][1] - bounds[1][0]) / pix_size,
                               (bounds[0][1] - bounds[0][0]) / pix_size)).astype(int)

    height_grid = np.zeros((heightmap_size[0], heightmap_size[0]), dtype=np.float32)

    for i in range(xyz.shape[0]):
        x = xyz[i][0]
        y = xyz[i][1]
        z = xyz[i][2]

        idx_x = int(np.floor((x + bounds[0][1]) / pix_size))
        idx_y = int(np.floor((y + bounds[1][1]) / pix_size))

        if 0 < idx_x < heightmap_size[0] - 1 and 0 < idx_y < heightmap_size[1] - 1:
            if height_grid[idx_y][idx_x] < z:
                height_grid[idx_y][idx_x] = z

    return cv2.flip(height_grid, 0)


class RealCamera:
    def __init__(self):
        (fx, fy, cx, cy) = pickle.load(open('scene_2/intrinsics', 'rb'))
        self.intrinsics = np.array([[fx, 0, cx],
                                    [0, fy, cy],
                                    [0, 0, 1]])
        # print(self.intrinsics)

        with open('scene_2/square_workspace.yaml', 'r') as stream:
            camera_params = yaml.safe_load(stream)

        t = camera_params[0]['TF_static']['t']
        q = camera_params[0]['TF_static']['quat']

        self.pose = np.eye(4)
        self.pose[0:3, 3] = t
        self.pose[0:3, 0:3] = Quaternion(x=q[0], y=q[1], z=q[2], w=q[3]).rotation_matrix()
        # print(self.pose)


class PushGraspingReal(PushGrasping):
    def __init__(self, params):
        super(PushGraspingReal, self).__init__(params)

        self.camera = RealCamera()
        self.bounds = np.array([[-0.25, 0.25],
                                [-0.25, 0.25],
                                [0.04, 0.2]])
        self.z = 0.11

        self.camera_robot_transform = np.eye(4)

    def state_representation(self, obs):
        state = get_fused_heightmap(obs, self.camera, self.bounds, self.pxl_size)
        return state

    def action(self, action):
        # Convert from pixels to 3d coordinates.
        x = self.pxl_size * action[0] - self.bounds[0][1]
        y = -(self.pxl_size * action[1] - self.bounds[1][1])
        angle = action[2]
        aperture = action[3]

        return {'pos': np.array([x, y, self.z]),
                'angle': angle,
                'aperture': aperture,
                'push_distance': self.push_distance}

    def transform_wrt_real_robot(self, workspace_action):
        # Transform point w.r.t. robot base
        pos = workspace_action['pos']
        pos_wrt_cam = np.matmul(np.linalg.inv(self.camera.pose), np.array([pos[0], pos[1], pos[2], 1.0]))
        print('Point w.r.t. camera:', pos_wrt_cam)

        p_1 = np.matmul(self.camera_robot_transform, pos_wrt_cam)
        print('Point w.r.t. robot:', p_1)

        # Estimate final point
        d = workspace_action['push_distance']
        angle = workspace_action['angle']
        pos_2 = np.array([pos[0] + d * np.cos(angle),
                          pos[1] + d * np.sin(angle),
                          pos[2]])

        print(pos)
        print(pos_2)

        # Transform angle
        theta = 0
        return 0


def test_real_exp(fcn_model, reg_model):
    # Load agent
    with open('../yaml/bhand.yml', 'r') as stream:
        params = yaml.safe_load(stream)
    policy = PushGraspingReal(params)
    policy.load(fcn_model=fcn_model, reg_model=reg_model)

    # Get observation
    color = pickle.load(open('scene_2/color', 'rb'))
    depth = pickle.load(open('scene_2/depth', 'rb'))

    # fig, ax = plt.subplots(1, 2)
    # ax[0].imshow(color)
    # ax[1].imshow(depth)
    # plt.show()

    obs = {'rgb': color, 'depth': depth}

    # Get state representation
    state = policy.state_representation(obs)

    # plt.imshow(state)
    # plt.show()

    # Predict action
    action = policy.predict(state)
    env_action = policy.action(action)

    real_action = policy.transform_wrt_real_robot(env_action)


    # print(env_action['pos'])
    # pos_wrt_cam = np.matmul(np.linalg.inv(policy.camera.pose), np.array([env_action['pos'][0],
    #                                                                      env_action['pos'][1],
    #                                                                      env_action['pos'][2], 1.0]))
    # print('Point w.r.t. camera:', pos_wrt_cam)
    #
    # g_rc = np.eye(4)
    # g_rc[0:3, 3] = np.array([0.3500, 0.1274, 0.5500])
    # g_rc[0:3, 0:3] = Quaternion(x=0.6812, y=-0.6447, z=0.2301, w=-0.2596).rotation_matrix()
    # p_c = np.matmul(g_rc, pos_wrt_cam)
    # print('Point w.r.t. robot:', p_c)
    # # print(action, env_action)


if __name__ == "__main__":

    test_real_exp(fcn_model='../downloads/fcn_model.pt',
                  reg_model='../downloads/reg_model.pt')