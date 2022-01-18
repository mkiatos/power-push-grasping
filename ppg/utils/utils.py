import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
import cv2

from gog.utils import pybullet_utils


def get_pointcloud(depth, intrinsics):
    """
    Creates a point cloud from a depth image given the camera intrinsics parameters.

    Parameters
    ----------
    depth: np.array
        The input image.
    intrinsics: np.array(3, 3)
        Intrinsics parameters of the camera.

    Returns
    -------
    o3d.geometry.PointCloud
        The point cloud of the scene.
    """
    depth = depth
    width, height = depth.shape
    c, r = np.meshgrid(np.arange(height), np.arange(width), sparse=True)
    valid = (depth > 0)
    z = np.where(valid, depth, 0)
    x = np.where(valid, z * (c - intrinsics[0, 2]) / intrinsics[0, 0], 0)
    y = np.where(valid, z * (r - intrinsics[1, 2]) / intrinsics[1, 1], 0)
    pcd = np.dstack((x, y, z))
    return o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pcd.reshape(-1, 3)))


def get_aligned_point_cloud(color, depth, configs, bounds, pixel_size, plot=False):
    """
    Returns the scene point cloud aligned with the center of the workspace.
    """
    full_point_cloud = o3d.geometry.PointCloud()
    for color, depth, config in zip(color, depth, configs):
        intrinsics = np.array(config['intrinsics']).reshape(3, 3)
        point_cloud = get_pointcloud(depth, intrinsics)

        transform = pybullet_utils.get_camera_pose(config['pos'],
                                                   config['target_pos'],
                                                   config['up_vector'])
        point_cloud = point_cloud.transform(transform)

        crop_box = o3d.geometry.AxisAlignedBoundingBox(min_bound=np.array([bounds[0, 0], bounds[1, 0], bounds[2, 0]]),
                                                       max_bound=np.array([bounds[0, 1], bounds[1, 1], bounds[2, 1]]))
        point_cloud = point_cloud.crop(crop_box)
        
        full_point_cloud += point_cloud

    # full_point_cloud.estimate_normals()
    if plot:
        mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
        o3d.visualization.draw_geometries([full_point_cloud, mesh_frame])
    return full_point_cloud


def get_fused_heightmap(obs, configs, bounds, pix_size):
    point_cloud = get_aligned_point_cloud(obs['color'], obs['depth'], configs, bounds, pix_size)
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

    return cv2.flip(height_grid, 1)


def rgb2bgr(rgb):
    """
    Converts a rgb image to bgr

    Parameters
    ----------
    rgb : np.array
        The rgb image

    Returns
    -------
    np.array:
        The image in bgr format
    """
    h, w, c = rgb.shape
    bgr = np.zeros((h, w, c), dtype=np.uint8)
    r = rgb[:, :, 0]
    g = rgb[:, :, 1]
    b = rgb[:, :, 2]
    bgr[:, :, 0] = b
    bgr[:, :, 1] = g
    bgr[:, :, 2] = r
    return bgr


class PinholeCameraIntrinsics:
    """
    PinholeCameraIntrinsics class stores intrinsic camera matrix,
    and image height and width.
    """
    def __init__(self, width, height, fx, fy, cx, cy):

        self.width, self.height = width, height
        self.fx, self.fy = fx, fy
        self.cx, self.cy = cx, cy

    @classmethod
    def from_params(cls, params):
        width, height = params['width'], params['height']
        fx, fy = params['fx'], params['fy']
        cx, cy = params['cx'], params['cy']
        return cls(width, height, fx, fy, cx, cy)

    def get_intrinsic_matrix(self):
        camera_matrix = np.array(((self.fx, 0, self.cx),
                                  (0, self.fy, self.cy),
                                  (0, 0, 1)))
        return camera_matrix

    def get_focal_length(self):
        return self.fx, self.fy

    def get_principal_point(self):
        return self.cx, self.cy

    def back_project(self, p, z):
        x = (p[0] - self.cx) * z / self.fx
        y = (p[1] - self.cy) * z / self.fy
        return np.array([x, y, z])
    

def min_max_scale(x, range, target_range):
    assert range[1] > range[0]
    assert target_range[1] > target_range[0]

    range_min = range[0] * np.ones(x.shape)
    range_max = range[1] * np.ones(x.shape)
    target_min = target_range[0] * np.ones(x.shape)
    target_max = target_range[1] * np.ones(x.shape)
    

    return target_min + ((x - range_min) * (target_max - target_min)) / (range_max - range_min)
