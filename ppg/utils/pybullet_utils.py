"""
PyBullet Utilities
==================

"""

import pybullet as p
import numpy as np
import os
import time

from ppg.utils.orientation import Quaternion


def get_joint_indices(names, body_unique_id=0):
    indices = []
    for name in names:
        for i in range(p.getNumJoints(body_unique_id)):
            joint = p.getJointInfo(body_unique_id, i)
            if str(joint[1], 'utf-8') == name:
                indices.append(joint[0])
                break
    return indices


def get_link_indices(names, body_unique_id=0):
    indices = []
    for name in names:
        for i in range(p.getNumJoints(body_unique_id)):
            joint = p.getJointInfo(body_unique_id, i)
            if joint[12].decode('utf-8') == name:
                indices.append(joint[0])
    return indices


def get_link_pose(name, body_unique_id=0):
    index = get_link_indices([name], body_unique_id)[0]
    state = p.getLinkState(0, index)
    pos = [state[0][0], state[0][1], state[0][2]]
    q = state[1]
    quat = Quaternion(w=q[3], x=q[0], y=q[1], z=q[2])
    return pos, quat


def get_camera_pose(pos, target_pos, up_vector):
    """
    Computes the camera pose w.r.t. world

    Parameters
    ----------
    pos: np.array([3, 1])
        The position of the camera in world coordinates
    target_pos: np.array([3, 1])
        The position of the focus point in world coordinates e.g. lookAt vector
    up_vector: np.array([3, 1])
        Up vector of the camera in world coordinates

    Returns
    -------
    np.array([4, 4])
        The camera pose w.r.t. world frame
    """
    # Compute z axis
    z = target_pos - pos
    z /= np.linalg.norm(z)

    # Compute y axis. Project the up vector to the plane defined
    # by the point pos and unit vector z
    dist = np.dot(z, -up_vector)
    y = -up_vector - dist * z
    y /= np.linalg.norm(y)

    # Compute x axis as the cross product of y and z
    x = np.cross(y, z)
    x /= np.linalg.norm(x)

    camera_pose = np.eye(4)
    camera_pose[0:3, 0] = x
    camera_pose[0:3, 1] = y
    camera_pose[0:3, 2] = z
    camera_pose[0:3, 3] = pos
    return camera_pose


def load_urdf(pybullet_client, file_path, *args, **kwargs):
  """Loads the given URDF filepath."""
  # Handles most general file open case.
  try:
    return pybullet_client.loadURDF(file_path, *args, **kwargs)
  except pybullet_client.error:
    pass


def add_line(start, end, color=[0, 0, 1], width=3, lifetime=None):
    """[summary]
    Parameters
    ----------
    start : [type]
        [description]
    end : [type]
        [description]
    color : tuple, optional
        [description], by default (0, 0, 0)
    width : int, optional
        [description], by default 1
    lifetime : [type], optional
        [description], by default None
    parent : int, optional
        [description], by default NULL_ID
    parent_link : [type], optional
        [description], by default BASE_LINK
    Returns
    -------
    [type]
        [description]
    """
    return p.addUserDebugLine(start, end, lineColorRGB=color[:3], lineWidth=width)


def draw_pose(pos, quat, length=0.1, **kwargs):
    pose = np.eye(4)
    pose[0:3, 0:3] = quat.rotation_matrix()
    pose[0:3, 3] = pos
    for k in range(3):
        axis = np.zeros(3)
        axis[k] = 1
        frame_axis = pose[0:3, k] * length + pose[0:3, 3]
        add_line(start=pose[0:3, 3], end=frame_axis, color=axis, **kwargs)


def load_obj(obj_path, scaling=1.0, position=[0, 0, 0], orientation=Quaternion(), fixed_base=False):
    template = """<?xml version="1.0" encoding="UTF-8"?>
                  <robot name="obj.urdf">
                      <link name="baseLink">
                          <contact>
                              <lateral_friction value="1.0"/>
                              <rolling_friction value="0.0001"/>
                              <inertia_scaling value="3.0"/>
                          </contact>
                          <inertial>
                              <origin rpy="0 0 0" xyz="0 0 0"/>
                              <mass value="1"/>
                              <inertia ixx="1" ixy="0" ixz="0" iyy="1" iyz="0" izz="1"/>
                          </inertial>
                          <visual>
                              <origin rpy="0 0 0" xyz="0 0 0"/>
                              <geometry>
                                  <mesh filename="{0}" scale="1 1 1"/>
                              </geometry>
                              <material name="mat_2_0">
                                  <color rgba="0.5 0.5 0.5 1.0" />
                              </material>
                          </visual>
                          <collision>
                              <origin rpy="0 0 0" xyz="0 0 0"/>
                              <geometry>
                                  <mesh filename="{1}" scale="1 1 1"/>
                              </geometry>
                          </collision>
                      </link>
                  </robot>"""
    urdf_path = '.tmp_my_obj_%.8f%.8f.urdf' % (time.time(), np.random.rand())
    with open(urdf_path, "w") as f:
        f.write(template.format(obj_path, obj_path))
    body_id = p.loadURDF(
        fileName=urdf_path,
        basePosition=position,
        baseOrientation=orientation,
        globalScaling=scaling,
        useFixedBase=fixed_base
    )
    os.remove(urdf_path)

    return body_id


def get_distances_from_target(obs):
    objects = obs['full_state']

    distances = {}
    for target in objects:
        if target.pos[2] < 0:
            continue

        distances[target.body_id] = []
        for obj in objects:
            if obj.pos[2] < 0 or target.body_id == obj.body_id:
                continue

            points = p.getClosestPoints(target.body_id, obj.body_id, distance=10)
            if len(points) > 0:
                dist = np.linalg.norm(np.array(points[0][5]) - np.array(points[0][6]))
                distances[target.body_id].append(dist)

    for obj_id in distances:
        if len(distances[obj_id]) != 0:
            distances[obj_id] = np.min(distances[obj_id])

    return distances
