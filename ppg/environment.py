import pybullet as p
import numpy as np
import os
import math
import time
import cv2
import matplotlib.pyplot as plt

from ppg.utils.orientation import Quaternion, Affine3, angle_axis2rot, rot_y, rot_z
from ppg.utils import robotics, pybullet_utils, urdf_editor
from ppg import cameras
import ppg.utils.utils as utils

UR5_URDF_PATH = 'ur5e_bhand.urdf'
UR5_WORKSPACE_URDF_PATH = 'table/table.urdf'
PLANE_URDF_PATH = "plane/plane.urdf"


class Object:
    def __init__(self,
                 name='',
                 pos=[0.0, 0.0, 0.0],
                 quat=Quaternion(),
                 size=[],
                 color=[1.0, 1.0, 1.0, 1.0],
                 body_id=None):
        """
          Represents an object.

          Parameters
          ----------
          name : str
              The name of the object
          pos : list
              The position of the object
          quat : Quaternion
              The orientation of the object in the form of quaternion
          body_id : int
              A unique id for the object
        """
        self.name = name
        self.pos = pos
        self.size = size
        self.home_quat = quat
        self.color = color
        self.body_id = body_id


class FloatingGripper:
    """
    A mouving mount and a gripper. The mount has 4 joints:
            0: prismatic x
            1: prismatic y
            2: prismatic z
            3: revolute z
    """

    def __init__(self,
                 robot_hand_urdf,
                 home_position,
                 pos_offset,
                 orn_offset):
        self.home_position = home_position
        self.mount_urdf = '../assets/mount.urdf'

        # If there is no urdf file, generate the mounted-gripper urdf.
        mounted_urdf_name = "../assets/mounted_" + robot_hand_urdf.split('/')[-1].split('.')[0] + ".urdf"
        if not os.path.exists(mounted_urdf_name):
            self.generate_mounted_urdf(robot_hand_urdf, pos_offset, orn_offset)

        # rotation w.r.t. inertia frame
        self.home_quat = Quaternion.from_rotation_matrix(rot_y(-np.pi / 2))

        # Load robot hand urdf.
        self.robot_hand_id = pybullet_utils.load_urdf(
            p,
            mounted_urdf_name,
            useFixedBase=True,
            basePosition=self.home_position,
            baseOrientation=self.home_quat.as_vector("xyzw")
        )

        # Mount joints.
        self.joint_ids = [0, 1, 2, 3]

    def generate_mounted_urdf(self,
                              robot_hand_urdf,
                              pos_offset,
                              orn_offset):
        """
        Generates the urdf with a moving mount attached to a gripper.
        """

        # Load gripper.
        robot_id = pybullet_utils.load_urdf(
            p,
            robot_hand_urdf,
            flags=p.URDF_USE_SELF_COLLISION
        )

        # Load mount.
        mount_body_id = pybullet_utils.load_urdf(
            p,
            self.mount_urdf,
            useFixedBase=True
        )

        # Combine mount and gripper by a joint.
        ed_mount = urdf_editor.UrdfEditor()
        ed_mount.initializeFromBulletBody(mount_body_id, 0)
        ed_gripper = urdf_editor.UrdfEditor()
        ed_gripper.initializeFromBulletBody(robot_id, 0)

        self.gripper_parent_index = 4  # 4 joints of mount
        new_joint = ed_mount.joinUrdf(
            childEditor=ed_gripper,
            parentLinkIndex=self.gripper_parent_index,
            jointPivotXYZInParent=pos_offset,
            jointPivotRPYInParent=p.getEulerFromQuaternion(orn_offset),
            jointPivotXYZInChild=[0, 0, 0],
            jointPivotRPYInChild=[0, 0, 0],
            parentPhysicsClientId=0,
            childPhysicsClientId=0
        )
        new_joint.joint_type = p.JOINT_FIXED
        new_joint.joint_name = "joint_mount_gripper"
        urdfname = "../assets/mounted_" + robot_hand_urdf.split('/')[-1].split('.')[0] + ".urdf"
        ed_mount.saveUrdf(urdfname)

        # Remove mount and gripper bodies.
        p.removeBody(mount_body_id)
        p.removeBody(robot_id)

    def move(self, target_pos, target_quat, duration=2.0, stop_at_contact=False):
        # Compute translation.
        affine_trans = np.eye(4)
        affine_trans[0:3, 0:3] = self.home_quat.rotation_matrix()
        affine_trans[0:3, 3] = self.home_position
        target_pos = np.matmul(np.linalg.inv(affine_trans), np.append(target_pos, 1.0))[0:3]

        # Compute angle.
        relative_rot = np.matmul(self.home_quat.rotation_matrix().transpose(), target_quat.rotation_matrix())
        angle = np.arctan2(relative_rot[2, 1], relative_rot[1, 1])
        target_states = [target_pos[0], target_pos[1], target_pos[2], angle]

        current_pos = []
        for i in self.joint_ids:
            current_pos.append(p.getJointState(0, i)[0])

        trajectories = []
        for i in range(len(self.joint_ids)):
            trajectories.append(robotics.Trajectory([0, duration], [current_pos[i], target_states[i]]))

        t = 0
        dt = 0.001
        is_in_contact = False
        while t < duration:
            command = []
            for i in range(len(self.joint_ids)):
                command.append(trajectories[i].pos(t))

            p.setJointMotorControlArray(
                self.robot_hand_id,
                self.joint_ids,
                p.POSITION_CONTROL,
                targetPositions=command,
                forces=[100 * self.force, 100 * self.force, 100 * self.force, 100 * self.force],
                positionGains=[100 * self.speed, 100 * self.speed, 100 * self.speed, 100 * self.speed]
            )

            # points = p.getContactPoints(bodyA=self.robot_hand_id)
            # if len(points) > 0:
            #     for pnt in points:
            #         print('forces:', np.linalg.norm(np.array(pnt[-1])))

            if stop_at_contact:
                points = p.getContactPoints(bodyA=self.robot_hand_id)
                if len(points) > 0:
                    for pnt in points:
                        if pnt[9] > 0:
                            is_in_contact = True
                            break
                if is_in_contact:
                    break

            t += dt
            p.stepSimulation()
            time.sleep(dt)

        return is_in_contact

    def step_constraints(self):
        raise NotImplementedError

    def close(self):
        raise NotImplementedError

    def open(self):
        raise NotImplementedError


class FloatingBHand(FloatingGripper):
    def __init__(self,
                 bhand_urdf,
                 home_position):

        # Define the mount link position w.r.t. hand base link.
        pos_offset = np.array([0.0, 0, -0.065])
        orn_offset = p.getQuaternionFromEuler([0, 0.0, 0.0])

        super(FloatingBHand, self).__init__(bhand_urdf, home_position, pos_offset, orn_offset)

        pose = pybullet_utils.get_link_pose('mount_link')
        pybullet_utils.draw_pose(pose[0], pose[1])

        # Define force and speed (movement of mount).
        self.force = 10000
        self.speed = 0.01

        # Bhand joints.
        self.joint_names = ['bh_j11_joint', 'bh_j21_joint', 'bh_j12_joint', 'bh_j22_joint',
                            'bh_j32_joint', 'bh_j13_joint', 'bh_j23_joint', 'bh_j33_joint']
        self.indices = pybullet_utils.get_joint_indices(self.joint_names, self.robot_hand_id)

        # Bhand links (for contact check)
        self.link_names = ['bh_base_link',
                           'bh_finger_32_link', 'bh_finger_33_link',
                           'bh_finger_22_link', 'bh_finger_23_link',
                           'bh_finger_12_link', 'bh_finger_13_link']
        self.link_indices = pybullet_utils.get_link_indices(self.link_names, body_unique_id=self.robot_hand_id)
        self.distals = ['bh_finger_33_link', 'bh_finger_23_link', 'bh_finger_13_link']
        self.distal_indices = pybullet_utils.get_link_indices(self.distals, body_unique_id=self.robot_hand_id)

        # Move fingers to home position.
        init_aperture_value = 0.6
        self.move_fingers([0.0, init_aperture_value, init_aperture_value, init_aperture_value])

        # pose_13 = pybullet_utils.get_link_pose('bh_finger_1_tip_link')
        # pose_23 = pybullet_utils.get_link_pose('bh_finger_2_tip_link')
        # pose_33 = pybullet_utils.get_link_pose('bh_finger_3_tip_link')
        # p1 = (np.array(pose_13[0]) + np.array(pose_23[0])) / 2.0
        # p2 = np.array(pose_33[0])
        # print(p1, p2)
        # dist = np.linalg.norm(p1 - p2)
        # print(dist)
        # input('')

        self.configure(n_links_before=4)

    def set_hand_joint_position(self, joint_position, force):
        for i in range(len(self.joint_names)):
            if self.joint_names[i] in ['bh_j32_joint', 'bh_j33_joint']:
                apply_force = 1.7 * force
            else:
                apply_force = force
            p.setJointMotorControl2(bodyUniqueId=0,
                                    jointIndex=self.indices[i],
                                    controlMode=p.POSITION_CONTROL,
                                    targetPosition=joint_position[i],
                                    force=apply_force)

    def move_fingers(self, final_joint_values, duration=1, force=2):
        """
        Move fingers while keeping the hand to the same pose.
        """

        # Get current joint positions
        current_pos = []
        for i in self.indices:
            current_pos.append(p.getJointState(0, i)[0])

        hand_pos = []
        for i in self.joint_ids:
            hand_pos.append(p.getJointState(0, i)[0])

        final = [final_joint_values[0], final_joint_values[0],
                 final_joint_values[1], final_joint_values[2],
                 final_joint_values[3], final_joint_values[1] / 3,
                 final_joint_values[2] / 3, final_joint_values[3] / 3]

        trajectories = []
        for i in range(len(self.indices)):
            trajectories.append(robotics.Trajectory([0, duration], [current_pos[i], final[i]]))

        t = 0
        dt = 0.001
        while t < duration:
            command = []
            for i in range(len(self.joint_names)):
                command.append(trajectories[i].pos(t))
            self.set_hand_joint_position(command, force)
            # self.step_constraints()

            # Keep the hand the same pose.
            p.setJointMotorControlArray(
                self.robot_hand_id,
                self.joint_ids,
                p.POSITION_CONTROL,
                targetPositions=hand_pos,
                forces=[100 * self.force] * len(self.joint_ids),
                positionGains=[100 * self.speed] * len(self.joint_ids)
            )

            t += dt
            p.stepSimulation()
            time.sleep(dt)

    def step_constraints(self):
        current_pos = []
        for i in self.indices:
            current_pos.append(p.getJointState(0, i)[0])

        p.setJointMotorControlArray(
            self.robot_hand_id,
            self.indices,
            p.POSITION_CONTROL,
            targetPositions=current_pos,
            forces=[100 * self.force] * len(self.indices),
            positionGains=[100 * self.speed] * len(self.indices)
        )

    def close(self, joint_vals=[0.0, 1.8, 1.8, 1.8], duration=2):
        self.move_fingers(joint_vals)

    def open(self, joint_vals=[0.0, 0.6, 0.6, 0.6]):
        self.move_fingers(joint_vals, duration=.1)

    def configure(self, n_links_before):
        # Set friction coefficients for gripper fingers
        for i in range(n_links_before, p.getNumJoints(self.robot_hand_id)):
            p.changeDynamics(self.robot_hand_id, i,
                             lateralFriction=1.0,
                             spinningFriction=1.0,
                             rollingFriction=0.0001,
                             frictionAnchor=True)

    def is_grasp_stable(self):
        distal_contacts = 0
        for link_id in self.distal_indices:
            contacts = p.getContactPoints(bodyA=self.robot_hand_id, linkIndexA=link_id)
            distal_contacts += len(contacts)

        body_b = []
        total_contacts = 0
        for link_id in self.link_indices:
            contacts = p.getContactPoints(bodyA=self.robot_hand_id, linkIndexA=link_id)
            if len(contacts) == 0:
                continue
            for pnt in contacts:
                body_b.append(pnt[2])
            total_contacts += len(contacts)

        # print('grasped objects:', len(np.unique(body_b)))
        # print('distals/total: {}/{}'.format(distal_contacts, total_contacts))

        if distal_contacts == total_contacts or len(np.unique(body_b)) != 1:
            return False, 0
        elif distal_contacts > total_contacts:
            assert (distal_contacts > total_contacts)
        else:
            return True, total_contacts


class SimCamera:
    def __init__(self, config):
        self.pos = np.array(config['pos'])
        self.target_pos = np.array(config['target_pos'])
        self.up_vector = np.array(config['up_vector'])

        # Compute view matrix.
        self.view_matrix = p.computeViewMatrix(cameraEyePosition=self.pos,
                                               cameraTargetPosition=self.target_pos,
                                               cameraUpVector=self.up_vector)

        self.z_near = config['zrange'][0]
        self.z_far = config['zrange'][1]
        self.width, self.height = config['image_size'][1], config['image_size'][0]
        self.fy = config['intrinsics'][0]

        # Compute projection matrix.
        fov_h = math.atan(self.height / 2 / self.fy) * 2 / math.pi * 180
        self.projection_matrix = p.computeProjectionMatrixFOV(fov=fov_h, aspect=self.width / self.height,
                                                              nearVal=self.z_near, farVal=self.z_far)

    def get_pose(self):
        """
        Returns the camera pose w.r.t. world

        Returns
        -------
        np.array()
            4x4 matrix representing the camera pose w.r.t. world
        """
        return pybullet_utils.get_camera_pose(self.pos, self.target_pos, self.up_vector)

    def get_depth(self, depth_buffer):
        """
        Converts the depth buffer to depth map.

        Parameters
        ----------
        depth_buffer: np.array()
            The depth buffer as returned from opengl
        """
        depth = self.z_far * self.z_near / (self.z_far - (self.z_far - self.z_near) * depth_buffer)
        return depth

    def get_data(self):
        """
        Returns
        -------
        np.array(), np.array(), np.array()
            The rgb, depth and segmentation images
        """
        image = p.getCameraImage(self.width, self.height,
                                 self.view_matrix, self.projection_matrix,
                                 flags=p.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX)
        return image[2], self.get_depth(image[3]), image[4]


class Environment:
    """
    Class that implements the environment in pyBullet.
    Parameters
    ----------
    """

    def __init__(self,
                 assets_root,
                 workspace_pos,
                 disp=True,
                 hz=240):

        self.bounds = np.array([[-0.25, 0.25], [-0.25, 0.25]])  # workspace limits
        self.assets_root = assets_root
        self.workspace_pos = np.array(workspace_pos)
        self.scene_generator = SceneGenerator(assets_root, [2], self.bounds)

        # Setup cameras.
        self.agent_cams = []
        for config in cameras.RealSense.CONFIG:
            config_world = config.copy()
            config_world['pos'] = self.workspace2world(config['pos'])[0]
            config_world['target_pos'] = self.workspace2world(config['target_pos'])[0]
            self.agent_cams.append(SimCamera(config_world))

        # Start PyBullet.
        if disp:
            p.connect(p.GUI)

            # Move default camera closer to the scene.
            target = np.array(workspace_pos)
            p.resetDebugVisualizerCamera(
                cameraDistance=0.75,
                cameraYaw=180,
                cameraPitch=-45,
                cameraTargetPosition=target)
        else:
            p.connect(p.DIRECT)

        p.setAdditionalSearchPath(self.assets_root)
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        p.setTimeStep(1.0 / hz)

    def workspace2world(self, pos=None, quat=None, inv=False):
        """
        Transforms a pose in workspace coordinates to world coordinates

        Parameters
        ----------
        pos: list
            The position in workspace coordinates

        quat: Quaternion
            The quaternion in workspace coordinates

        Returns
        -------

        list: position in worldcreate_scene coordinates
        Quaternion: quaternion in world coordinates
        """
        world_pos, world_quat = None, None
        tran = Affine3.from_vec_quat(self.workspace_pos, Quaternion()).matrix()

        if inv:
            tran = Affine3.from_matrix(np.linalg.inv(tran)).matrix()

        if pos is not None:
            world_pos = np.matmul(tran, np.append(pos, 1))[:3]
        if quat is not None:
            world_rot = np.matmul(tran[0:3, 0:3], quat.rotation_matrix())
            world_quat = Quaternion.from_rotation_matrix(world_rot)

        return world_pos, world_quat

    def load_obj(self, obj_path, scaling=1.0, position=[0, 0, 0], orientation=Quaternion(), fixed_base=False,
                 visual_path=None):
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

    def add_object(self, obj_path, pos, quat):
        """
        Adds an object in the scene
        """
        # col_box_id = p.createCollisionShape(shapeType=p.GEOM_MESH, fileName=obj_path)
        # visual_shape_id = p.createVisualShape(shapeType=p.GEOM_MESH, fileName=obj_path)

        # length = 0.15
        # col_box_id = p.createCollisionShape(p.GEOM_CYLINDER, radius=0.05, height=length)
        # visual_shape_id = p.createVisualShape(shapeType=p.GEOM_CYLINDER,
        #                                       radius=0.05, length=length)

        base_position, base_orientation = self.workspace2world(pos, quat)
        body_id = self.load_obj(obj_path, scaling=1.0, position=base_position,
                                orientation=base_orientation.as_vector("xyzw"), visual_path=obj_path)

        # mass = 1.0
        # body_id = p.createMultiBody(mass, col_box_id, visual_shape_id,
        #                             base_position, base_orientation.as_vector("xyzw"))
        # p.changeDynamics(body_id, -1, lateralFriction=1.0, spinningFriction=1.0,
        #                  rollingFriction=0.0001, frictionAnchor=True)
        for i in range(1000):
            p.stepSimulation()

        return Object(name=obj_path.split('/')[-1].split('.')[0],
                      pos=base_position,
                      quat=base_orientation,
                      body_id=body_id)

    def add_objects(self):
        bounds = np.array([[-0.25, 0.25], [-0.25, 0.25], [0.01, 0.3]])

        def get_pxl_distance(meters, pxl_size=0.005):
            return meters / pxl_size

        def get_xyz(pxl, size, bounds, pxl_size=0.005):
            x = -(pxl_size * pxl[0] - bounds[0][1])
            y = pxl_size * pxl[1] - bounds[1][1]
            z = size[2] / 2.0
            return np.array([x, y, z])

        # Sample n objects from the database.
        nr_objects = self._random.randint(low=2, high=3)
        obj_paths = self._random.choice(self.scene_generator.obj_files, nr_objects)

        for i in range(len(obj_paths)):
            obj = Object()
            base_position, base_orientation = self.workspace2world(np.array([1.0, 1.0, 0.0]), Quaternion())
            body_id = self.load_obj(obj_paths[i], scaling=1.0, position=base_position,
                                    orientation=base_orientation.as_vector("xyzw"), visual_path=obj_paths[i])
            obj.body_id = body_id
            size = (np.array(p.getAABB(body_id)[1]) - np.array(p.getAABB(body_id)[0])) / 2.0
            obj.size = size

            max_size = np.sqrt(obj.size[0] ** 2 + obj.size[1] ** 2)
            erode_size = int(np.round(get_pxl_distance(meters=max_size)))

            obs = self.get_obs()
            state = utils.get_fused_heightmap(obs, cameras.RealSense.CONFIG, bounds, 0.005)

            free = np.zeros(state.shape, dtype=np.uint8)
            free[state == 0] = 1
            free[0, :], free[:, 0], free[-1, :], free[:, -1] = 0, 0, 0, 0
            free = cv2.erode(free, np.ones((erode_size, erode_size), np.uint8))

            if np.sum(free) == 0:
                return
            pixx = utils.sample_distribution(np.float32(free), self._random)
            pix = np.array([pixx[1], pixx[0]])

            # plt.imshow(free)
            # plt.plot(pix[0], pix[1], 'ro')
            # plt.show()

            obj.pos = get_xyz(pix, size, bounds=bounds)
            theta = self._random.rand() * 2 * np.pi
            obj.quat = Quaternion().from_rotation_matrix(rot_z(theta))

            p.removeBody(body_id)

            self.objects.append(self.add_object(obj_paths[i], obj.pos, obj.quat))
            # self.add_box(obj)

    def is_static(self):
        """
        Checks if the objects are still moving
        """
        for obj in self.objects:

            pos, quat = p.getBasePositionAndOrientation(bodyUniqueId=obj.body_id)
            if pos[2] < 0:
                continue

            vel, rot_vel = p.getBaseVelocity(bodyUniqueId=obj.body_id)
            norm_1 = np.linalg.norm(vel)
            norm_2 = np.linalg.norm(rot_vel)
            if norm_1 > 0.001 or norm_2 > 0.1:
                return True
        return False

    def hug(self, force_magnitude=10, duration=2000):
        """
        Move objects towards the workspace center by applying a constant force.
        """
        t = 0
        while t < duration:
            for obj in self.objects:

                pos, quat = p.getBasePositionAndOrientation(bodyUniqueId=obj.body_id)

                # Ignore objects that have fallen of the table
                if pos[2] < 0:
                    continue

                error = self.workspace2world(np.array([0.0, 0.0, 0.0]))[0] - pos
                error[2] = 0.0
                force_direction = error / np.linalg.norm(error)
                p.applyExternalForce(obj.body_id, -1, force_magnitude * force_direction,
                                     np.array([pos[0], pos[1], 0.0]), p.WORLD_FRAME)

            p.stepSimulation()
            t += 1

        for obj in self.objects:
            pos, quat = p.getBasePositionAndOrientation(bodyUniqueId=obj.body_id)
            obj.pos, obj.quat = self.workspace2world(pos=np.array(pos),
                                                     quat=Quaternion(x=quat[0], y=quat[1], z=quat[2], w=quat[3]),
                                                     inv=True)

    def seed(self, seed):
        self._random = np.random.RandomState(seed)
        self.scene_generator.seed(seed)
        return seed

    def reset(self):

        self.objects = []
        p.resetSimulation()
        p.setGravity(0, 0, -9.8)

        # Temporarily disable rendering to load scene faster.
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)

        # Load UR5 robot arm equipped with Barrett hand.
        # robot_id = pybullet_utils.load_urdf(p, UR5_URDF_PATH, flags=p.URDF_USE_SELF_COLLISION)
        # self.robot = Ur5Bhand(robot_id=robot_id)
        self.bhand = FloatingBHand('../assets/robot_hands/barrett/bh_282.urdf',
                                   np.array([0.7, 0.0, 0.2]))

        # Load plane and workspace.
        pybullet_utils.load_urdf(p, PLANE_URDF_PATH, [0, 0, -0.7])
        table_id = pybullet_utils.load_urdf(p, UR5_WORKSPACE_URDF_PATH, self.workspace_pos)
        p.changeDynamics(table_id, -1, lateralFriction=0.1)

        # Re-enable rendering.
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)

        # Generate a scene with randomly placed.
        # self.scene_generator.reset(self)
        self.add_objects()

        self.remove_flats()

        # Pack objects.
        self.hug(force_magnitude=1)

        for t in range(3000):
            p.stepSimulation()

        while self.is_static():
            time.sleep(0.001)
            p.stepSimulation()

        return self.get_obs()

    def step(self, action):
        # Move to pre-grasp position.
        pre_grasp_pos = action['pos'].copy()
        pre_grasp_pos[2] += 0.3
        self.bhand.move(pre_grasp_pos, action['quat'], duration=0.1)

        # Set finger configuration.
        theta = action['aperture']
        self.bhand.move_fingers([0.0, theta, theta, theta], duration=.1, force=5)

        is_in_contact = self.bhand.move(action['pos'], action['quat'], duration=.5, stop_at_contact=True)

        if not is_in_contact:
            # Push the hand forward.
            rot = action['quat'].rotation_matrix()
            grasp_pos = action['pos'] + rot[0:3, 2] * action['push_distance']
            self.bhand.move(grasp_pos, action['quat'], duration=2)

            # Close the fingers.
            self.bhand.close()
        else:
            grasp_pos = action['pos']

        # Move up.
        final_pos = grasp_pos.copy()
        final_pos[2] += 0.3
        self.bhand.move(final_pos, action['quat'], duration=.1)

        # Check grasp stability.
        self.bhand.move(final_pos, action['quat'], duration=0.5)
        stable_grasp, num_contacts = self.bhand.is_grasp_stable()

        # Move home
        self.bhand.move(self.bhand.home_position, action['quat'], duration=.1)

        # Open fingers
        self.bhand.open()

        return self.get_obs(), {'collision': is_in_contact,
                                'stable': stable_grasp,
                                'num_contacts': num_contacts}

    def close(self):
        p.disconnect()

    def get_obs(self):
        obs = {'color': [], 'depth': [], 'seg': [], 'full_state': []}

        for cam in self.agent_cams:
            color, depth, seg = cam.get_data()
            obs['color'] += (color,)
            obs['depth'] += (depth,)
            obs['seg'] += (seg,)

        # Update position and orientation
        for obj in self.objects:
            pos, quat = p.getBasePositionAndOrientation(bodyUniqueId=obj.body_id)
            obj.pos, obj.quat = self.workspace2world(pos=np.array(pos),
                                                     quat=Quaternion(x=quat[0], y=quat[1], z=quat[2], w=quat[3]),
                                                     inv=True)
        obs['full_state'] = self.objects

        return obs

    def remove_flats(self):
        is_terminal = True

        tmp_objects = []
        for obj in self.objects:
            obj_pos, obj_quat = p.getBasePositionAndOrientation(obj.body_id)

            rot_mat = Quaternion(x=obj_quat[0], y=obj_quat[1], z=obj_quat[2], w=obj_quat[3]).rotation_matrix()
            angle_z = np.arccos(np.dot(np.array([0, 0, 1]), rot_mat[0:3, 2]))

            if obj_pos[2] < 0 or np.abs(angle_z) > 0.1:
                p.removeBody(obj.body_id)
                continue
            tmp_objects.append(obj)

        self.objects = tmp_objects

    def get_state_id(self):
        return p.saveState()

    def restore_state(self, state_id):
        p.restoreState(stateId=state_id)

