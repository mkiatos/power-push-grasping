import numpy as np
import pybullet as p


"""Camera configs."""


class RealSense():
  """Default configuration with 2 RealSense RGB-D cameras."""

  image_size = (480, 640)
  intrinsics = (450., 0, 320., 0, 450., 240., 0, 0, 1)

  # Set default camera poses. (w.r.t. workspace center)
  front_position = np.array([0.0, 0.5, 0.5])
  front_target_pos = np.array([0.0, 0.0, 0.0])
  front_up_vector = np.array([0.0, 0.0, 1.0])

  top_position = np.array([0.0, 0.0, 0.5])
  top_target_pos = np.array([0.0, 0.0, 0.0])
  top_up_vector = np.array([0.0, -1.0, 0.0])


  # Default camera configs.
  CONFIG = [{
      'image_size': image_size,
      'intrinsics': intrinsics,
      'pos': front_position,
      'target_pos': front_target_pos,
      'up_vector': front_up_vector,
      'zrange': (0.01, 10.),
  }, {
      'image_size': image_size,
      'intrinsics': intrinsics,
      'pos': top_position,
      'target_pos': top_target_pos,
      'up_vector': top_up_vector,
      'zrange': (0.01, 10.),
  }]