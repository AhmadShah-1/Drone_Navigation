import airsim
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from PIL import Image
import time


class AirSimDroneEnv(gym.Env):
    def __init__(self, ip_address=""):
        super(AirSimDroneEnv, self).__init__()

        self.client = airsim.MultirotorClient(ip=ip_address)
        self.client.confirmConnection()
        self.client.enableApiControl(True)
        self.client.armDisarm(True)
        self.client.takeoffAsync().join()

        # Define action and observation space
        self.action_space = spaces.Discrete(7)  # 6 directions + hover
        self.observation_space = spaces.Box(low=0, high=255, shape=(84, 84, 3), dtype=np.uint8)

        self.image_request = airsim.ImageRequest("0", airsim.ImageType.DepthPerspective, True)
        self.segmentation_request = airsim.ImageRequest("0", airsim.ImageType.Segmentation, False, False)
        self.target_position = np.array([0, 0, -10])  # Set your target position

        self.max_duration = 60  # Maximum duration of each episode in seconds

    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
        self.client.reset()
        self.client.enableApiControl(True)
        self.client.armDisarm(True)
        self.client.takeoffAsync().join()
        self.start_time = time.time()  # Initialize the timer
        return self._get_obs(), {}

    def step(self, action):
        quad_offset = {
            0: (1, 0, 0),  # forward
            1: (-1, 0, 0),  # backward
            2: (0, 1, 0),  # right
            3: (0, -1, 0),  # left
            4: (0, 0, -1),  # up
            5: (0, 0, 1),  # down
            6: (0, 0, 0),  # hover
        }[action]

        self.client.moveByVelocityAsync(quad_offset[0], quad_offset[1], quad_offset[2], 5).join()

        obs = self._get_obs()
        reward = self._compute_reward()
        done = self._is_done()

        info = {}
        return obs, reward, done, False, info

    def _get_obs(self):
        responses = self.client.simGetImages([self.image_request])
        response = responses[0]
        img1d = np.array(response.image_data_float, dtype=float)
        img2d = np.reshape(img1d, (response.height, response.width))
        image = np.invert(np.array(Image.fromarray(img2d).resize((84, 84)).convert('L')))
        image = np.stack((image,) * 3, axis=-1)
        return image

    def _compute_reward(self):
        current_position = self.client.getMultirotorState().kinematics_estimated.position
        distance_to_target = np.linalg.norm(
            np.array([current_position.x_val, current_position.y_val, current_position.z_val]) - self.target_position)

        reward = -distance_to_target  # Negative reward for distance to target

        collision = self.client.simGetCollisionInfo().has_collided
        if collision:
            reward -= 100  # Large penalty for collisions

        # Penalize for being off the street
        if not self._is_on_street():
            reward -= 10  # Adjust penalty value as needed

        # Penalize for flying too high
        max_altitude = 15  # Define the maximum altitude
        if current_position.z_val < -max_altitude:  # Remember that in AirSim, z is negative upwards
            reward -= 10  # Adjust penalty value as needed

        return reward

    def _is_done(self):
        current_position = self.client.getMultirotorState().kinematics_estimated.position
        distance_to_target = np.linalg.norm(
            np.array([current_position.x_val, current_position.y_val, current_position.z_val]) - self.target_position)
        collision = self.client.simGetCollisionInfo().has_collided
        elapsed_time = time.time() - self.start_time  # Calculate elapsed time

        # Check if time limit has been exceeded
        if collision or distance_to_target < 1 or elapsed_time > self.max_duration:
            return True
        return False

    def _is_on_street(self):
        responses = self.client.simGetImages([self.segmentation_request])
        response = responses[0]
        img1d = np.frombuffer(response.image_data_uint8, dtype=np.uint8)
        img2d = img1d.reshape(response.height, response.width, 3)

        # Check if the central part of the image is the road color
        road_color = [128, 64, 128]  # Example color for the road, this may vary
        center_pixel = img2d[response.height // 2, response.width // 2]

        if np.array_equal(center_pixel, road_color):
            return True
        return False
