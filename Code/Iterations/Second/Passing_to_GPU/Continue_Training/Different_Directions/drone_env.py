import airsim
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from PIL import Image
import time
import cv2
import torch

class AirSimDroneEnv(gym.Env):
    def __init__(self, ip_address="", target_position=None, device='cpu'):
        super(AirSimDroneEnv, self).__init__()

        self.client = airsim.MultirotorClient(ip=ip_address)
        self.client.confirmConnection()
        self.client.enableApiControl(True)
        self.client.armDisarm(True)
        self.client.takeoffAsync().join()

        # Define action and observation space
        self.action_space = spaces.Discrete(9)  # 6 directions + hover + yaw_left + yaw_right
        self.observation_space = spaces.Box(low=0, high=255, shape=(84, 84, 12), dtype=np.uint8)  # 4 cameras * 3 channels

        # Define image requests for the four cameras
        self.image_requests = [
            airsim.ImageRequest("front_center", airsim.ImageType.Scene, False, False),
            airsim.ImageRequest("front_right", airsim.ImageType.Scene, False, False),
            airsim.ImageRequest("front_left", airsim.ImageType.Scene, False, False),
            airsim.ImageRequest("back_center", airsim.ImageType.Scene, False, False)
        ]

        self.max_duration = 80  # Maximum duration of each episode in seconds
        self.ground_not_detected_start = None

        self.target_position = target_position if target_position else np.array([30, 3, -10])
        self.device = torch.device(device)

        self.road_object_id = 42  # Set this to an appropriate ID for the road
        self.client.simSetSegmentationObjectID("Road", self.road_object_id)
        self.client.simSetSegmentationObjectID("road[\w]*", self.road_object_id, True)
        self.client.simSetSegmentationObjectID("Road[\w]*", self.road_object_id, True)
        self.client.simSetSegmentationObjectID("Road_[\w]*", self.road_object_id, True)

    def reset(self, seed=None, options=None, target_position=None):
        if seed is not None:
            np.random.seed(seed)
        self.client.reset()
        self.client.enableApiControl(True)
        self.client.armDisarm(True)
        self.client.takeoffAsync().join()
        self.start_time = time.time()  # Initialize the timer
        self.ground_not_detected_start = None

        # Update target position if provided
        if target_position is not None:
            self.target_position = target_position

        observation = self._get_obs()  # Return the observation for the drone
        return observation, {}  # Return observation and an empty dictionary

    def step(self, action):
        quad_offset = {
            0: (1, 0, 0),  # forward
            1: (-1, 0, 0),  # backward
            2: (0, 1, 0),  # right
            3: (0, -1, 0),  # left
            4: (0, 0, -1),  # up
            5: (0, 0, 1),  # down
            6: (0, 0, 0),  # hover
            7: 'yaw_left',  # turn left in place
            8: 'yaw_right'  # turn right in place
        }[action]

        if action in [7, 8]:
            yaw_rate = 30 if action == 8 else -30
            self.client.rotateByYawRateAsync(yaw_rate, 1).join()
        else:
            self.client.moveByVelocityAsync(quad_offset[0], quad_offset[1], quad_offset[2], 5).join()

        observation = self._get_obs()
        reward = self._compute_reward()
        done, reason = self._is_done()
        truncated = False  # Set to True if the episode is truncated due to a time limit or other conditions

        return observation, reward, done, truncated, {"done_reason": reason}

    def _get_obs(self):
        responses = self.client.simGetImages(self.image_requests)
        images = []
        for response in responses:
            img1d = np.frombuffer(response.image_data_uint8, dtype=np.uint8)
            img2d = img1d.reshape(response.height, response.width, 3)
            img_resized = np.array(Image.fromarray(img2d).resize((84, 84)))
            images.append(img_resized)

        # Stack images along the depth dimension to create a single observation
        observation = np.concatenate(images, axis=-1)

        return observation  # Return as numpy array

    def _compute_reward(self):
        current_position = self.client.getMultirotorState().kinematics_estimated.position
        distance_to_target = np.linalg.norm(
            np.array([current_position.x_val, current_position.y_val, current_position.z_val]) - self.target_position)

        reward = -distance_to_target  # Negative reward for distance to target
        print("Current Position: ", current_position)
        print("Distance to target: ", distance_to_target)

        collision = self.client.simGetCollisionInfo().has_collided
        if collision:
            reward -= 100  # Large penalty for collisions

        if not self._is_on_road():
            if self.ground_not_detected_start is None:
                self.ground_not_detected_start = time.time()
            elif time.time() - self.ground_not_detected_start > 1:
                reward -= 10  # Penalty for not detecting the ground for over 1 second
        else:
            self.ground_not_detected_start = None  # Reset timer if ground is detected

        max_altitude = 15  # Define the maximum altitude
        if current_position.z_val < -max_altitude:
            reward -= 10  # Adjust penalty value as needed

        print("Reward: ", reward)
        return reward

    def _is_done(self):
        current_position = self.client.getMultirotorState().kinematics_estimated.position
        distance_to_target = np.linalg.norm(
            np.array([current_position.x_val, current_position.y_val, current_position.z_val]) - self.target_position)
        collision = self.client.simGetCollisionInfo().has_collided
        elapsed_time = time.time() - self.start_time  # Calculate elapsed time

        if collision:
            return True, "collision"
        if distance_to_target < 1:
            return True, "reached_target"
        if elapsed_time > self.max_duration:
            return True, "time_exceeded"
        return False, "none"

    def _is_on_road(self):
        responses = self.client.simGetImages([airsim.ImageRequest("bottom_center", airsim.ImageType.Segmentation, False, False)])
        response = responses[0]
        img1d = np.frombuffer(response.image_data_uint8, dtype=np.uint8)  # get numpy array
        img2d = img1d.reshape(response.height, response.width, 3)  # reshape to 3-channel image

        # Debug output for the center pixel's RGB values
        center_pixel = img2d[response.height // 2, response.width // 2]

        # Check if the road object ID (42) is in the unique IDs of any channel
        return ([106, 31, 92] in center_pixel)

    def render(self, mode='human'):
        responses = self.client.simGetImages(self.image_requests)
        for i, response in enumerate(responses):
            img1d = np.frombuffer(response.image_data_uint8, dtype=np.uint8)
            img_rgb = img1d.reshape(response.height, response.width, 3)
            img_rgb = np.flipud(img_rgb)
            camera_name = ["Front Center", "Front Right", "Front Left", "Back Center"]
            cv2.imshow(f"{camera_name[i]} Camera Feed", img_rgb)
        cv2.waitKey(1)

    def get_obs_tensor(self):
        observation = self._get_obs()
        # Convert to PyTorch tensor and move to GPU
        observation_tensor = torch.tensor(observation, dtype=torch.float32).to(self.device)
        return observation_tensor
