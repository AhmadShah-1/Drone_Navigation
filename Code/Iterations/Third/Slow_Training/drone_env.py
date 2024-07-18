import airsim
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from PIL import Image
import time
import cv2

class AirSimDroneEnv(gym.Env):
    def __init__(self, ip_address="", drone_names=["Drone1", "Drone2"], target_positions=None):
        super(AirSimDroneEnv, self).__init__()

        self.client = airsim.MultirotorClient(ip=ip_address)
        self.client.confirmConnection()

        self.drone_names = drone_names
        for drone_name in self.drone_names:
            self.client.enableApiControl(True, drone_name)
            self.client.armDisarm(True, drone_name)
            self.client.takeoffAsync(vehicle_name=drone_name).join()

        self.action_space = spaces.Discrete(9)  # 6 directions + hover + yaw_left + yaw_right
        self.observation_space = spaces.Box(low=0, high=255, shape=(84, 84, 3), dtype=np.uint8)

        self.image_request = airsim.ImageRequest("0", airsim.ImageType.DepthPerspective, True)
        self.segmentation_request = airsim.ImageRequest("bottom_center", airsim.ImageType.Segmentation, False, False)
        self.bottom_camera_request = airsim.ImageRequest("bottom_center", airsim.ImageType.Scene, False, False)

        self.max_duration = 120  # Maximum duration of each episode in seconds
        self.ground_not_detected_start = {drone_name: None for drone_name in self.drone_names}

        self.target_positions = target_positions if target_positions else [np.array([30, 3, -10]) for _ in self.drone_names]

        self.road_object_id = 42  # Set this to an appropriate ID for the road
        self.client.simSetSegmentationObjectID("Road", self.road_object_id)
        self.client.simSetSegmentationObjectID("road[\w]*", self.road_object_id, True)
        self.client.simSetSegmentationObjectID("Road[\w]*", self.road_object_id, True)
        self.client.simSetSegmentationObjectID("Road_[\w]*", self.road_object_id, True)

    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
        self.client.reset()
        for drone_name in self.drone_names:
            self.client.enableApiControl(True, drone_name)
            self.client.armDisarm(True, drone_name)
            self.client.takeoffAsync(vehicle_name=drone_name).join()
        self.start_time = time.time()  # Initialize the timer
        self.ground_not_detected_start = {drone_name: None for drone_name in self.drone_names}
        observation = self._get_obs(self.drone_names[0])  # Return the observation for the first drone as an example
        return observation, {}  # Return observation and an empty dictionary

    # Ensure that the step method returns the correct format for a single drone as well
    def step(self, action):
        drone_name = self.drone_names[0]  # For simplicity, using the first drone
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
            self.client.rotateByYawRateAsync(yaw_rate, 1, vehicle_name=drone_name).join()
        else:
            self.client.moveByVelocityAsync(quad_offset[0], quad_offset[1], quad_offset[2], 5,
                                            vehicle_name=drone_name).join()

        observation = self._get_obs(drone_name)
        reward = self._compute_reward(drone_name)
        done, reason = self._is_done(drone_name)
        truncated = False  # Set to True if the episode is truncated due to a time limit or other conditions

        return observation, reward, done, truncated, {"done_reason": reason}

    def _get_obs(self, drone_name):
        responses = self.client.simGetImages([self.image_request], vehicle_name=drone_name)
        response = responses[0]
        img1d = np.array(response.image_data_float, dtype=float)
        img2d = np.reshape(img1d, (response.height, response.width))
        image = np.invert(np.array(Image.fromarray(img2d).resize((84, 84)).convert('L')))
        image = np.stack((image,) * 3, axis=-1)
        return image

    def _compute_reward(self, drone_name):
        current_position = self.client.getMultirotorState(vehicle_name=drone_name).kinematics_estimated.position
        distance_to_target = np.linalg.norm(
            np.array([current_position.x_val, current_position.y_val, current_position.z_val]) - self.target_positions[self.drone_names.index(drone_name)])

        reward = -distance_to_target  # Negative reward for distance to target
        print(distance_to_target)

        collision = self.client.simGetCollisionInfo(vehicle_name=drone_name).has_collided
        if collision:
            reward -= 100  # Large penalty for collisions

        if not self._is_on_road(drone_name):
            if self.ground_not_detected_start[drone_name] is None:
                self.ground_not_detected_start[drone_name] = time.time()
            elif time.time() - self.ground_not_detected_start[drone_name] > 1:
                reward -= 10  # Penalty for not detecting the ground for over 1 second
        else:
            self.ground_not_detected_start[drone_name] = None  # Reset timer if ground is detected

        max_altitude = 15  # Define the maximum altitude
        if current_position.z_val < -max_altitude:
            reward -= 10  # Adjust penalty value as needed

        print("Reward: ", reward, " DroneName: ", drone_name)
        return reward

    def _is_done(self, drone_name):
        current_position = self.client.getMultirotorState(vehicle_name=drone_name).kinematics_estimated.position
        distance_to_target = np.linalg.norm(
            np.array([current_position.x_val, current_position.y_val, current_position.z_val]) - self.target_positions[self.drone_names.index(drone_name)])
        collision = self.client.simGetCollisionInfo(vehicle_name=drone_name).has_collided
        elapsed_time = time.time() - self.start_time  # Calculate elapsed time

        if collision:
            return True, "collision"
        if distance_to_target < 1:
            return True, "reached_target"
        if elapsed_time > self.max_duration:
            return True, "time_exceeded"
        return False, "none"

    def _is_on_road(self, drone_name):
        responses = self.client.simGetImages([self.segmentation_request], vehicle_name=drone_name)
        response = responses[0]
        img1d = np.frombuffer(response.image_data_uint8, dtype=np.uint8)  # get numpy array
        img2d = img1d.reshape(response.height, response.width, 3)  # reshape to 3-channel image

        # Debug output for the center pixel's RGB values
        center_pixel = img2d[response.height // 2, response.width // 2]

        # Check if the road object ID (42) is in the unique IDs of any channel
        return ([106, 31, 92] in center_pixel)

    def render(self, mode='human'):
        for drone_name in self.drone_names:
            responses = self.client.simGetImages([self.bottom_camera_request], vehicle_name=drone_name)
            response = responses[0]
            img1d = np.frombuffer(response.image_data_uint8, dtype=np.uint8)
            img_rgb = img1d.reshape(response.height, response.width, 3)
            img_rgb = np.flipud(img_rgb)

            cv2.imshow(f"Bottom Camera Feed - {drone_name}", img_rgb)
            cv2.waitKey(1)
