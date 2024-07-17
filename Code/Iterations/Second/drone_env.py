import airsim
import numpy as np
import gymnasium as gym
from airsim import ImageRequest
from gymnasium import spaces
from PIL import Image
import time
import cv2


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
        # self.segmentation_request = airsim.ImageRequest("bottom_center", airsim.ImageType.Segmentation, False, False)
        self.bottom_camera_request = airsim.ImageRequest("bottom_center", airsim.ImageType.Scene, False, False)
        self.target_position = np.array([30, 3, -10])  # Set your target position

        self.max_duration = 80  # Maximum duration of each episode in seconds

        # Object ID was not being identified correctly by the bottom camera, however it kept outputting 106, possibly referring to the color
        # of the road, so the number 106 will be compared to the bottom camera output and if True will mean the drone is over the road

        # Set object ID for the road (ground)
        self.road_object_id = 42  # Set this to an appropriate ID for the road
        self.client.simSetSegmentationObjectID("Road", self.road_object_id)
        self.client.simSetSegmentationObjectID("road[\w]*", self.road_object_id, True)
        self.client.simSetSegmentationObjectID("Road[\w]*", self.road_object_id, True)
        self.client.simSetSegmentationObjectID("Road_[\w]*", self.road_object_id, True)

        self.ground_not_detected_start = None  # To track when ground is not detected

    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
        self.client.reset()
        self.client.enableApiControl(True)
        self.client.armDisarm(True)
        self.client.takeoffAsync().join()
        self.start_time = time.time()  # Initialize the timer
        self.ground_not_detected_start = None  # Reset ground not detected timer
        return self._get_obs(), {}

    def step(self, action):
        action = int(action)  # Convert action to integer if it is not
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

        self.client.moveByVelocityAsync(quad_offset[0], quad_offset[1], quad_offset[2], 5).join()

        obs = self._get_obs()
        reward = self._compute_reward()
        done, reason = self._is_done()

        info = {"done_reason": reason}
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

        # Penalize if the ground is not detected for over 1 second
        if not self._is_on_road():
            if self.ground_not_detected_start is None:
                self.ground_not_detected_start = time.time()
                print("Detecting Ground")
            elif time.time() - self.ground_not_detected_start > 1:
                print("Ground not detected")
                reward -= 10  # Penalty for not detecting the ground for over 1 second
        else:
            print("Ground Detected")
            self.ground_not_detected_start = None  # Reset timer if ground is detected

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
        if collision:
            return True, "collision"
        if distance_to_target < 1:
            return True, "reached_target"
        if elapsed_time > self.max_duration:
            return True, "time_exceeded"
        return False, "none"

    def _is_on_road(self):
        responses = self.client.simGetImages(
            [ImageRequest("bottom_center", airsim.ImageType.Segmentation, False, False)])
        response = responses[0]
        img1d = np.frombuffer(response.image_data_uint8, dtype=np.uint8)  # get numpy array
        img2d = img1d.reshape(response.height, response.width, 3)  # reshape to 3-channel image

        # Debug output for the center pixel's RGB values
        center_pixel = img2d[response.height // 2, response.width // 2]

        # Check if the road object ID (42) is in the unique IDs of any channel
        return ([106, 31, 92] in center_pixel)

    '''
    def _is_on_road(self):
        responses = self.client.simGetImages(
            [ImageRequest("bottom_center", airsim.ImageType.Segmentation, False, False)])
        response = responses[0]
        img1d = np.frombuffer(response.image_data_uint8, dtype=np.uint8)  # get numpy array
        img2d = img1d.reshape(response.height, response.width, 3)  # reshape to 3-channel image

        # Debug output for the center pixel's RGB values
        center_pixel = img2d[response.height // 2, response.width // 2]
        print(f"Center pixel RGB values: {center_pixel}")

        # Check if the object ID (42) is present in any channel
        unique_ids_red, counts_red = np.unique(img2d[:, :, 0], return_counts=True)
        unique_ids_green, counts_green = np.unique(img2d[:, :, 1], return_counts=True)
        unique_ids_blue, counts_blue = np.unique(img2d[:, :, 2], return_counts=True)
        print(f"Red channel unique IDs: {unique_ids_red}, counts: {counts_red}")
        print(f"Green channel unique IDs: {unique_ids_green}, counts: {counts_green}")
        print(f"Blue channel unique IDs: {unique_ids_blue}, counts: {counts_blue}")

        # Check if the road object ID (42) is in the unique IDs of any channel
        return (self.road_object_id in unique_ids_red or
                self.road_object_id in unique_ids_green or
                self.road_object_id in unique_ids_blue)

    
    def _is_on_road(self):
        responses = self.client.simGetImages([ImageRequest("bottom_center", airsim.ImageType.Segmentation, False, False)])
        response = responses[0]
        img1d = np.fromstring(response.image_data_uint8, dtype=np.uint8)  # get numpy array
        img_rgb = img1d.reshape(response.height, response.width, 3)  # reshape array to 3 channel image array H X W X 3
        img_rgb = np.flipud(img_rgb)  # original image is fliped vertically

        # find unique colors
        print(np.unique(img_rgb[:, :, 0], return_counts=True))  # red
        print(np.unique(img_rgb[:, :, 1], return_counts=True))  # green
        print(np.unique(img_rgb[:, :, 2], return_counts=True))  # blue
        '''
    '''
    def _is_on_road(self):
        responses = self.client.simGetImages([self.segmentation_request])
        response = responses[0]
        print(responses)
        
        img1d = np.frombuffer(response.image_data_uint8, dtype=np.uint8)
        img2d = img1d.reshape(response.height, response.width, 3)

        # Check if the center pixel of the bottom_center camera matches the road object ID
        center_pixel = img2d[response.height // 2, response.width // 2]
        print(center_pixel)
        return center_pixel[0] == 106

        '''
    def render(self, mode='human'):
        # Get the bottom camera feed
        responses = self.client.simGetImages([self.bottom_camera_request])
        response = responses[0]
        img1d = np.frombuffer(response.image_data_uint8, dtype=np.uint8)
        img_rgb = img1d.reshape(response.height, response.width, 3)
        img_rgb = np.flipud(img_rgb)

        # Display the image using OpenCV
        cv2.imshow("Bottom Camera Feed", img_rgb)
        cv2.waitKey(1)

    '''
    # Additional code to ensure the drone takes off and shows the live feed
    if __name__ == "__main__":
        env = AirSimDroneEnv(ip_address="127.0.0.1")
        env.client.moveToPositionAsync(0, 0, -10, 5).join()
        while True:
            env.render()
    
    '''