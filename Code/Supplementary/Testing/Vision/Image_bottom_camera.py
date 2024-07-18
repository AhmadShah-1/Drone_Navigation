import airsim
import numpy as np
import os

# Connect to the AirSim simulator
client = airsim.MultirotorClient()
client.confirmConnection()

# Define the camera settings to capture the bottom center image
image_request = airsim.ImageRequest("bottom_center", airsim.ImageType.Scene, False, False)

# Take the image
responses = client.simGetImages([image_request])
response = responses[0]

# Save the image
filename = '/Supplementary/Output/bottom_center_image.png'
img1d = np.frombuffer(response.image_data_uint8, dtype=np.uint8)
img2d = img1d.reshape(response.height, response.width, 3)
airsim.write_png(os.path.normpath(filename), img2d)
