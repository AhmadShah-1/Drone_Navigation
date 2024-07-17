import airsim
import numpy as np
import cv2

# Connect to the AirSim simulator
client = airsim.MultirotorClient()
client.confirmConnection()
client.enableApiControl(True)
client.armDisarm(True)

# Take off and move up
client.takeoffAsync().join()
client.moveToPositionAsync(0, 0, 0, 5).join()

# Set object ID for the street (ground)
street_object_id = 42  # This should be the ID for the ground
client.simSetSegmentationObjectID("Road", street_object_id)
client.simSetSegmentationObjectID("Road[\w]*", street_object_id, True)
client.simSetSegmentationObjectID("Road_[\w]*", street_object_id, True)
client.simSetSegmentationObjectID("road[\w]*", street_object_id, True)

# Capture the segmentation image from the bottom camera
segmentation_request = airsim.ImageRequest("bottom_center", airsim.ImageType.Segmentation, False, False)
responses = client.simGetImages([segmentation_request])
response = responses[0]

# Process the segmentation image
img1d = np.frombuffer(response.image_data_uint8, dtype=np.uint8)
img2d = img1d.reshape(response.height, response.width, 3)
img2d = np.flipud(img2d)

# Create the expected color image based on the street object ID
expected_color_image = np.full((response.height, response.width, 3), street_object_id, dtype=np.uint8)

# Extract the center pixel to determine if the drone is over the street
center_pixel = img2d[response.height // 2, response.width // 2]
print(center_pixel)

# Check if the center pixel matches the street object ID
if np.array_equal(center_pixel, [street_object_id] * 3):
    print("Drone is over the street.")
else:
    print("Drone is not over the street.")

# Display the segmentation image and the expected color side by side
combined_image = np.hstack((img2d, expected_color_image))
cv2.imshow("Bottom Segmentation Camera Feed (Left) and Expected Color (Right)", combined_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
