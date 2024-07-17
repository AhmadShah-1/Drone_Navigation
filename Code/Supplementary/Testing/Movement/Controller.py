import airsim
import numpy as np
import cv2
import time
import keyboard  # For keyboard controls

# Connect to the AirSim simulator
client = airsim.MultirotorClient()
client.confirmConnection()
client.enableApiControl(True)
client.armDisarm(True)

# Take off and move up
client.takeoffAsync().join()
client.moveToPositionAsync(4, 10, -10, 5).join()

# Set object ID for the street (ground)
street_object_id = 42  # This should be the ID for the ground
client.simSetSegmentationObjectID("Road", street_object_id)
client.simSetSegmentationObjectID("Road[\w]*", street_object_id, True)
client.simSetSegmentationObjectID("Road_[\w]*", street_object_id, True)
client.simSetSegmentationObjectID("road[\w]*", street_object_id, True)

# Function to get the segmentation object ID at the drone's position
def get_object_id_beneath_drone():
    segmentation_request = airsim.ImageRequest("bottom_center", airsim.ImageType.Segmentation, False, False)
    response = client.simGetImages([segmentation_request])[0]

    # Process the segmentation image
    img1d = np.frombuffer(response.image_data_uint8, dtype=np.uint8)
    img2d = img1d.reshape(response.height, response.width, 3)
    img2d = np.flipud(img2d)

    # Get the object ID of the center pixel
    center_pixel = img2d[response.height // 2, response.width // 2]
    object_id = center_pixel[0]  # Assuming object ID is stored in the red channel
    return object_id

# Main loop to control the drone
try:
    while True:
        if keyboard.is_pressed('w'):
            client.moveByVelocityAsync(1, 0, 0, 5).join()
        if keyboard.is_pressed('s'):
            client.moveByVelocityAsync(-1, 0, 0, 5).join()
        if keyboard.is_pressed('a'):
            client.moveByVelocityAsync(0, 1, 0, 5).join()
        if keyboard.is_pressed('d'):
            client.moveByVelocityAsync(0, -1, 0, 5).join()
        if keyboard.is_pressed('up'):
            client.moveByVelocityAsync(0, 0, -1, 5).join()
        if keyboard.is_pressed('down'):
            client.moveByVelocityAsync(0, 0, 1, 5).join()

        # Check if the drone is above the road
        object_id = get_object_id_beneath_drone()
        print(f"Segmentation ID at bottom center: {object_id}")
        if object_id == street_object_id:
            print("Drone is over the street.")
        else:
            print("Drone is not over the street.")

        time.sleep(0.1)  # Small delay to avoid excessive CPU usage

except KeyboardInterrupt:
    print("Control interrupted by user.")

finally:
    client.armDisarm(False)
    client.enableApiControl(False)
