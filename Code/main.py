import airsim
import time

# Connect to the AirSim simulator
client = airsim.MultirotorClient()
client.confirmConnection()
client.enableApiControl(True)
client.armDisarm(True)

# Takeoff
client.takeoffAsync().join()

# Fly in a square path
side_length = 10
for i in range(4):
    client.moveToPositionAsync(side_length, 0, -10, 5).join()
    client.rotateByYawRateAsync(90, 1).join()

# Land
client.landAsync().join()

# Disable API control and disarm
client.armDisarm(False)
client.enableApiControl(False)

print("Script finished!")
