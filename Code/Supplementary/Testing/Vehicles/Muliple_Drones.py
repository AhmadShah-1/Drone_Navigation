import airsim
import time

# Connect to the AirSim simulator
client = airsim.MultirotorClient()
client.confirmConnection()

# List of drone names
drones = ["Drone1", "Drone2"]

# Initialize drones
for drone in drones:
    client.enableApiControl(True, drone)
    client.armDisarm(True, drone)
    client.takeoffAsync(vehicle_name=drone).join()

# Function to control drones
def control_drones(client, drones):
    # Move each drone to a different position
    client.moveToPositionAsync(10, 10, -10, 5, vehicle_name="Drone1").join()
    client.moveToPositionAsync(-10, -10, -10, 5, vehicle_name="Drone2").join()

    # Example: Move Drone1 forward and Drone2 backward
    client.moveByVelocityAsync(5, 0, 0, 5, vehicle_name="Drone1").join()
    client.moveByVelocityAsync(-5, 0, 0, 5, vehicle_name="Drone2").join()

# Control the drones
control_drones(client, drones)

# Land drones
for drone in drones:
    client.landAsync(vehicle_name=drone).join()
    client.armDisarm(False, drone)
    client.enableApiControl(False, drone)

print("Drones have been controlled successfully.")
