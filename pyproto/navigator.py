from typing import List
import numpy as np
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.animation import FuncAnimation

class Navigator:
  colors = {
    "Area": "red",
    "Oasis": "blue",
    "KIZ": "green"
  }
  alpha = 0.2

  areas = {
    "Area 1": (10.42, -10.58, 4.82, 11.48, -10.58, 5.57),
    "Area 2": (10.3, -9.25, 3.76203, 11.55, -8.5, 3.76203),
    "Area 3": (10.3, -8.4, 3.76093, 11.55, -7.45, 3.76093),
    "Area 4": (9.866984, -7.34, 4.32, 9.866984, -6.365, 5.57),
    "Oasis 1": (10.425, -10.2, 4.445, 11.425, -9.5, 4.945),
    "Oasis 2": (10.925, -9.5, 4.945, 11.425, -8.45, 5.445),
    "Oasis 3": (10.425, -8.45, 4.945, 10.975, -7.4, 5.445),
    "Oasis 4": (10.925, -7.4, 4.425, 11.425, -6.35, 4.945),
    "KIZ 1": (10.3, -10.2, 4.32, 11.55, -6.0, 5.57),
    "KIZ 2": (9.5, -10.5, 4.02, 10.5, -9.6, 4.8)
  }

  poses = {
    "dock": (9.815, -9.806, 4.293, 1, 0, 0, 0),
    "patrol1":(10.95, -10, 4.8, 0.0, -0.398, -0.584, 0.707),
    "patrol2":(10.925, -8.875, 4.462, 0, 0.707, 0, 0.707),
    "patrol3":(10.925, -7.925, 4.462, 0, 0.707, 0, 0.707),
    "patrol4":(10.567, -6.853, 4.945, 0, 0, 1, 0),
    "report": (11.143, -6.7607, 4.9654, 0, 0, 0.707, 0.707),
  }

  def __init__(self, ax):
    self.ax = ax

  def generateTrajectory(self, currentPose: List[float], targetPose: List[float]) -> List[List[float]]:
    """
    a star

    :param current_pose: current pose of the robot
    :param target_pose: target pose of the robot
    :return: trajectory
    """
    pass

  def getQuaternionToFacePoint(self, x1: float, y1: float, z1: float, x2: float, y2: float, z2: float):
    """
    Calculate quaternion to orient from current point (x1, y1, z1) to target point (x2, y2, z2).
    Returns quaternion as [qx, qy, qz, qw].
    """
    # Calculate direction vector
    vx = x2 - x1
    vy = y2 - y1
    vz = z2 - z1

    # Normalize direction vector
    magnitude = math.sqrt(vx**2 + vy**2 + vz**2)
    if magnitude == 0:
      return [0, 0, 0, 1]  # Same point, return identity quaternion

    vx /= magnitude
    vy /= magnitude
    vz /= magnitude

    # Default direction (e.g., positive z-axis)
    ux, uy, uz = [1, 0, 0]  # Default direction vector

    # Dot product to find angle
    dot = ux * vx + uy * vy + uz * vz  # u · v
    theta = math.acos(dot)  # Angle in radians

    # Special cases
    if abs(dot - 1) < 1e-6:  # Already aligned with default direction
      return [0, 0, 0, 1]
    elif abs(dot + 1) < 1e-6:  # 180° opposite to default direction
      # Choose an arbitrary perpendicular axis (e.g., x-axis)
      return [1, 0, 0, 0] if abs(ux) < 0.9 else [0, 1, 0, 0]

    # Cross product to find rotation axis
    axisX = uy * vz - uz * vy
    axisY = uz * vx - ux * vz
    axisZ = ux * vy - uy * vx
    axisMagnitude = math.sqrt(axisX**2 + axisY**2 + axisZ**2)
    
    # Avoid division by zero
    if axisMagnitude < 1e-6:
      return [0, 0, 0, 1]  # Shouldn't happen due to prior checks, but safeguard
    
    axisX /= axisMagnitude
    axisY /= axisMagnitude
    axisZ /= axisMagnitude

    # Calculate quaternion
    halfTheta = theta / 2
    sinHalfTheta = math.sin(halfTheta)
    qx = axisX * sinHalfTheta
    qy = axisY * sinHalfTheta
    qz = axisZ * sinHalfTheta
    qw = math.cos(halfTheta)

    return [qx, qy, qz, qw]
  
  def interplate(self, start: List[float], end: List[float]) -> List[List[float]]:
    linearUnit = 0.08

    totalDistance = math.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2 + (end[2] - start[2])**2)
    numSteps = int(totalDistance / linearUnit)
    unitStep = np.array([(end[0] - start[0]) / numSteps, (end[1] - start[1]) / numSteps, (end[2] - start[2]) / numSteps])

    trajectory = []
    current = np.array([start[0], start[1], start[2]])
    start = np.array([start[0], start[1], start[2], start[3], start[4], start[5], start[6]])
    while numSteps > 0:
      current = current + unitStep
      newPose = np.concatenate((current, start[3:]))
      trajectory.append(newPose)
      numSteps -= 1
    return trajectory
  
  def plot(self):
    # Plot areas
    for label, area in self.areas.items():
      self.draw_area(area, label, self.colors[label.split()[0]])
    
    # Plot poses
    for label, pose in self.poses.items():
      draw_point(self.ax, pose, label)

    # x = [pose[0] for pose in self.poses.values()]
    # y = [pose[1] for pose in self.poses.values()]
    # z = [pose[2] for pose in self.poses.values()]
    # self.ax.plot(x, y, z, color='black', linewidth=2)

    # Draw the trajectory
    trajectory = self.interplate(self.poses["dock"], self.poses["patrol1"])
    for i in range(len(trajectory) - 1):
      self.ax.scatter(trajectory[i][0], trajectory[i][1], trajectory[i][2], color='black', s=10)

    trajectory = self.interplate(self.poses["patrol1"], self.poses["patrol2"])
    for i in range(len(trajectory) - 1):
      self.ax.scatter(trajectory[i][0], trajectory[i][1], trajectory[i][2], color='black', s=10)

    trajectory = self.interplate(self.poses["patrol2"], self.poses["patrol3"])
    for i in range(len(trajectory) - 1):
      self.ax.scatter(trajectory[i][0], trajectory[i][1], trajectory[i][2], color='black', s=10)

    trajectory = self.interplate(self.poses["patrol3"], self.poses["patrol4"])
    for i in range(len(trajectory) - 1):
      self.ax.scatter(trajectory[i][0], trajectory[i][1], trajectory[i][2], color='black', s=10)

    trajectory = self.interplate(self.poses["patrol4"], self.poses["report"])
    for i in range(len(trajectory) - 1):
      self.ax.scatter(trajectory[i][0], trajectory[i][1], trajectory[i][2], color='black', s=10)

  def draw_area(self, area, label, color='cyan'):
    x_min, y_min, z_min, x_max, y_max, z_max = area[0], area[1], area[2], area[3], area[4], area[5]
    verts1 = [[(x_min, y_min, z_min), (x_max, y_min, z_min), (x_max, y_max, z_min), (x_min, y_max, z_min)]]
    verts2 = [[(x_min, y_min, z_min), (x_min, y_max, z_min), (x_min, y_max, z_max), (x_min, y_min, z_max)]]
    verts3 = [[(x_min, y_min, z_min), (x_min, y_min, z_max), (x_max, y_min, z_max), (x_max, y_min, z_min)]]
    verts4 = [[(x_max, y_max, z_max), (x_min, y_max, z_max), (x_min, y_min, z_max), (x_max, y_min, z_max)]]
    verts5 = [[(x_max, y_max, z_max), (x_max, y_min, z_max), (x_max, y_min, z_min), (x_max, y_max, z_min)]]
    verts6 = [[(x_max, y_max, z_max), (x_max, y_max, z_min), (x_min, y_max, z_min), (x_min, y_max, z_max)]]
    self.ax.add_collection3d(Poly3DCollection(verts1, color=color, alpha=self.alpha))
    self.ax.add_collection3d(Poly3DCollection(verts2, color=color, alpha=self.alpha))
    self.ax.add_collection3d(Poly3DCollection(verts3, color=color, alpha=self.alpha))
    self.ax.add_collection3d(Poly3DCollection(verts4, color=color, alpha=self.alpha))
    self.ax.add_collection3d(Poly3DCollection(verts5, color=color, alpha=self.alpha))
    self.ax.add_collection3d(Poly3DCollection(verts6, color=color, alpha=self.alpha))

    # Draw the tag
    # x_center = (x_min + x_max) / 2
    # y_center = (y_min + y_max) / 2
    # z_center = (z_min + z_max) / 2
    # self.ax.text(x_center, y_center, z_center, label, color='black', fontsize=10, ha='center')
    # print("Center of area:", label, (x_center, y_center, z_center))

def draw_point(ax, point, label):
  x, y, z, qx, qy, qz, qw = point[0], point[1], point[2], point[3], point[4], point[5], point[6]
  ax.scatter(x, y, z, label=f"{label}", s=50)
  
  # Convert quaternion to direction vector
  direction = np.array([1, 0, 0])  # Default direction vector (along x-axis)
  
  # Quaternion to Rotation Matrix
  q = np.array([qw, qx, qy, qz])
  R = np.array([
    [1 - 2 * (qy**2 + qz**2), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
    [2 * (qx * qy + qz * qw), 1 - 2 * (qx**2 + qz**2), 2 * (qy * qz - qx * qw)],
    [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx**2 + qy**2)]
  ])

  # Rotate the direction vector using the quaternion (i.e., applying the rotation matrix)
  rotated_direction = np.dot(R, direction)
  
  # Plot the arrow representing orientation (from the point with the direction vector)
  ax.quiver(x, y, z, rotated_direction[0], rotated_direction[1], rotated_direction[2], length=0.5, normalize=True, color='r', linewidth=2)
  # ax.text(x, y, z, label, color='black', fontsize=10, ha='center')


fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

navigator = Navigator(ax)
navigator.plot()

def set_axes_equal(ax):
  x_limits = ax.get_xlim()
  y_limits = ax.get_ylim()
  z_limits = ax.get_zlim()
  
  x_middle = np.mean(x_limits)
  y_middle = np.mean(y_limits)
  z_middle = np.mean(z_limits)

  max_range = max(
    (x_limits[1] - x_limits[0]),
    (y_limits[1] - y_limits[0]),
    (z_limits[1] - z_limits[0])
  ) / 2.0

  ax.set_xlim(x_middle - max_range, x_middle + max_range)
  ax.set_ylim(y_middle - max_range, y_middle + max_range)
  ax.set_zlim(z_middle - max_range, z_middle + max_range)

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.view_init(elev=200, azim=0)

# Function to update the view angle for animation
def update(frame):
  ax.view_init(elev=200, azim=frame*0.5)
  return []
ani = FuncAnimation(fig, update, frames=np.arange(0, 360, 2), interval=50, blit=False)

set_axes_equal(ax)
fig.tight_layout()
plt.show()
