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
    "patrol1":(10.95, -9.88, 5.2, 0.0, 0.0, -0.707, 0.707),
    "patrol2":(10.925, -8.875, 4.462, 0.5, 0.5, -0.5, 0.5),
    "patrol3":(10.925, -7.925, 4.462, 0.5, 0.5, -0.5, 0.5),
    "patrol4":(10.57, -6.853, 4.945, 0.0, -0.707, 0.707, 0.0),
    "report": (11.143, -6.7607, 4.9654, -0.5, -0.5, 0.5, 0.5),
  }

  def __init__(self, ax):
    self.ax = ax

  def getQuaternionToFacePointWithUp(self, x1, y1, z1, x2, y2, z2, up=(0, 0, 1)):
    # Direction to target
    forward = [x2 - x1, y2 - y1, z2 - z1]
    magnitude = math.sqrt(forward[0]**2 + forward[1]**2 + forward[2]**2)
    if magnitude == 0:
        return [0, 0, 0, 1]
    forward = [f / magnitude for f in forward]

    # Right vector
    right = [
        up[1] * forward[2] - up[2] * forward[1],
        up[2] * forward[0] - up[0] * forward[2],
        up[0] * forward[1] - up[1] * forward[0]
    ]
    right_mag = math.sqrt(right[0]**2 + right[1]**2 + right[2]**2)
    if right_mag < 1e-6:
        # forward 和 up 太接近，無法產生穩定的 right
        # 可以隨便選一個新的 up 試試
        up = (0, 1, 0)
        right = [
          up[1] * forward[2] - up[2] * forward[1],
          up[2] * forward[0] - up[0] * forward[2],
          up[0] * forward[1] - up[1] * forward[0]
        ]
        right_mag = math.sqrt(right[0]**2 + right[1]**2 + right[2]**2)
    
    right = [r / right_mag for r in right]

    # Recompute true up
    true_up = [
      forward[1] * right[2] - forward[2] * right[1],
      forward[2] * right[0] - forward[0] * right[2],
      forward[0] * right[1] - forward[1] * right[0]
    ]

    # 旋轉矩陣
    rot = [
      [forward[0], right[0], true_up[0]],
      [forward[1], right[1], true_up[1]],
      [forward[2], right[2], true_up[2]],
    ]

    # 把旋轉矩陣轉成四元數
    trace = rot[0][0] + rot[1][1] + rot[2][2]
    if trace > 0:
      s = 0.5 / math.sqrt(trace + 1.0)
      qw = 0.25 / s
      qx = (rot[2][1] - rot[1][2]) * s
      qy = (rot[0][2] - rot[2][0]) * s
      qz = (rot[1][0] - rot[0][1]) * s
    elif (rot[0][0] > rot[1][1]) and (rot[0][0] > rot[2][2]):
      s = 2.0 * math.sqrt(1.0 + rot[0][0] - rot[1][1] - rot[2][2])
      qw = (rot[2][1] - rot[1][2]) / s
      qx = 0.25 * s
      qy = (rot[0][1] + rot[1][0]) / s
      qz = (rot[0][2] + rot[2][0]) / s
    elif rot[1][1] > rot[2][2]:
      s = 2.0 * math.sqrt(1.0 + rot[1][1] - rot[0][0] - rot[2][2])
      qw = (rot[0][2] - rot[2][0]) / s
      qx = (rot[0][1] + rot[1][0]) / s
      qy = 0.25 * s
      qz = (rot[1][2] + rot[2][1]) / s
    else:
      s = 2.0 * math.sqrt(1.0 + rot[2][2] - rot[0][0] - rot[1][1])
      qw = (rot[1][0] - rot[0][1]) / s
      qx = (rot[0][2] + rot[2][0]) / s
      qy = (rot[1][2] + rot[2][1]) / s
      qz = 0.25 * s

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

def draw_point(ax, point, label):
  x, y, z, qx, qy, qz, qw = point[0], point[1], point[2], point[3], point[4], point[5], point[6]
  ax.scatter(x, y, z, label=f"{label}", s=50)

  # Quaternion to Rotation Matrix
  R = np.array([
    [1 - 2 * (qy**2 + qz**2),     2 * (qx*qy - qz*qw),     2 * (qx*qz + qy*qw)],
    [    2 * (qx*qy + qz*qw), 1 - 2 * (qx**2 + qz**2),     2 * (qy*qz - qx*qw)],
    [    2 * (qx*qz - qy*qw),     2 * (qy*qz + qx*qw), 1 - 2 * (qx**2 + qy**2)]
  ])

  # Local axes: x, y, z directions
  x_axis = R[:, 0] # 第一列是 local x軸
  y_axis = R[:, 1] # 第二列是 local y軸
  z_axis = R[:, 2] # 第三列是 local z軸

  length = 0.5 # Arrow length

  # 畫三個方向的小箭頭
  ax.quiver(x, y, z, x_axis[0], x_axis[1], x_axis[2], color='r', length=length, normalize=True, linewidth=2, label=label+"_x")
  ax.quiver(x, y, z, y_axis[0], y_axis[1], y_axis[2], color='g', length=length, normalize=True, linewidth=2, label=label+"_y")
  ax.quiver(x, y, z, z_axis[0], z_axis[1], z_axis[2], color='b', length=length, normalize=True, linewidth=2, label=label+"_z")

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

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

navigator = Navigator(ax)

x1, y1, z1, x2, y2, z2 = navigator.areas["Area 4"]
x_pose, y_pose, z_pose, x_q, y_q, z_q, w_q = navigator.poses["patrol4"]
target_x = (x1 + x2) / 2
target_y = (y1 + y2) / 2
target_z = (z1 + z2) / 2
current_x = x_pose
current_y = y_pose
current_z = z_pose
print(navigator.getQuaternionToFacePointWithUp(current_x, current_y, current_z, target_x, target_y, target_z, (0, 1, 0)))

navigator.plot()

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.view_init(elev=200, azim=0)
set_axes_equal(ax)

# Function to update the view angle for animation
# def update(frame):
#   ax.view_init(elev=200, azim=frame*0.5)
#   return []
# ani = FuncAnimation(fig, update, frames=np.arange(0, 360, 2), interval=50, blit=False)

fig.tight_layout()
plt.show()
