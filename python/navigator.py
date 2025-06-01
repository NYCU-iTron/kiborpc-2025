from typing import List
import numpy as np
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.animation import FuncAnimation
from scipy.spatial.transform import Rotation as R

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
  # Astronaut 11.143 -6.7607 4.9654 0 0 0.707 0.707
  poses = {
    "dock": (9.815, -9.806, 4.293, 0, 0, 0, 1),
    "patrol1":(11.11, -9.49, 5.435, 0.0, 0.0, -0.707, 0.707),
    "patrol2":(10.925, -8.875, 4.462, 0.5, 0.5, -0.5, 0.5),
    "patrol3":(10.925, -7.925, 4.462, 0.5, 0.5, -0.5, 0.5),
    "patrol4":(11.35, -6.7607, 4.935, 0.0, -1.0, 0.0, 0.0),
    "patrol5":(10.925, -8.35, 5.3, 0.0, 0.707, 0.0, 0.707), # Combined Area 2 3
    "report": (11.35, -6.7607, 4.935, 0.633, 0.754, -0.133, 0.112),
  }

  def __init__(self, ax):
    self.ax = ax

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

  def calculate_target_quaternion(self, current_pos, target_pos, local_axis='z', keep_horizontal_axis='auto'):
    """
    計算從當前位置面向目標位置的四元數
    
    Args:
        current_pos: 當前位置 [x, y, z]
        target_pos: 目標位置 [x, y, z]  
        local_axis: 指定哪個local軸要朝向目標 ('x', 'y', 'z')
        keep_horizontal_axis: 指定哪個軸要保持水平 ('x', 'y', 'z', 'auto')
                            'auto' 會自動選擇最接近xy平面的軸
    
    Returns:
        quaternion: [qx, qy, qz, qw] (scipy格式)
    """
    
    current_pos = np.array(current_pos)
    target_pos = np.array(target_pos)
    
    # 計算朝向目標的方向向量
    direction = target_pos - current_pos
    direction_norm = np.linalg.norm(direction)
    
    if direction_norm < 1e-10:
      # 如果距離太近，返回單位四元數
      return np.array([0, 0, 0, 1])
    
    direction = direction / direction_norm
    
    # 定義local軸的標準向量
    local_axes = {
      'x': np.array([1, 0, 0]),
      'y': np.array([0, 1, 0]), 
      'z': np.array([0, 0, 1])
    }
    
    target_local_axis = local_axes[local_axis]
    
    # 如果目標方向和local軸已經對齊，則不需要旋轉主軸
    if np.abs(np.dot(direction, target_local_axis)) > 0.99999:
      primary_rotation = R.identity()
    else:
      # 計算將指定local軸對齊到目標方向的旋轉
      primary_rotation = R.align_vectors([direction], [target_local_axis])[0]
    
    # 應用主要旋轉後的軸
    rotated_axes = {
      'x': primary_rotation.apply(local_axes['x']),
      'y': primary_rotation.apply(local_axes['y']),
      'z': primary_rotation.apply(local_axes['z'])
    }
    
    # 決定哪個軸要保持水平
    if keep_horizontal_axis == 'auto':
      # 自動選擇最接近xy平面的軸（z分量最小的軸）
      other_axes = [k for k in local_axes.keys() if k != local_axis]
      z_components = [abs(rotated_axes[axis][2]) for axis in other_axes]
      horizontal_axis = other_axes[np.argmin(z_components)]
    else:
      horizontal_axis = keep_horizontal_axis
    
    # 如果要保持水平的軸就是朝向目標的軸，選擇另一個軸
    if horizontal_axis == local_axis:
      other_axes = [k for k in local_axes.keys() if k != local_axis]
      z_components = [abs(rotated_axes[axis][2]) for axis in other_axes]
      horizontal_axis = other_axes[np.argmin(z_components)]
    
    # 將選定的軸投影到xy平面並標準化
    horizontal_vector = rotated_axes[horizontal_axis].copy()
    horizontal_vector[2] = 0  # 投影到xy平面
    horizontal_norm = np.linalg.norm(horizontal_vector)
    
    if horizontal_norm > 1e-10:
      horizontal_vector = horizontal_vector / horizontal_norm
        
      # 計算需要的額外旋轉來保持軸水平
      current_horizontal = rotated_axes[horizontal_axis]
      
      # 計算繞目標方向軸的旋轉角度
      # 使用叉積和點積計算角度
      cross_product = np.cross(current_horizontal, horizontal_vector)
      dot_product = np.dot(current_horizontal, horizontal_vector)
      
      # 計算旋轉角度（繞direction軸）
      rotation_angle = np.arctan2(np.dot(cross_product, direction), dot_product)
      
      # 創建繞direction軸的旋轉
      if abs(rotation_angle) > 1e-10:
        secondary_rotation = R.from_rotvec(rotation_angle * direction)
        final_rotation = secondary_rotation * primary_rotation
      else:
        final_rotation = primary_rotation
    else:
      final_rotation = primary_rotation
    
    return final_rotation.as_quat()  # 返回 [qx, qy, qz, qw] 格式
  
def draw_point(ax, point, label):
  x, y, z, qx, qy, qz, qw = point[0], point[1], point[2], point[3], point[4], point[5], point[6]
  ax.scatter(x, y, z, label=f"{label}", s=50)

  quat_norm = np.sqrt(qx**2 + qy**2 + qz**2 + qw**2)
  qx, qy, qz, qw = qx/quat_norm, qy/quat_norm, qz/quat_norm, qw/quat_norm
  
  # Create rotation from quaternion (scipy handles normalization)
  rot = R.from_quat([qx, qy, qz, qw])  # Note: scipy uses [x,y,z,w] order
  rotation_matrix = rot.as_matrix()
    
  # Extract axes
  x_axis = rotation_matrix[:, 0]
  y_axis = rotation_matrix[:, 1] 
  z_axis = rotation_matrix[:, 2]

  length = 0.5 # Arrow length

  # Draw local axis
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
quat1 = navigator.calculate_target_quaternion([current_x, current_y, current_z], [target_x, target_y, target_z], local_axis='x', keep_horizontal_axis='y')
print(quat1)
# d = np.sqrt((target_x - current_x)**2 + (target_y - current_y)**2 + (target_z - current_z)**2)
# print(d)

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
