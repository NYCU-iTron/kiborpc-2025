from typing import List
import numpy as np
import heapq
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.animation import FuncAnimation
import json
import math

class Astar:
  colors = {
    "Area": "red",
    "Oasis": "blue",
    "KIZ": "green"
  }
  alpha = 0.2

  areas = {
    # "Area 1": (10.42, -10.58, 4.82, 11.48, -10.58, 5.57),
    # "Area 2": (10.3, -9.25, 3.76203, 11.55, -8.5, 3.76203),
    # "Area 3": (10.3, -8.4, 3.76093, 11.55, -7.45, 3.76093),
    # "Area 4": (9.866984, -7.34, 4.32, 9.866984, -6.365, 5.57),
    "Oasis 1": (10.425, -10.2, 4.445, 11.425, -9.5, 4.945),
    "Oasis 2": (10.925, -9.5, 4.945, 11.425, -8.45, 5.445),
    "Oasis 3": (10.425, -8.45, 4.945, 10.975, -7.4, 5.445),
    "Oasis 4": (10.925, -7.4, 4.425, 11.425, -6.35, 4.945),
    "KIZ 1": (10.3, -10.2, 4.32, 11.55, -6.0, 5.57),
    "KIZ 2": (9.5, -10.5, 4.02, 10.5, -9.6, 4.8)
  }

  points = {
    "dock": (9.815, -9.806, 4.293),
    "patrol1":(10.95, -10, 4.8),
    "enhance1":(10.925, -8.875, 5.32),
    "patrol2":(10.925, -8.875, 4.462),
    "enhance2":(10.925, -8.45, 5.32),
    "patrol3":(10.925, -7.925, 4.462),
    "enhance3":(10.925, -6.853, 4.945),
    "patrol4":(10.567, -6.853, 4.945),
    "report": (11.143, -6.7607, 4.9654),

    "oasis1Center":((areas["Oasis 1"][0]+areas["Oasis 1"][3])/2, (areas["Oasis 1"][1]+areas["Oasis 1"][4])/2, (areas["Oasis 1"][2]+areas["Oasis 1"][5])/2),
    "oasis2Center":((areas["Oasis 2"][0]+areas["Oasis 2"][3])/2, (areas["Oasis 2"][1]+areas["Oasis 2"][4])/2, (areas["Oasis 2"][2]+areas["Oasis 2"][5])/2),
    "oasis3Center":((areas["Oasis 3"][0]+areas["Oasis 3"][3])/2, (areas["Oasis 3"][1]+areas["Oasis 3"][4])/2, (areas["Oasis 3"][2]+areas["Oasis 3"][5])/2),
    "oasis4Center":((areas["Oasis 4"][0]+areas["Oasis 4"][3])/2, (areas["Oasis 4"][1]+areas["Oasis 4"][4])/2, (areas["Oasis 4"][2]+areas["Oasis 4"][5])/2),
  }

  def __init__(self, ax):
    self.ax = ax

    # Grid parameters
    self.step = 0.05
    self.x_min, self.x_max = 9.5, 11.55
    self.y_min, self.y_max = -10.5, -6.0
    self.z_min, self.z_max = 4.02, 5.57
    self.grid_shape = (
      int((self.x_max - self.x_min) / self.step) + 1,  # 41
      int((self.y_max - self.y_min) / self.step) + 1,  # 96
      int((self.z_max - self.z_min) / self.step) + 1   # 32
    )

    self.areas_grid = {}
    for area_name, bounds in self.areas.items():
      min_point = (bounds[0], bounds[1], bounds[2])
      max_point = (bounds[3], bounds[4], bounds[5])

      # Convert bounding corners to grid indices
      min_i, min_j, min_k = self.get_cell(min_point)
      max_i, max_j, max_k = self.get_cell(max_point)

      self.areas_grid[area_name] = [min_i, min_j, min_k, max_i, max_j, max_k]

    self.points_grid = {}
    for point_name, point in self.points.items():
      i, j, k = self.get_cell(point)
      self.points_grid[point_name] = (i, j, k)

    self.grid = self.get_grid()
    point_order = ["dock", "patrol1", "enhance1", "patrol2", "enhance2", "patrol3", "enhance3", "patrol4", "report"]
    # point_order = ["dock", "patrol1", "patrol2", "patrol3", "patrol4", "report"]

    self.paths = []
    for i in range(len(point_order) - 1):
      start_name = point_order[i]
      end_name = point_order[i + 1]
      
      start_cell = self.points_grid[start_name]
      end_cell = self.points_grid[end_name]
      
      path = self.get_astar_path(start_cell, end_cell)
      self.paths.append(path)
 
  def draw_areas(self) -> None:
    for label, area in self.areas.items():
      self.draw_area(area, self.colors[label.split()[0]])
  
  def draw_area(self, area, color='cyan') -> None:
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

  def draw_path(self):    
    for path in self.paths:
      xs, ys, zs = zip(*[(9.5 + i * 0.05, -10.5 + j * 0.05, 4.02 + k * 0.05) for i, j, k in path])
      self.ax.plot(xs, ys, zs, 'r-', label='Path')

  def draw_oasis_score(self):
    xs, ys, zs, scores = [], [], [], []

    for i in range(len(self.grid)):
      for j in range(len(self.grid[0])):
        for k in range(len(self.grid[0][0])):
          cell = self.grid[i][j][k]
          if cell["oasis"] > 0:
            x = self.x_min + i * self.step
            y = self.y_min + j * self.step
            z = self.z_min + k * self.step
            xs.append(x)
            ys.append(y)
            zs.append(z)
            scores.append(cell["oasis"])  # score between 0.0 ~ 1.0

    sc = self.ax.scatter(xs, ys, zs, c=scores, cmap='viridis', s=5, alpha=0.8)
    # plt.colorbar(sc, label='Oasis Score')
    # plt.title("3D Oasis Score Distribution")


  def set_axes_equal(self):
    x_limits = self.ax.get_xlim()
    y_limits = self.ax.get_ylim()
    z_limits = self.ax.get_zlim()
    
    x_middle = np.mean(x_limits)
    y_middle = np.mean(y_limits)
    z_middle = np.mean(z_limits)

    max_range = max(
      (x_limits[1] - x_limits[0]),
      (y_limits[1] - y_limits[0]),
      (z_limits[1] - z_limits[0])
    ) / 2.0

    self.ax.set_xlim(x_middle - max_range, x_middle + max_range)
    self.ax.set_ylim(y_middle - max_range, y_middle + max_range)
    self.ax.set_zlim(z_middle - max_range, z_middle + max_range)

  def is_in_zone(self, x, y, z, zone):
    x_min, y_min, z_min, x_max, y_max, z_max = zone
    return x_min <= x <= x_max and y_min <= y <= y_max and z_min <= z <= z_max

  def get_grid(self):
    grid = [[[None for _ in range(self.grid_shape[2])] 
                   for _ in range(self.grid_shape[1])] 
                   for _ in range(self.grid_shape[0])]

    for i in range(self.grid_shape[0]):
      for j in range(self.grid_shape[1]):
        for k in range(self.grid_shape[2]):
          x = self.x_min + i * self.step
          y = self.y_min + j * self.step
          z = self.z_min + k * self.step
          cell = {"valid": False, "oasis": 0}

          # Check if the cell is in any KIZ
          if self.is_in_zone(x, y, z, self.areas["KIZ 1"]) or self.is_in_zone(x, y, z, self.areas["KIZ 2"]):
            cell["valid"] = True

            # Check if the cell is in any Oasis
            for zone_name, coords in self.areas.items():
              if zone_name == "Oasis 1" and self.is_in_zone(x, y, z, coords):
                cell["oasis"] = (j - self.areas_grid[zone_name][1]) / (self.areas_grid[zone_name][4] - self.areas_grid[zone_name][1])
                break
              elif (zone_name == "Oasis 2" or zone_name == "Oasis 3") and self.is_in_zone(x, y, z, coords):
                cell["oasis"] = (k - self.areas_grid[zone_name][2]) / (self.areas_grid[zone_name][5] - self.areas_grid[zone_name][2])
                break
              elif zone_name == "Oasis 4" and self.is_in_zone(x, y, z, coords):
                cell["oasis"] = (i - self.areas_grid[zone_name][0]) / (self.areas_grid[zone_name][3] - self.areas_grid[zone_name][0])
                break
              else:
                continue

            if cell['oasis'] == 0:
              # Not in oasis
              min_dist = float("inf")
              for point_name, point in self.points_grid.items():
                if point_name == "oasis1Center" or point_name == "oasis2Center" or point_name == "oasis3Center" or point_name == "oasis4Center":
                  l, m, n = point
                  min_dist = min(min_dist, ((i-l)**2+(j-m)**2+(k-n)**2)**0.5)
              cell["oasis_dist"] = min_dist                
          grid[i][j][k] = cell
    
    return grid
  
  def get_cell(self, point):
    x, y, z = point
    i = round((x - self.x_min) / self.step)
    j = round((y - self.y_min) / self.step)
    k = round((z - self.z_min) / self.step)

    if 0 <= i < self.grid_shape[0] and 0 <= j < self.grid_shape[1] and 0 <= k < self.grid_shape[2]:
      return (i, j, k)
    return None
  
  def get_neighbors(self, cell):
    i, j, k = cell
    neighbors = []
    for di in [-1, 0, 1]:
      for dj in [-1, 0, 1]:
        for dk in [-1, 0, 1]:
          if di == 0 and dj == 0 and dk == 0:
            continue
          ni, nj, nk = i + di, j + dj, k + dk
          if (0 <= ni < self.grid_shape[0] and 
              0 <= nj < self.grid_shape[1] and 
              0 <= nk < self.grid_shape[2] and 
              self.grid[ni][nj][nk]["valid"]):
            neighbors.append((ni, nj, nk))
    return neighbors
  
  def get_astar_path(self, start, goal):
    """"
    g_score: The exact cost from the start node to the current node.
    h_score, heuristic: The estimated cost from the current node to the goal.
    f_score: The total estimated cost of a path through the current node, defined as f = g + h.
    
    Oasis 的加分是結果，而不是路徑規劃的直接成本。
    目標是找到一條路徑，優化時間成本 g ，同時考慮 Oasis 的吸引力。
    Oasis 節點的 edge_time 減半，降低 g_score，使包含 Oasis 的路徑在 f = g + h 中更具競爭力。
    """
    open_set = [(0, start)] # (f_score, cell)
    came_from = {}
    g_score = {start: 0} # Time from start
    f_score = {start: self.h_compound(start, goal)}  # g + h
    
    while open_set:
      _, current = heapq.heappop(open_set)
      if current == goal:
        # Reconstruct path
        path = []
        while current in came_from:
          path.append(current)
          current = came_from[current]
        path.append(start)
        return path[::-1]
      
      for neighbor in self.get_neighbors(current):
        # Edge cost
        dx = neighbor[0] - current[0]
        dy = neighbor[1] - current[1]
        dz = neighbor[2] - current[2]

        distance = ((dx) ** 2 + (dy) ** 2 + (dz) ** 2) ** 0.5
        distance -= self.grid[neighbor[0]][neighbor[1]][neighbor[2]]["oasis"]
        
        tentative_g = g_score[current] + distance
        if tentative_g < g_score.get(neighbor, float("inf")):
          came_from[neighbor] = current
          g_score[neighbor] = tentative_g
          h = self.h_compound(neighbor, goal)
          f_score[neighbor] = tentative_g + h
          heapq.heappush(open_set, (f_score[neighbor], neighbor))
  
    return None  # No path
  
  def h_compound(self, cell, goal, w_distance=0.2, w_oasis=0.8):
    """
    Compound heuristic combining time, Oasis, and KIZ factors.
    
    Args:
      cell: Tuple (i, j, k) - current cell indices.
      goal: Tuple (i, j, k) - goal cell indices.
      grid: 3D list grid[i][j][k] with {"valid": bool, "oasis": str/None}.
      areas: Dict of zone coordinates (KIZ 1, KIZ 2, Oasis zones).
      speed: Astrobee speed in meters/second (default 0.1).
      w_time: Weight for time term (default 1.0).
      w_oasis: Weight for Oasis discount (default 0.5).
    
    Returns:
      Estimated distance to goal.
    """
    # Convert indices to real-world coordinates
    dx = goal[0] - cell[0]
    dy = goal[1] - cell[1]
    dz = goal[2] - cell[2]
    
    # Time term: Euclidean distance / speed
    distance = ((dx) ** 2 + (dy) ** 2 + (dz) ** 2) ** 0.5
    
    # Oasis term: Discount if in Oasis zone
    h_oasis = 0
    if self.grid[cell[0]][cell[1]][cell[2]]["oasis"] > 0:
      # Cap discount at 10% of time or 0.5 m
      h_oasis = -max(0.1 * distance, 1)
    else:
      oasis_dist = self.grid[cell[0]][cell[1]][cell[2]]["oasis_dist"]
      max_effect = 1.2  # 最大吸引力
      min_dist = 5
      max_dist = 15
      normalized = max(0, min(1, (oasis_dist - min_dist) / (max_dist - min_dist)))
      h_oasis = -max_effect * (1 - normalized)      
      # print(h_oasis)

    # Combine with weights
    return w_distance * distance + w_oasis * h_oasis
  
  def show_grid_info(self):
    total_cells = self.grid_shape[0] * self.grid_shape[1] * self.grid_shape[2]
    valid_cells = sum(1 for i in range(self.grid_shape[0]) 
                        for j in range(self.grid_shape[1]) 
                        for k in range(self.grid_shape[2]) 
                        if self.grid[i][j][k]["valid"])
    oasis_cells = sum(1 for i in range(self.grid_shape[0]) 
                        for j in range(self.grid_shape[1]) 
                        for k in range(self.grid_shape[2]) 
                        if self.grid[i][j][k]["oasis"] > 0)

    print(f"Total cells: {total_cells}")
    print(f"Valid cells (in KIZ): {valid_cells}")
    print(f"Oasis cells: {oasis_cells}")
  
  def write_json(self):
    pose_list = []
    threshold=0.15
    last = None

    for subpath in self.paths:
      for i, j, k in subpath:          
        x = self.x_min + i * self.step
        y = self.y_min + j * self.step
        z = self.z_min + k * self.step
        if last is None or self.euclidean((x, y, z), last) >= threshold:
          pose = {
            "point": {"x": x, "y": y, "z": z},
            "quaternion": {"x": 0, "y": 0, "z": 0, "w": 1}
          }
          pose_list.append(pose)
          last = (x, y, z)

    with open("pose_path.json", "w") as f:
      json.dump(pose_list, f, indent=2)

  def euclidean(self, p1, p2):
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(p1, p2)))


                      
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

astar = Astar(ax)
astar.draw_areas()
astar.draw_path()
astar.set_axes_equal()

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.view_init(elev=200, azim=-45)

# Function to update the view angle for animation
# def update(frame):
#   ax.view_init(elev=200, azim=frame*0.5)
#   return []
# ani = FuncAnimation(fig, update, frames=np.arange(0, 360, 2), interval=50, blit=False)

astar.write_json()
fig.tight_layout()
plt.show()