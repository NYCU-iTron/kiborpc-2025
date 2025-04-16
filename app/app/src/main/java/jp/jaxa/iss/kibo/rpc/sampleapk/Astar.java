// package jp.jaxa.iss.kibo.rpc.sampleapk;

// import gov.nasa.arc.astrobee.types.Point;

// import java.util.*;
// import android.util.Log;

// public class Astar {
//   private final String TAG = this.getClass().getSimpleName();
  
//   private static final double STEP = 0.05;
//   private static final double X_MIN = 9.5, X_MAX = 11.55;
//   private static final double Y_MIN = -10.5, Y_MAX = -6.0;
//   private static final double Z_MIN = 4.02, Z_MAX = 5.57;

//   private static final int X_SIZE = (int)((X_MAX - X_MIN) / STEP) + 1;
//   private static final int Y_SIZE = (int)((Y_MAX - Y_MIN) / STEP) + 1;
//   private static final int Z_SIZE = (int)((Z_MAX - Z_MIN) / STEP) + 1;

//   public static class GridCell {
//     public boolean valid = false;
//     public String oasis = null;
//   }

//   private void initializeAreas() {
//     areas = new HashMap<>();
//     areas.put("Area 1", new Point[]{
//       new Point(10.42, -10.58, 4.82),
//       new Point(11.48, -10.58, 5.57)
//     });
//     areas.put("Area 2", new Point[]{
//       new Point(10.3, -9.25, 3.76203),
//       new Point(11.55, -8.5, 3.76203)
//     });
//     areas.put("Area 3", new Point[]{
//       new Point(10.3, -8.4, 3.76093),
//       new Point(11.55, -7.45, 3.76093)
//     });
//     areas.put("Area 4", new Point[]{
//       new Point(9.866984, -7.34, 4.32),
//       new Point(9.866984, -6.365, 5.57)
//     });
//     areas.put("Oasis 1", new Point[]{
//       new Point(10.425, -10.2, 4.445),
//       new Point(11.425, -9.5, 4.945)
//     });
//     areas.put("Oasis 2", new Point[]{
//       new Point(10.925, -9.5, 4.945),
//       new Point(11.425, -8.45, 5.445)
//     });
//     areas.put("Oasis 3", new Point[]{
//       new Point(10.425, -8.45, 4.945),
//       new Point(10.975, -7.4, 5.445)
//     });
//     areas.put("Oasis 4", new Point[]{
//       new Point(10.925, -7.4, 4.425),
//       new Point(11.425, -6.35, 4.945)
//     });
//     areas.put("KIZ 1", new Point[]{
//       new Point(10.3, -10.2, 4.32),
//       new Point(11.55, -6.0, 5.57)
//     });
//     areas.put("KIZ 2", new Point[]{
//       new Point(9.5, -10.5, 4.02),
//       new Point(10.5, -9.6, 4.8)
//     });
//   }

//   private void initializePoints() {
//     points = new HashMap<>();
//     points.put("dock", new Point(9.815, -9.806, 4.293));
//     points.put("patrol1", new Point(10.95, -10.0, 4.8));
//     points.put("patrol2", new Point(10.925, -8.875, 4.462));
//     points.put("patrol3", new Point(10.925, -7.925, 4.462));
//     points.put("patrol4", new Point(10.567, -6.853, 4.945));
//     points.put("report", new Point(11.143, -6.7607, 4.9654));
//   }

//   private void initializeGrid() {
//     grid = new GridCell[X_SIZE][Y_SIZE][Z_SIZE];
//     for (int i = 0; i < X_SIZE; i++) {
//       for (int j = 0; j < Y_SIZE; j++) {
//         for (int k = 0; k < Z_SIZE; k++) {
//           double x = X_MIN + i * STEP;
//           double y = Y_MIN + j * STEP;
//           double z = Z_MIN + k * STEP;
//           Point point = new Point(x, y, z);
//           GridCell cell = new GridCell();

//           if (inZone(point, areas.get("KIZ 1")) || inZone(point, areas.get("KIZ 2"))) {
//             cell.valid = true;
//             for (Map.Entry<String, double[]> entry : areas.entrySet()) {
//               if (entry.getKey().startsWith("Oasis") && inZone(x, y, z, entry.getValue())) {
//                 cell.oasis = entry.getKey();
//                 break;
//               }
//             }
//           }

//           grid[i][j][k] = cell;
//         }
//       }
//     }
//   }

//   private boolean inZone(Point p, double[] zone) {
//     return zone[0] <= p.getX() && p.getX() <= zone[3] &&
//            zone[1] <= p.getY() && p.getY() <= zone[4] &&
//            zone[2] <= p.getZ() && p.getZ() <= zone[5];
//   }

//   public int[] getCell(Point p) {
//     int i = (int) Math.round(((int) p.getX() - X_MIN) / STEP);
//     int j = (int) Math.round(((int) p.getY() - Y_MIN) / STEP);
//     int k = (int) Math.round(((int) p.getZ() - Z_MIN) / STEP);
//     if (i < 0 || j < 0 || k < 0 || i >= X_SIZE || j >= Y_SIZE || k >= Z_SIZE)
//       return null;
//     return new int[]{i, j, k};
//   }

//   public static List<Point> getNeighbors(Point point, int[][][] grid) {
//     List<Point> neighbors = new ArrayList<>();
//     int[][] directions = {
//         {1, 0, 0}, {-1, 0, 0},
//         {0, 1, 0}, {0, -1, 0},
//         {0, 0, 1}, {0, 0, -1}
//     };

//     for (int[] dir : directions) {
//       int nx = (int) point.getX() + dir[0];
//       int ny = (int) point.getY() + dir[1];
//       int nz = (int) point.getZ() + dir[2];

//       if (nx >= 0 && nx < grid.length &&
//           ny >= 0 && ny < grid[0].length &&
//           nz >= 0 && nz < grid[0][0].length &&   
//           grid[nx][ny][nz] == 0) {
//         neighbors.add(new Point(nx, ny, nz));
//       }
//     }

//     return neighbors;
//   }

//   public static List<Point> getAstarPath(Point start, Point goal, int[][][] grid) {
//     PriorityQueue<Node> openSet = new PriorityQueue<>(Comparator.comparingDouble(n -> n.fScore));
//     Map<Point, Point> cameFrom = new HashMap<>();
//     Map<Point, Double> gScore = new HashMap<>();

//     gScore.put(start, 0.0);
//     openSet.add(new Node(start, heuristic(start, goal)));

//     while (!openSet.isEmpty()) {
//       Node currentNode = openSet.poll();
//       Point current = currentNode.point;

//       if (current.equals(goal)) {
//         return reconstructPath(cameFrom, current);
//       }

//       for (Point neighbor : getNeighbors(current, grid)) {
//         double tentativeG = gScore.get(current) + 1;

//         if (!gScore.containsKey(neighbor) || tentativeG < gScore.get(neighbor)) {
//           cameFrom.put(neighbor, current);
//           gScore.put(neighbor, tentativeG);
//           double f = tentativeG + heuristic(neighbor, goal);
//           openSet.add(new Node(neighbor, f));
//         }
//       }
//     }

//     return new ArrayList<>(); // No path found
//   }
// }