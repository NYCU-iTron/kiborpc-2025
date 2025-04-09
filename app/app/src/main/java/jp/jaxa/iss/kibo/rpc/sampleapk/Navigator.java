import jp.jaxa.iss.kibo.rpc.api.KiboRpcService;

import android.util.Log;

import gov.nasa.arc.astrobee.types.Point;
import gov.nasa.arc.astrobee.types.Quaternion;
import gov.nasa.arc.astrobee.Kinematics;
import gov.nasa.arc.astrobee.Result;

/**
 * Class to represent a 3D pose consisting of a position and orientation.
 * 
 * @param point The position in 3D space.
 * @param quaternion The orientation represented as a quaternion.
 */
public class Pose {
  private Point point;
  private Quaternion quaternion;

  public Pose(Point point, Quaternion quaternion) {
    this.point = point;
    this.quaternion = quaternion;
  }

  public Point getPoint() {
    return point;
  }

  public Quaternion getQuaternion() {
    return quaternion;
  }
}

/**
 * Class to handle navigation commands for the Astrobee robot.
 */
public class Navigator extends KiboRpcService {

  // Target poses in each area
  public static final Pose Dock = new Pose(new Point(9.815, -9.806, 4.293), new Quaternion(1, 0, 0, 0));
  public static final Pose Patrol1 = new Pose(new Point(10.95, -10, 4.8), new Quaternion(0, -0.398, -0.584, 0.707));
  public static final Pose Patrol2 = new Pose(new Point(10.925, -8.875, 4.462), new Quaternion(0, 0.707, 0, 0.707));
  public static final Pose Patrol3 = new Pose(new Point(10.925, -7.925, 4.462), new Quaternion(0, 0.707, 0, 0.707));
  public static final Pose Patrol4 = new Pose(new Point(10.567, -6.853, 4.945), new Quaternion(0, 0, 1, 0));
  public static final Pose Report = new Pose(new Point(11.143, -6.7607, 4.9654), new Quaternion(0, 0, 0.707, 0.707));

  public Pose getCurrentPose() {
    // TODO : Deal with measure error and low confidence
    Kinematics kinematics = api.getRobotKinematics();
    return new Pose(kinematics.getPosition(), kinematics.getOrientation());
  }

  public Result moveTo(Pose target) {
    // TODO : Deal with failure
    Result result = api.moveTo(target.getPoint(), target.getQuaternion());
    return result;
  }

  /**
   * Moves the robot through a series of waypoints.
   * 
   * @param waypoints A list of poses representing the waypoints to navigate through.
   * @return The result of the last move command.
   */
  public Result navigateThrough(List<Pose> waypoints) {
    Result lastResult = null;

    for (Pose pose : waypoints) {
      lastResult = api.moveTo(pose.getPoint(), pose.getQuaternion());
      
      int maxRetries = 3;
      while (lastResult != Result.SUCCESS && maxRetries > 0) {
        Log.d("Navigator", "Retrying move to: " + pose.toString());
        lastResult = api.moveTo(pose.getPoint(), pose.getQuaternion());
        maxRetries--;
      }

      // If still not successful after retries, log the error and continue to the next waypoint
      if (lastResult != Result.SUCCESS) {
        Log.e("Navigator", "Failed to move to: " + pose.toString());
      }

    }
    Log.d("Navigator", "Moved to: " + pose.toString());
    return lastResult;
  }

  /**
   * Interpolates between two poses to create a smooth transition.
   * 
   * @param start The starting pose.
   * @param end The ending pose.
   * @return A list of poses representing the interpolated path.
   */
  public static List<Pose> interpolate(Pose start, Pose end) {
    float dx = targetPoint.getX() - currentPoint.getX();
    float dy = targetPoint.getY() - currentPoint.getY();
    float dz = targetPoint.getZ() - currentPoint.getZ();

    float linearUnit = 0.08f;
    float distance = Math.sqrt(dx * dx + dy * dy + dz * dz);
    int numSteps = (int) (distance / linearUnit);
    List<Pose> poses = new ArrayList<>();

    if (numSteps == 0) {
      poses.add(end);
      return poses;
    }
    
    for (int i = 0; i <= numSteps; i++) {
      float t = (float) i / numSteps;
      new Pose current = new Pose(
        lerpPoint(start.getPoint(), end.getPoint(), t),
        slerp(start.getQuaternion(), end.getQuaternion(), t)
      );
      poses.add(current);
    }

    return poses;
  }

  /**
   * Performs linear interpolation between two points.
   * 
   * @param a The first point.
   * @param b The second point.
   * @param t The interpolation parameter (0 <= t <= 1).
   * @return A new point that is the result of the interpolation.
   */
  public static Point lerpPoint(Point a, Point b, float t) {
    float x = a.getX() + (b.getX() - a.getX()) * t;
    float y = a.getY() + (b.getY() - a.getY()) * t;
    float z = a.getZ() + (b.getZ() - a.getZ()) * t;
    return new Point(x, y, z);
  }

  /**
   * Performs spherical linear interpolation (SLERP) between two quaternions.
   * 
   * @param q1 The first quaternion.
   * @param q2 The second quaternion.
   * @param t The interpolation parameter (0 <= t <= 1).
   * @return A new quaternion that is the result of the interpolation.
   */
  public static Quaternion slerp(Quaternion q1, Quaternion q2, float t) {
    float dot = q1.getX() * q2.getX() + q1.getY() * q2.getY() + q1.getZ() * q2.getZ() + q1.getW() * q2.getW();

    // If dot < 0, the interpolation will take the long way around the sphere. So we reverse one quaternion.
    if (dot < 0.0f) {
      q2 = new Quaternion(-q2.getX(), -q2.getY(), -q2.getZ(), -q2.getW());
      dot = -dot;
    }

    final float DOT_THRESHOLD = 0.995f;
    if (dot > DOT_THRESHOLD) {
      // Use linear interpolation to avoid division by 0
      float x = q1.getX() + t * (q2.getX() - q1.getX());
      float y = q1.getY() + t * (q2.getY() - q1.getY());
      float z = q1.getZ() + t * (q2.getZ() - q1.getZ());
      float w = q1.getW() + t * (q2.getW() - q1.getW());
      float norm = (float) Math.sqrt(x * x + y * y + z * z + w * w);
      return new Quaternion(x / norm, y / norm, z / norm, w / norm);
    }

    float theta_0 = (float) Math.acos(dot); // angle between input quaternions
    float theta = theta_0 * t;              // angle at interpolation parameter t
    float sin_theta = (float) Math.sin(theta);
    float sin_theta_0 = (float) Math.sin(theta_0);

    float s1 = (float) Math.cos(theta) - dot * sin_theta / sin_theta_0;
    float s2 = sin_theta / sin_theta_0;

    float x = s1 * q1.getX() + s2 * q2.getX();
    float y = s1 * q1.getY() + s2 * q2.getY();
    float z = s1 * q1.getZ() + s2 * q2.getZ();
    float w = s1 * q1.getW() + s2 * q2.getW();

    return new Quaternion(x, y, z, w);
  }

  /**
   * Calculates the quaternion to rotate from the current pose to face the target pose.
   * 
   * @param current The current pose.
   * @param target The target pose.
   * @return A new pose with the same position as the current pose but rotated to face the target pose.
   */
  public static Pose getPoseToFaceTarget(Pose current, Pose target) {
    Point currentPoint = current.getPoint();
    Point targetPoint = target.getPoint();

    // Calculate direction vector
    float dx = targetPoint.getX() - currentPoint.getX();
    float dy = targetPoint.getY() - currentPoint.getY();
    float dz = targetPoint.getZ() - currentPoint.getZ();

    // Normalize the direction vector
    float length = Math.sqrt(dx * dx + dy * dy + dz * dz);
    if (length == 0) {
      return current; // No movement needed
    }

    dx /= length;
    dy /= length;
    dz /= length;

    // Default direction vector
    float ux = 1f;
    float uy = 0f;
    float uz = 0f;

    // Dot product to find angle
    float dot = ux * dx + uy * dy + uz * dz;
    dot = Math.max(-1.0f, Math.min(1.0f, dot));
    float theta = (float) Math.acos(dot);

    // Special cases
    if (Math.abs(dot - 1) < 1e-6) { // Already aligned with default direction
      return current;
    } else if (Math.abs(dot + 1) < 1e-6) { // 180° opposite to default direction
      // Choose an arbitrary perpendicular axis (e.g., x-axis or y-axis)
      return new Pose(currentPoint, new Quaternion(1, 0, 0, 0)); // If aligned with x, rotate around x
    }

    // Cross product to find rotation axis
    float axisX = uy * dz - uz * dy;
    float axisY = uz * dx - ux * dz;
    float axisZ = ux * dy - uy * dx;
    float axisMagnitude = (float) Math.sqrt(axisX * axisX + axisY * axisY + axisZ * axisZ);

    // Avoid division by zero
    if (axisMagnitude < 1e-6) {
      return new Pose(currentPoint, new Quaternion(0, 0, 0, 1));
    }

    axisX /= axisMagnitude;
    axisY /= axisMagnitude;
    axisZ /= axisMagnitude;

    // Calculate quaternion
    float halfTheta = theta / 2;
    float sinHalfTheta = (float) Math.sin(halfTheta);
    float qx = axisX * sinHalfTheta;
    float qy = axisY * sinHalfTheta;
    float qz = axisZ * sinHalfTheta;
    float qw = (float) Math.cos(halfTheta);

    return new Pose(currentPoint, new Quaternion(qx, qy, qz, qw));
  }
}
