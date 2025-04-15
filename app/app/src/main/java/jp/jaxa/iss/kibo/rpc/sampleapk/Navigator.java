package jp.jaxa.iss.kibo.rpc.sampleapk;

import jp.jaxa.iss.kibo.rpc.api.KiboRpcService;
import jp.jaxa.iss.kibo.rpc.api.KiboRpcApi;

import gov.nasa.arc.astrobee.types.Point;
import gov.nasa.arc.astrobee.types.Quaternion;
import gov.nasa.arc.astrobee.Kinematics;
import gov.nasa.arc.astrobee.Result;

import java.util.List;
import java.util.ArrayList;

import android.util.Log;


/**
 * Class to handle navigation commands for the Astrobee robot.
 */
public class Navigator {
  private final KiboRpcApi api;
  private final String TAG = this.getClass().getSimpleName();
  long startTime;

  // Target poses in each area
  public static final Pose Dock = new Pose(new Point(9.815, -9.806, 4.293), new Quaternion(1.0f, 0.0f, 0.0f, 0.0f));
  public static final Pose Patrol1 = new Pose(new Point(10.95, -10, 4.8), new Quaternion(0.0f, -0.398f, -0.584f, 0.707f));
  public static final Pose Patrol2 = new Pose(new Point(10.925, -8.875, 4.462), new Quaternion(0.0f, 0.707f, 0.0f, 0.707f));
  public static final Pose Patrol3 = new Pose(new Point(10.925, -7.925, 4.462), new Quaternion(0.0f, 0.707f, 0.0f, 0.707f));
  public static final Pose Patrol4 = new Pose(new Point(10.567, -6.853, 4.945), new Quaternion(0.0f, 0.0f, 1.0f, 0.0f));
  public static final Pose Report = new Pose(new Point(11.143, -6.7607, 4.9654), new Quaternion(0.0f, 0.0f, 0.707f, 0.707f));

  /**
   * Constructor
   * 
   * @param apiRef API reference from runPlan in YourService.java
   * @return The current average pose
   */
  public Navigator(KiboRpcApi apiRef) {
    this.api = apiRef;
    startTime = System.currentTimeMillis();
    Log.i(TAG,  "Initialized at " + startTime);
  }

  public Pose getCurrentPose() {
    int numAttempts = 10;
    double accPosX = 0, accPosY = 0, accPosZ = 0;
    double accOriX = 0, accOriY = 0, accOriZ = 0, accOriW = 0;
    double weight = 0, totalWeight = 0;

    // Collect kinematics data from API
    for (int i = 0; i < numAttempts; i++) {
      Kinematics kinematics = api.getRobotKinematics();
      Kinematics.Confidence confidence = kinematics.getConfidence();

      if (confidence == Kinematics.Confidence.GOOD) {
        weight = 1.0;
      } else if (confidence == Kinematics.Confidence.POOR) {
        weight = 0.5;
      } else { // confidence == Confidence.LOST
        Log.w(TAG, "Get current pose with low confidence");
        continue;      
      }

      totalWeight += weight;

      Point position = kinematics.getPosition();
      accPosX += position.getX() * weight;
      accPosY += position.getY() * weight;
      accPosZ += position.getZ() * weight;

      Quaternion orientation = kinematics.getOrientation();
      accOriX += (double) orientation.getX() * weight;
      accOriY += (double) orientation.getY() * weight;
      accOriZ += (double) orientation.getZ() * weight;
      accOriW += (double) orientation.getW() * weight;
    }
    
    // Compute avergae
    double avgPosX = accPosX / totalWeight;
    double avgPosY = accPosY / totalWeight;
    double avgPosZ = accPosZ / totalWeight;
    
    float avgOriX = (float) (accOriX / totalWeight);
    float avgOriY = (float) (accOriY / totalWeight);
    float avgOriZ = (float) (accOriZ / totalWeight);
    float avgOriW = (float) (accOriW / totalWeight);

    // Normalize
    float norm = (float) Math.sqrt(avgOriX * avgOriX + avgOriY * avgOriY + avgOriZ * avgOriZ + avgOriW * avgOriW);
    avgOriX /= norm;
    avgOriY /= norm;
    avgOriZ /= norm;
    avgOriW /= norm;

    Pose pose = new Pose(new Point(avgPosX, avgPosY, avgPosZ), new Quaternion(avgOriX, avgOriY, avgOriZ, avgOriW));
    return pose;
  }

  /* -------------------------------------------------------------------------- */
  /*                                  Movement                                  */
  /* -------------------------------------------------------------------------- */

  public Result moveTo(Pose target) {
    int maxRetries = 5;
    Result result = api.moveTo(target.getPoint(), target.getQuaternion(), false);
    
    // Enter retry loop if failed
    while (!result.hasSucceeded() && maxRetries > 0) {
      Log.i(TAG, "Retrying move to: " + target.toString());
      result = api.moveTo(target.getPoint(), target.getQuaternion(), false);
      maxRetries--;
    }

    // If still not successful after retries, log the error.
    if (!result.hasSucceeded()) {
      Log.w(TAG, "Failed to move to: " + target.toString());
    }
    
    return result;
  }

  /**
   * Moves the robot through a series of waypoints.
   * 
   * @param waypoints A list of poses representing the waypoints to navigate through.
   * @return The result of the last move command.
   */
  public Result navigateThrough(Pose targetPose) {
    Pose currentPose = getCurrentPose();
    List<Pose> poses = interpolate(currentPose, targetPose);
    Result result = null;

    for (Pose pose : poses) {
      result = moveTo(pose);
    }

    return result;
  }

  public Result navigatePrecomputed(Context context) {
    List<Pose> poses = loadPathFromJson(context, "pose_path.json");
    Result result = null;

    for (Pose pose : poses) {
      result = moveTo(pose);
    }

    return result;
  }

  public void navigateArea1() {
    // Pose point1 = new Pose (
    //   new Point(10.45, -9.7, 4.47), 
    //   new Quaternion(1.0f, 0.0f, 0.0f, 0.0f)
    // );
    // Pose point2 = new Pose (
    //   new Point(10.45, -9.52, 4.47), 
    //   new Quaternion(1.0f, 0.0f, 0.0f, 0.0f)
    // );
    // Pose point3 = new Pose(
    //   new Point(10.95, -9.52, 4.9), 
    //   new Quaternion(1.0f, 0.0f, 0.0f, 0.0f)
    // );
    Pose finalPose = new Pose(
      new Point(10.95, -9.9, 4.9),
      new Quaternion(0.0f, -0.281f, -0.649f, 0.707f)
    );

    // moveTo(point1);
    // moveTo(point2);
    // moveTo(point3);
    moveTo(finalPose);
  }

  public void navigateArea2() {
    // Pose point1 = new Pose(
    //   new Point(10.94, -9.5, 4.9), 
    //   new Quaternion(0.0f, 0.707f, 0.0f, 0.707f)
    // );
    // Pose point2 = new Pose(
    //   new Point(10.94, -9.48, 5.43), 
    //   new Quaternion(0.0f, 0.707f, 0.0f, 0.707f)
    // );
    // Pose point3 = new Pose(
    //   new Point(10.94, -8.875, 5.43), 
    //   new Quaternion(0.0f, 0.707f, 0.0f, 0.707f)
    // );
    Pose finalPose = new Pose(
      new Point(10.925, -8.875, 4.462),
      new Quaternion(0.0f, 0.707f, 0.0f, 0.707f)
    );
    // moveTo(point1);
    // moveTo(point2);
    // moveTo(point3);
    moveTo(finalPose);
  }

  public void navigateArea3() {
    // Pose point1 = new Pose(
    //   new Point(10.925, -8.875, 5.43), 
    //   new Quaternion(0.0f, 0.707f, 0.0f, 0.707f)
    // );
    // Pose point2 = new Pose(
    //   new Point(10.925, -7.925, 5.43), 
    //   new Quaternion(0.0f, 0.707f, 0.0f, 0.707f)
    // );
    Pose finalPose = new Pose(
      new Point(10.925, -7.925, 4.462),
      new Quaternion(0.0f, 0.707f, 0.0f, 0.707f)
    );
    // moveTo(point1);
    // moveTo(point2);
    moveTo(finalPose);
  }

  public void navigateArea4() {
    // Pose point1 = new Pose(
    //   new Point(11.4, -7.4, 4.462), 
    //   new Quaternion(0.0f, 0.0f, 1.0f, 0.0f)
    // );
    // Pose point2 = new Pose(
    //   new Point(11.4, -6.853, 4.92), 
    //   new Quaternion(0.0f, 0.0f, 1.0f, 0.0f)
    // );
    Pose finalPose = new Pose(
      new Point(10.567, -6.853, 4.92),
      new Quaternion(0.0f, -1.0f, 0.02f, 0.02f)
    );
    // moveTo(point1);
    // moveTo(point2);
    moveTo(finalPose);
  }

  public void navigateReport() {
    Pose finalPose = new Pose(
      new Point(11.143, -6.7607, 4.9654),
      new Quaternion(0.0f, 0.0f, 0.707f, 0.707f)
    );
    moveTo(finalPose);
  }

  /* -------------------------------------------------------------------------- */
  /*                               Tool Functions                               */
  /* -------------------------------------------------------------------------- */

  /**
   * Interpolates between two poses to create a smooth transition.
   * 
   * @param start The starting pose.
   * @param end The ending pose.
   * @return A list of poses representing the interpolated path.
   */
  public static List<Pose> interpolate(Pose start, Pose end) {
    Point startPoint = start.getPoint();
    Point endPoint = end.getPoint();

    double dx = endPoint.getX() - startPoint.getX();
    double dy = endPoint.getY() - startPoint.getY();
    double dz = endPoint.getZ() - startPoint.getZ();

    double linearUnit = 0.2;
    double distance = Math.sqrt(dx * dx + dy * dy + dz * dz);
    int numSteps = (int) (distance / linearUnit);
    List<Pose> poses = new ArrayList<>();

    if (numSteps == 0) {
      poses.add(end);
      return poses;
    }
    
    for (int i = 0; i <= numSteps; i++) {
      double t = (double) i / numSteps;
      Pose current = new Pose(
        lerpPoint(start.getPoint(), end.getPoint(), t),
        slerp(start.getQuaternion(), end.getQuaternion(), (float) t)
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
  public static Point lerpPoint(Point a, Point b, double t) {
    double x = a.getX() + (b.getX() - a.getX()) * t;
    double y = a.getY() + (b.getY() - a.getY()) * t;
    double z = a.getZ() + (b.getZ() - a.getZ()) * t;
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
    float dx = (float) targetPoint.getX() - (float) currentPoint.getX();
    float dy = (float) targetPoint.getY() - (float) currentPoint.getY();
    float dz = (float) targetPoint.getZ() - (float) currentPoint.getZ();

    // Normalize the direction vector
    float length = (float) Math.sqrt(dx * dx + dy * dy + dz * dz);
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
      return new Pose(currentPoint, new Quaternion(1.0f, 0.0f, 0.0f, 0.0f)); // If aligned with x, rotate around x
    }

    // Cross product to find rotation axis
    float axisX = uy * dz - uz * dy;
    float axisY = uz * dx - ux * dz;
    float axisZ = ux * dy - uy * dx;
    float axisMagnitude = (float) Math.sqrt(axisX * axisX + axisY * axisY + axisZ * axisZ);

    // Avoid division by zero
    if (axisMagnitude < 1e-6) {
      return new Pose(currentPoint, new Quaternion(0.0f, 0.0f, 0.0f, 1.0f));
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

  public static Pose getBodyPoseFromCamera(Pose cameraPose) {
    Point cameraPoint = cameraPose.getPoint();
    
    double bodyX = cameraPoint.getX() - 0.1177;
    double bodyY = cameraPoint.getY() + 0.0422;
    double bodyZ = cameraPoint.getZ() + 0.0826;

    Pose bodyPose = new Pose(new Point(bodyX, bodyY, bodyZ), cameraPose.getQuaternion());

    return bodyPose;
  }
}
