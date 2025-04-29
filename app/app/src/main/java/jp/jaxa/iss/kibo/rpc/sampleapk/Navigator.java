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
   */
  public Navigator(KiboRpcApi apiRef) {
    this.api = apiRef;
    startTime = System.currentTimeMillis();
    Log.i(TAG,  "Initialized at " + startTime);
  }

  /* -------------------------------------------------------------------------- */
  /*                                 Measurement                                */
  /* -------------------------------------------------------------------------- */

  /**
   * Process and return the measured pose.
   * 
   * @return current pose.
   */
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

  /**
   * Move the robot to target pose.
   * 
   * @param targetPose
   * @return The result of the last move command.
   */
  public Result moveTo(Pose targetPose) {
    int maxRetries = 5;
    Result result = api.moveTo(targetPose.getPoint(), targetPose.getQuaternion(), false);
    
    // Enter retry loop if failed
    while (!result.hasSucceeded() && maxRetries > 0) {
      Log.i(TAG, "Retrying move to: " + targetPose.toString());
      result = api.moveTo(targetPose.getPoint(), targetPose.getQuaternion(), false);
      maxRetries--;
    }

    // If still not successful after retries, log the error.
    if (!result.hasSucceeded()) {
      Log.w(TAG, "Failed to move to " + targetPose.toString() + "because " + result.getMessage());
    }
    
    return result;
  }

  /**
   * Moves the robot from currentPose to targetPose through a series of waypoints.
   * 
   * @param targetPose
   * @return The result of the last move command.
   */
  public Result navigateThrough(Pose targetPose) {
    Pose currentPose = getCurrentPose();
    List<Pose> poses = interpolate(currentPose, targetPose);
    Result result = null;

    for (Pose pose : poses) {
      result = moveTo(pose);
    }

    Log.i(TAG, "Move to: " + targetPose.toString() + " Result: " + result.getMessage());
    return result;
  }

  /**
   * Moves the robot to the pose of taking photo in area 1.
   * 
   * @return The result of the last move command.
   * 
   * Example:
   * @code
   * // Area 1
   * navigator.navigateToArea(1);
   * visionHandler.getCurrentPose(navigator.getCurrentPose());
   * visionHandler.inspectArea();

   * // Area 2
   * navigator.navigateToArea(2);
   * visionHandler.getCurrentPose(navigator.getCurrentPose());
   * visionHandler.inspectArea();
   * @endcode
   */
  public Result navigateToArea(int area) {
    Point point;
    Quaternion quaternion;
    Pose pose;
    Result result;
    switch (area) {

    case 1:
      point = new Point(10.8, -9.78, 4.7);
      quaternion = new Quaternion(-0.172f, -0.208f, -0.615f, 0.741f);
      break;

    case 2:
      point = new Point(10.8, -8.875, 4.56);
      quaternion = new Quaternion(-0.537f, 0.46f, 0.46f, 0.537f);
      break;

    case 3:
      point = new Point(10.8, -7.925, 4.56);
      quaternion = new Quaternion(-0.537f, 0.46f, 0.46f, 0.537f);
      break;

    case 4:
      point = new Point(10.667, -6.853, 4.9654);
      quaternion = new Quaternion(-0.013f, 0.0f, 1.0f, 0.0f);
      break;

    case 5: // Report
      point = new Point(11.143, -6.7607, 4.9654);
      quaternion = new Quaternion(0.0f, 0.0f, 0.707f, 0.707f);
      break;

    default:
      Log.w(TAG, "Unknown area Id, navigation aborted.");
      return null;
    }

    result = moveTo(new Pose(point, quaternion));

    Log.i(TAG, "Move to area " + area);
    return result;
  }

  /**
   * Moves the robot to find the treasure item.
   * 
   * @return The result of the last move command.
   */
  public Result navigateToTreasure(Item treasureItem) {
    int areaId = treasureItem.getAreaId();

    Pose currentPose = getCurrentPose();
    Pose treasurePose = treasureItem.getItemPose();

    Point currentPoint = currentPose.getPoint();
    Point treasurePoint = treasurePose.getPoint();

    // The final point should be within 0.9 m vertically from the plane of the area.
    double safeDistance = 0.8;

    double t;
    double currentX = currentPoint.getX();
    double currentY = currentPoint.getY();
    double currentZ = currentPoint.getZ();
    double finalX = 0, finalY = 0, finalZ = 0;
    double treasureX = 0, treasureY = 0, treasureZ = 0;

    switch (areaId) {
      case 1:
        // Area1 lies on the XZ plane, so the vertical distance along the Y-axis should be within 0.9 m
        finalY = -10.58 + safeDistance;

        // Treasure point
        treasureX = treasurePoint.getX();
        treasureY = -10.58; // or treasurePoint.getY(), not sure whether to trust the rule book or the ARTagDetector
        treasureZ = treasurePoint.getZ();

        // Interpolate
        t = (finalY - currentY) / (treasureY - currentY);
        finalX = currentX + t * (treasureX - currentX);
        finalZ = currentZ + t * (treasureZ - currentZ);
        break;

      case 2:
      case 3:
        // Area2 and Area3 lie on the XY plane, so the vertical distance along the Z-axis should be within 0.9 m
        finalZ = 3.76203 + safeDistance;

        // Treasure point
        treasureX = treasurePoint.getX();
        treasureY = treasurePoint.getY();
        treasureZ = 3.76203; // or treasurePoint.getZ();

        // Interpolate
        t = (finalZ - currentZ) / (treasureZ - currentZ);
        finalX = currentX + t * (treasureX - currentX);
        finalY = currentY + t * (treasureY - currentY);   
        break;

      case 4:
        // Area4 lies on the YZ plane, so the vertical distance along the X-axis should be within 0.9 m
        finalX = 9.866984 + safeDistance;

        // Treasure point
        treasureX = 9.866984; // or treasurePoint.getX();
        treasureY = treasurePoint.getY();
        treasureZ = treasurePoint.getZ();

        // Interpolate
        t = (finalX - currentX) / (treasureX - currentX);
        finalY = currentY + t * (treasureY - currentY);
        finalZ = currentZ + t * (treasureZ - currentZ);
        break;

      default:
        // Handle error
        Log.w(TAG, String.format("Invalid areaId: %d. Using default final pose.", areaId));
        
        // Guess the item is in the middle of area 4
        // Area 4: (9.866984, -7.34, 4.32, 9.866984, -6.365, 5.57)
        finalX = 9.866984 + safeDistance;

        // Treasure point
        treasureX = 9.866984;
        treasureY = (-7.34 - 6.365) / 2;
        treasureZ = (4.32 + 5.57) / 2;
        
        // Interpolate
        t = (finalX - currentX) / (treasureX - currentX);
        finalY = currentY + t * (treasureY - currentY);
        finalZ = currentZ + t * (treasureZ - currentZ);
        break;
    }
    
    // Set the final Quaternion to face the treasure
    Point finalPoint = new Point(finalX, finalY, finalZ);
    Quaternion finalQuaternion = getQuaternionTo(currentPoint, finalPoint);
    Pose finalPose = new Pose(finalPoint, finalQuaternion);
    Log.i(TAG, "I'm goint to " + finalPose.toString());

    Result result = moveTo(finalPose);

    Log.i(TAG, "Move to find the treasure.");
    return result;
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
   * Calculates the quaternion to rotate from the current point to face the target point.
   * 
   * @param currentPoint The current point.
   * @param targetPoint The target point.
   * @return A new quaternion that rotates current point to face the target point.
   */
  public static Quaternion getQuaternionTo(Point currentPoint, Point targetPoint) {
    // Step 1: Compute forward vector
    float fx = (float) targetPoint.getX() - (float) currentPoint.getX();
    float fy = (float) targetPoint.getY() - (float) currentPoint.getY();
    float fz = (float) targetPoint.getZ() - (float) currentPoint.getZ();
    float fMagnitude = (float) Math.sqrt(fx * fx + fy * fy + fz * fz);

    // Already face target
    if (fMagnitude < 1e-6) {
      return new Quaternion(0.0f, 0.0f, 0.0f, 1.0f);
    }

    fx /= fMagnitude;
    fy /= fMagnitude;
    fz /= fMagnitude;

    // Step 2: Default up vector
    float ux = 0f;
    float uy = 0f;
    float uz = 1f;

    // Step 3: Compute right vector (up × forward)
    float rx = uy * fz - uz * fy;
    float ry = uz * fx - ux * fz;
    float rz = ux * fy - uy * fx;
    float rMagnitude = (float) Math.sqrt(rx * rx + ry * ry + rz * rz);

    if (rMagnitude < 1e-6) {
      // Forward and up are too close，change default up vector to  (0, 1, 0)
      ux = 0f;
      uy = 1f;
      uz = 0f;
      rx = uy * fz - uz * fy;
      ry = uz * fx - ux * fz;
      rz = ux * fy - uy * fx;
      rMagnitude = (float) Math.sqrt(rx * rx + ry * ry + rz * rz);
    }

    rx /= rMagnitude;
    ry /= rMagnitude;
    rz /= rMagnitude;

    // Step 4: Compute true up (forward × right)
    float tx = fy * rz - fz * ry;
    float ty = fz * rx - fx * rz;
    float tz = fx * ry - fy * rx;

    // Step 5: Compute rotation matrix
    float[][] rot = {
      {fx, rx, tx},
      {fy, ry, ty},
      {fz, rz, tz}
    };

    // Step 6: Transform ratation matrix to quarternion
    float trace = rot[0][0] + rot[1][1] + rot[2][2];
    float qw, qx, qy, qz;

    if (trace > 0f) {
      float s = (float) (0.5f / Math.sqrt(trace + 1.0f));
      qw = 0.25f / s;
      qx = (rot[2][1] - rot[1][2]) * s;
      qy = (rot[0][2] - rot[2][0]) * s;
      qz = (rot[1][0] - rot[0][1]) * s;
    } else if (rot[0][0] > rot[1][1] && rot[0][0] > rot[2][2]) {
      float s = (float) (2.0f * Math.sqrt(1.0f + rot[0][0] - rot[1][1] - rot[2][2]));
      qw = (rot[2][1] - rot[1][2]) / s;
      qx = 0.25f * s;
      qy = (rot[0][1] + rot[1][0]) / s;
      qz = (rot[0][2] + rot[2][0]) / s;
    } else if (rot[1][1] > rot[2][2]) {
      float s = (float) (2.0f * Math.sqrt(1.0f + rot[1][1] - rot[0][0] - rot[2][2]));
      qw = (rot[0][2] - rot[2][0]) / s;
      qx = (rot[0][1] + rot[1][0]) / s;
      qy = 0.25f * s;
      qz = (rot[1][2] + rot[2][1]) / s;
    } else {
      float s = (float) (2.0f * Math.sqrt(1.0f + rot[2][2] - rot[0][0] - rot[1][1]));
      qw = (rot[1][0] - rot[0][1]) / s;
      qx = (rot[0][2] + rot[2][0]) / s;
      qy = (rot[1][2] + rot[2][1]) / s;
      qz = 0.25f * s;
    }

    return new Quaternion(qx, qy, qz, qw);
  }

  /**
   * Given the pose of Navcam, compute the body pose.
   * 
   * @param cameraPose the pose of NavCam.
   * @return The pose of center body. 
   */
  public static Pose getBodyPoseFromCamera(Pose cameraPose) {
    Point cameraPoint = cameraPose.getPoint();
    
    double bodyX = cameraPoint.getX() - 0.1177;
    double bodyY = cameraPoint.getY() + 0.0422;
    double bodyZ = cameraPoint.getZ() + 0.0826;

    Pose bodyPose = new Pose(new Point(bodyX, bodyY, bodyZ), cameraPose.getQuaternion());

    return bodyPose;
  }
}
