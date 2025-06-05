package jp.jaxa.iss.kibo.rpc.sampleapk;

import jp.jaxa.iss.kibo.rpc.api.KiboRpcService;
import jp.jaxa.iss.kibo.rpc.api.KiboRpcApi;

import gov.nasa.arc.astrobee.types.Point;
import gov.nasa.arc.astrobee.types.Quaternion;
import gov.nasa.arc.astrobee.Kinematics;
import gov.nasa.arc.astrobee.Result;

import java.util.List;
import java.util.ArrayList;
import java.util.Map;
import java.util.HashMap;

import android.util.Log;


/**
 * Class to handle navigation commands for the Astrobee robot.
 */
public class Navigator {
  private final KiboRpcApi api;
  private final String TAG = this.getClass().getSimpleName();

  // Target poses in each area
  private static final Map<Integer, Pose> areaPoses = new HashMap<>();
  static {
    areaPoses.put(1, new Pose(new Point(11.11, -9.49, 5.435), new Quaternion(0.0f, 0.0f, -0.707f, 0.707f)));          // Area 1
    areaPoses.put(2, new Pose(new Point(10.925, -8.6, 4.55), new Quaternion(0.5f, 0.5f, -0.5f, 0.5f)));               // Area 2
    areaPoses.put(3, new Pose(new Point(10.925, -7.925, 4.462), new Quaternion(0.5f, 0.5f, -0.5f, 0.5f)));            // Area 3
    areaPoses.put(4, new Pose(new Point(11.35, -6.76, 4.935), new Quaternion(0.0f, -1.0f, 0.0f, 0.0f)));              // Area 4
    areaPoses.put(5, new Pose(new Point(10.925, -8.35, 5.3), new Quaternion(0.0f, 0.707f, 0.0f, 0.707f)));            // Combined Area 2 3
    areaPoses.put(0, new Pose(new Point(11.35, -6.76, 4.935), new Quaternion(0.633f, 0.754f, -0.133f, 0.112f)));      // Report
  }

  // Safety factors
  private static final double safeDistance = 0.8; // Vertical distance to the plane of the area.
  private static final double subSafeDistance = 0.05; // Distance to the boundary of the projected zone of the area.

  // Boundaries
  private static final double area1MaxX = 11.48;
  private static final double area1MinX = 10.42;
  private static final double area1MaxZ = 5.57;
  private static final double area1MinZ = 4.82;

  private static final double area2MaxX = 11.55;
  private static final double area2MinX = 10.3;
  private static final double area2MaxY = -8.5;
  private static final double area2MinY = -9.25;

  private static final double area3MaxX = 11.55;
  private static final double area3MinX = 10.3;
  private static final double area3MaxY = -7.45;
  private static final double area3MinY = -8.4;

  private static final double area4MaxY = -6.365;
  private static final double area4MinY = -7.34;
  private static final double area4MaxZ = 5.57;
  private static final double area4MinZ = 4.32;

  /**
   * Constructor
   * 
   * @param apiRef API reference from runPlan in YourService.java
   */
  public Navigator(KiboRpcApi apiRef) {
    this.api = apiRef;
    Log.i(TAG,  "Initialized.");
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
    int numAttempts = 5;
    double accPosX = 0, accPosY = 0, accPosZ = 0;
    double accOriX = 0, accOriY = 0, accOriZ = 0, accOriW = 0;

    // Collect kinematics data from API
    for (int i = 0; i < numAttempts; i++) {
      Kinematics kinematics = api.getRobotKinematics();

      Point position = kinematics.getPosition();
      accPosX += position.getX();
      accPosY += position.getY();
      accPosZ += position.getZ();

      Quaternion orientation = kinematics.getOrientation();
      accOriX += (double) orientation.getX();
      accOriY += (double) orientation.getY();
      accOriZ += (double) orientation.getZ();
      accOriW += (double) orientation.getW();
    }
    
    // Compute avergae
    double avgPosX = accPosX / numAttempts;
    double avgPosY = accPosY / numAttempts;
    double avgPosZ = accPosZ / numAttempts;
    
    float avgOriX = (float) accOriX / numAttempts;
    float avgOriY = (float) accOriY / numAttempts;
    float avgOriZ = (float) accOriZ / numAttempts;
    float avgOriW = (float) accOriW / numAttempts;

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
    Result result = api.moveTo(targetPose.getPoint(), targetPose.getQuaternion(), false);
    if (result.hasSucceeded()) {
      return result;
    }

    // Enter retry loop if failed
    int retryMax = 5;
    for (int retry = 1; retry <= retryMax; retry++) {
      Log.i(TAG, "Moving to targetPose " + targetPose.toString() + " (retry " + retry + ")");
      result = api.moveTo(targetPose.getPoint(), targetPose.getQuaternion(), false);
      
      if (result.hasSucceeded()) {
        return result;
      }
    }

    // If still not successful after retries, log the error.
    if (!result.hasSucceeded()) {
      Log.w(TAG, "Failed to move to " + targetPose.toString() + " because " + result.getMessage());
    }
    
    return result;
  }

  /**
   * Moves the robot to the pose of taking photo in given area.
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
    Pose targetPose = areaPoses.get(area);

    if (targetPose == null) {
      Log.w(TAG, "Unknown area Id: " + area + ", navigation aborted.");
      return null;
    }

    Log.i(TAG, "Start moving to area " + area);
    Result result = moveTo(targetPose);

    int stableTime = 0;
    switch (area) {
      case 0:
        stableTime = 700;
        break;
      case 1:
        stableTime = 1000;
        break;
      case 4:
        stableTime = 2000;
        break;
      case 5:
        stableTime = 2400;
        break;
      default:
        stableTime = 700;
        break;
    }

    // Wait Stable
    try{
      Thread.sleep(stableTime);
    } catch (InterruptedException e) {
      Log.w(TAG, "Fail to sleep thread" + e);
    }
    
    return result;
  }

  /**
   * Moves the robot to find the treasure item.
   * 
   * @return The result of the last move command.
   */
  public Result navigateToTreasure(Item treasureItem) {
    int areaId = treasureItem.getAreaId();

    Pose treasurePose = treasureItem.getItemPose();
    Point treasurePoint = treasurePose.getPoint();

    Point finalPoint = null;
    Quaternion finalQuaternion = null;
    double finalX = 0, finalY = 0, finalZ = 0;

    switch (areaId) {
      case 1:
        // Area1 lies on the XZ plane, so the vertical distance along the Y-axis should be within 0.9 m
        finalY = -10.58 + safeDistance;

        finalX = treasurePoint.getX();
        finalX = Math.min(finalX, area1MaxX - subSafeDistance);
        finalX = Math.max(finalX, area1MinX + subSafeDistance);

        finalZ = treasurePoint.getZ();
        finalZ = Math.min(finalZ, area1MaxZ - subSafeDistance);
        finalZ = Math.max(finalZ, area1MinZ + subSafeDistance);

        finalPoint = new Point(finalX, finalY, finalZ);
        finalQuaternion = new Quaternion(0.0f, 0.0f, -0.707f, 0.707f);

        break;

      case 2:
        // Area2 lies on the XY plane, so the vertical distance along the Z-axis should be within 0.9 m
        finalZ = 3.76203 + safeDistance;

        finalX = treasurePoint.getX();
        finalX = Math.min(finalX, area2MaxX - subSafeDistance);
        finalX = Math.max(finalX, area2MinX + subSafeDistance);

        finalY = treasurePoint.getY();
        finalY = Math.min(finalY, area2MaxY - subSafeDistance);
        finalY = Math.max(finalY, area2MinY + subSafeDistance);

        finalPoint = new Point(finalX, finalY, finalZ);
        finalQuaternion = new Quaternion(0.5f, 0.5f, -0.5f, 0.5f);

        break;

      case 3:
        // Area3 lies on the XY plane, so the vertical distance along the Z-axis should be within 0.9 m
        finalZ = 3.76203 + safeDistance;

        finalX = treasurePoint.getX();
        finalX = Math.min(finalX, area3MaxX - subSafeDistance);
        finalX = Math.max(finalX, area3MinX + subSafeDistance);

        finalY = treasurePoint.getY();
        finalY = Math.min(finalY, area3MaxY - subSafeDistance);
        finalY = Math.max(finalY, area3MinY + subSafeDistance);

        finalPoint = new Point(finalX, finalY, finalZ);
        finalQuaternion = new Quaternion(0.5f, 0.5f, -0.5f, 0.5f);

        break;

      case 4:
        // Area4 lies on the YZ plane, so the vertical distance along the X-axis should be within 0.9 m
        finalX = 9.866984 + safeDistance;

        finalY = treasurePoint.getY();
        finalY = Math.min(finalY, area4MaxY - subSafeDistance);
        finalY = Math.max(finalY, area4MinY + subSafeDistance);

        finalZ = treasurePoint.getZ();
        finalZ = Math.min(finalZ, area4MaxZ - subSafeDistance);
        finalZ = Math.max(finalZ, area4MinZ + subSafeDistance);

        finalPoint = new Point(finalX, finalY, finalZ);
        finalQuaternion = new Quaternion(0.0f, -1.0f, 0.0f, 0.0f);

        break;

      default:
        // Handle error
        Log.w(TAG, String.format("Invalid areaId: %d. Using default final pose.", areaId));
        
        // Guess the item is in the middle of area 4
        // Area 4: (9.866984, -7.34, 4.32, 9.866984, -6.365, 5.57)
        finalX = 9.866984 + safeDistance;
        finalY = (-7.34 - 6.365) / 2;
        finalZ = (4.32 + 5.57) / 2;

        finalPoint = new Point(finalX, finalY, finalZ);
        finalQuaternion = new Quaternion(0.0f, -1.0f, 0.0f, 0.0f);

        break;
    }
    
    // Set the final Pose
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
