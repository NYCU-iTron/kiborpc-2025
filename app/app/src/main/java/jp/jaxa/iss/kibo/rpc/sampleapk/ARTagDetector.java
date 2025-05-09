package jp.jaxa.iss.kibo.rpc.sampleapk;

import jp.jaxa.iss.kibo.rpc.api.KiboRpcApi;

import gov.nasa.arc.astrobee.types.Point;
import gov.nasa.arc.astrobee.types.Quaternion;

import java.util.List;
import java.util.Map;
import java.util.HashMap;
import java.util.ArrayList;
import java.util.Collections;
import android.util.Log;

import org.opencv.core.Mat;
import org.opencv.core.CvType;
import org.opencv.aruco.Aruco;
import org.opencv.calib3d.Calib3d;
import org.opencv.aruco.Dictionary;
import org.opencv.core.MatOfDouble;
import org.opencv.imgproc.Imgproc;
import org.opencv.core.MatOfPoint2f;


/**
 * Class to extract AR tag information from a given image.
 */
public class ARTagDetector {
  private final KiboRpcApi api;
  private final String TAG = this.getClass().getSimpleName();
  private final Dictionary arucoDictionary;
  private final float markerLength;
  private final Mat cameraMatrix;
  private final Mat distortionCoefficients;

  /**
   * Constructor
   * 
   * @param apiRef API reference.
   */
  public ARTagDetector(KiboRpcApi apiRef) {
    this.api = apiRef;
    markerLength = 0.05f;
    cameraMatrix = getCameraMatrix();
    distortionCoefficients = getDistortionCoefficients();
    arucoDictionary = Aruco.getPredefinedDictionary(Aruco.DICT_5X5_250);

    Log.i(TAG, "Initialized");
  }

  /**
   * Detect AR tags in the given image.
   * 
   * @param undistortedImage The input image.
   * @return A list of detected AR tags.
   */
  public List<ARTag> detect(Mat undistortedImage) {
    List<Mat> corners = new ArrayList<>();
    Mat markerIds = new Mat();

    Aruco.detectMarkers(undistortedImage, arucoDictionary, corners, markerIds);
  
    if (markerIds.empty()) {
      Log.w(TAG, "No AR Tag Found.");
      return new ArrayList<>();
    }

    List<ARTag> results = new ArrayList<>();
    for (int i = 0; i < markerIds.rows(); i++) {
      int markerId = (int) markerIds.get(i, 0)[0];
      Mat corner = corners.get(i);
      ARTag arTag = new ARTag(markerId, corner);
      results.add(arTag);
    }

    return results;
  }

  public Map<Integer, Pose> filterResult(List<ARTag> arResults, Pose currentPose) {
    Map<Integer, Pose> tagPoses = new HashMap<>();

    for (ARTag arTag : arResults) {
      int markerId = arTag.getMarkerId();
      Mat corner = arTag.getCorner();

      Mat rvec = new Mat();
      Mat tvec = new Mat();

      // Estimate pose
      Aruco.estimatePoseSingleMarkers(Collections.singletonList(corner), markerLength, cameraMatrix, distortionCoefficients, rvec, tvec);

      // Extract shift matrix
      double[] t = tvec.get(0, 0);
      Point point = new Point(t[0], t[1], t[2]);

      // Extract rotation matrix
      double[] r = rvec.get(0, 0);
      Mat rotMat = new Mat();
      Calib3d.Rodrigues(new MatOfDouble(r), rotMat);
      Quaternion quaternion = new Quaternion(); // rotationMatrixToQuaternion(rotMat); this param is never used, so keep it simple.

      Pose tagPoseCamera = new Pose(point, quaternion);
      Pose tagPoseWorld = convertCameraToWorld(tagPoseCamera, currentPose);
      tagPoses.put(markerId, tagPoseWorld);
    }

    return tagPoses;
  }

  public Pose guessResult(int areaId) {
    Log.w(TAG, "No correct tag found, set tagPose to the center of this area.");
    
    Pose tagPose; 

    if (areaId == 1) {
      tagPose = new Pose(new Point((10.42 + 11.48) / 2, -10.58, (4.82 + 5.57) / 2), new Quaternion(1.0f, 0.0f, 0.0f, 0.0f));
    } else if (areaId == 2) {
      tagPose = new Pose(new Point((10.3 + 11.55) / 2, (-9.25 + -8.5) / 2, 3.76203), new Quaternion(1.0f, 0.0f, 0.0f, 0.0f));
    } else if (areaId == 3) {
      tagPose = new Pose(new Point((10.3 + 11.55) / 2, (-8.4 + -7.45) / 2, 3.76203), new Quaternion(1.0f, 0.0f, 0.0f, 0.0f));
    } else if (areaId == 4) {
      tagPose = new Pose(new Point(9.866984, (-7.34 + -6.365) / 2, (4.32 + 5.57) / 2), new Quaternion(1.0f, 0.0f, 0.0f, 0.0f));
    } else if (areaId == 0) {
      tagPose = new Pose();
    } else {
      tagPose = new Pose();
      Log.i(TAG, "Area id not valid.");
    }

    return tagPose;
  }

  /**
   * Get the clipped image of the detected AR tag.
   * 
   * @param undistortedImage The input image.
   * @return The clipped image.
   */
  public Map<Integer, Mat> getclippedImages(List<ARTag> arResults, Mat undistortedImage) {
    if (arResults.isEmpty()) {
      Log.w(TAG, "No AR Tag Found.");
      return null;
    }

    Map<Integer, Mat> clippedImages = new HashMap<>();
    for(ARTag arTag : arResults) {
      int markerId = arTag.getMarkerId();
      Mat corner = arTag.getCorner();

      // Clip the image
      Mat clippedImage = clip(undistortedImage, corner);
      if (clippedImage.empty()) {
        Log.w(TAG, "Clipped image is empty.");
        continue;
      }

      clippedImages.put(markerId, clippedImage);
    }
    return clippedImages;
  }

  /* ----------------------------- Tool Functions ----------------------------- */

  private Mat getCameraMatrix() {
    Mat cameraMatrix = new Mat(3, 3, CvType.CV_64F);
    cameraMatrix.put(0, 0, api.getNavCamIntrinsics()[0]);
    return cameraMatrix;
  }

  private Mat getDistortionCoefficients() {
    Mat distCoeffs = new Mat(1, 5, CvType.CV_64F);
    distCoeffs.put(0, 0, api.getNavCamIntrinsics()[1]);
    return distCoeffs;
  }

  private Quaternion rotationMatrixToQuaternion(Mat rotMat) {
    double[][] m = new double[3][3];
    for (int i = 0; i < 3; i++) {
      for (int j = 0; j < 3; j++) {
        m[i][j] = rotMat.get(i, j)[0];
      }
    }

    double trace = m[0][0] + m[1][1] + m[2][2];
    double w, x, y, z;

    if (trace > 0) {
      double s = 0.5 / Math.sqrt(trace + 1.0);
      w = 0.25 / s;
      x = (m[2][1] - m[1][2]) * s;
      y = (m[0][2] - m[2][0]) * s;
      z = (m[1][0] - m[0][1]) * s;
    } else {
      if (m[0][0] > m[1][1] && m[0][0] > m[2][2]) {
        double s = 2.0 * Math.sqrt(1.0 + m[0][0] - m[1][1] - m[2][2]);
        w = (m[2][1] - m[1][2]) / s;
        x = 0.25 * s;
        y = (m[0][1] + m[1][0]) / s;
        z = (m[0][2] + m[2][0]) / s;
      } else if (m[1][1] > m[2][2]) {
        double s = 2.0 * Math.sqrt(1.0 + m[1][1] - m[0][0] - m[2][2]);
        w = (m[0][2] - m[2][0]) / s;
        x = (m[0][1] + m[1][0]) / s;
        y = 0.25 * s;
        z = (m[1][2] + m[2][1]) / s;
      } else {
        double s = 2.0 * Math.sqrt(1.0 + m[2][2] - m[0][0] - m[1][1]);
        w = (m[1][0] - m[0][1]) / s;
        x = (m[0][2] + m[2][0]) / s;
        y = (m[1][2] + m[2][1]) / s;
        z = 0.25 * s;
      }
    }
    
    // Normalize
    float norm = (float) Math.sqrt(w * w + x * x + y * y + z * z);
    if (norm > 1e-6) { // Avoid division by zero
      w /= norm;
      x /= norm;
      y /= norm;
      z /= norm;
    } else {
      return new Quaternion(0, 0, 0, 1); // Fallback to identity
    }

    return new Quaternion((float) w, (float) x, (float) y, (float) z);
  }

  /**
   * Convert the pose in NavCam frame to World frame
   * Tag_Camera → Camera_Body → Body_World = Tag_World
   * 
   * @param tagInCamera The tag pose in NavCam frame.
   * @return The pose in world frame. 
   */
  private Pose convertCameraToWorld(Pose tagInCVcam, Pose bodyInWorld) {
    // OpenCV camera pose in NavCam frame
    Pose CVcamInNavcam = new Pose(
      new Point(0, 0, 0),
      new Quaternion(0.5f, 0.5f, 0.5f, 0.5f)
    );

    // Step 1: Convert tag pose from OpenCV camera frame to NavCam frame
    Pose tagInNavcam = composePoses(CVcamInNavcam, tagInCVcam);

    // NavCam pose in body frame (fixed offset and orientation)
    Pose navcamInBody = new Pose(
      new Point(0.1177, -0.0422, -0.0826),
      new Quaternion(0.0f, 0.0f, 0.0f, 1.0f)
    );

    // Step 2: Convert tag pose from NavCam frame to body frame
    Pose tagInBody = composePoses(navcamInBody, tagInNavcam);

    // Step 3: Convert tag pose from body frame to world frame
    Pose tagInWorld = composePoses(bodyInWorld, tagInBody);

    return tagInWorld;
  }

  private Pose composePoses(Pose poseA, Pose poseB) {
    // poseA followed by poseB
    Quaternion qa = poseA.getQuaternion();
    Point pa = poseA.getPoint();

    Quaternion qb = poseB.getQuaternion();
    Point pb = poseB.getPoint();

    // 1. Rotate B's position by A's orientation
    double[] rotated = rotatePointByQuaternion(pb.getX(), pb.getY(), pb.getZ(), qa);
    double x = pa.getX() + rotated[0];
    double y = pa.getY() + rotated[1];
    double z = pa.getZ() + rotated[2];

    // 2. Combine orientation: qa * qb
    Quaternion q = multiplyQuaternions(qa, qb);

    return new Pose(new Point(x, y, z), q);
  }

  private double[] rotatePointByQuaternion(double x, double y, double z, Quaternion q) {
    // Quaternion -> rotation matrix
    double x2 = q.getX() * q.getX();
    double y2 = q.getY() * q.getY();
    double z2 = q.getZ() * q.getZ();
    double xy = q.getX() * q.getY();
    double xz = q.getX() * q.getZ();
    double yz = q.getY() * q.getZ();
    double wx = q.getW() * q.getX();
    double wy = q.getW() * q.getY();
    double wz = q.getW() * q.getZ();

    double r11 = 1 - 2 * (y2 + z2);
    double r12 = 2 * (xy - wz);
    double r13 = 2 * (xz + wy);
    double r21 = 2 * (xy + wz);
    double r22 = 1 - 2 * (x2 + z2);
    double r23 = 2 * (yz - wx);
    double r31 = 2 * (xz - wy);
    double r32 = 2 * (yz + wx);
    double r33 = 1 - 2 * (x2 + y2);

    double rx = r11 * x + r12 * y + r13 * z;
    double ry = r21 * x + r22 * y + r23 * z;
    double rz = r31 * x + r32 * y + r33 * z;

    return new double[]{rx, ry, rz};
  }

  private Quaternion multiplyQuaternions(Quaternion q1, Quaternion q2) {
    float w1 = q1.getW(), x1 = q1.getX(), y1 = q1.getY(), z1 = q1.getZ();
    float w2 = q2.getW(), x2 = q2.getX(), y2 = q2.getY(), z2 = q2.getZ();

    float w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2;
    float x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2;
    float y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2;
    float z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2;

    // Normalize
    float norm = (float) Math.sqrt(w * w + x * x + y * y + z * z);
    if (norm > 1e-6) { // Avoid division by zero
      w /= norm;
      x /= norm;
      y /= norm;
      z /= norm;
    } else {
      return new Quaternion(0, 0, 0, 1); // Fallback to identity
    }

    return new Quaternion(x, y, z, w);
  }


  /**
   * Clip the image to the area of interest.
   * 
   * @param image The input image.
   * @param corner The corners of the area of interest.
   * @return The clipped image.
   */
  private Mat clip(Mat image, Mat corner) {
    final org.opencv.core.Point[] points = new org.opencv.core.Point[4];

    // point[0] left top
    // point[1] right top
    // point[2] right bottom
    // point[3] left bottom
    for (int i = 0; i < 4; i++) {
      points[i] = new org.opencv.core.Point(corner.get(0, i));
    }
    double[] corner0 = corner.get(0, 0);
    double[] corner1 = corner.get(0, 1);
    double[] corner2 = corner.get(0, 2);
    double[] corner3 = corner.get(0, 3);

    // Calculate the width and height of the rectangle
    points[0] = new org.opencv.core.Point(-103 / 20 * (corner1[0] - corner0[0]) - 5 / 4 * (corner3[0] - corner0[0]) + corner0[0], -103 / 20 * (corner1[1] - corner0[1]) - 5 / 4 * (corner3[1] - corner0[1]) + corner0[1]);
    points[1] = new org.opencv.core.Point(17 / 20 * (corner1[0] - corner0[0]) - 5 / 4 * (corner3[0] - corner0[0]) + corner0[0], 17 / 20 * (corner1[1] - corner0[1]) - 5 / 4 * (corner3[1] - corner0[1]) + corner0[1]);
    points[2] = new org.opencv.core.Point(17 / 20 * (corner1[0] - corner0[0]) + 15 / 4 * (corner3[0] - corner0[0]) + corner0[0], 17 / 20 * (corner1[1] - corner0[1]) + 15 / 4 * (corner3[1] - corner0[1]) + corner0[1]);
    points[3] = new org.opencv.core.Point(-103 / 20 * (corner1[0] - corner0[0]) + 15 / 4 * (corner3[0] - corner0[0]) + corner0[0], -103 / 20 * (corner1[1] - corner0[1]) + 15 / 4 * (corner3[1] - corner0[1]) + corner0[1]);

    double width = Math.sqrt(Math.pow(points[0].x - points[1].x, 2) + Math.pow(points[0].y - points[1].y, 2));
    double height = Math.sqrt(Math.pow(points[0].x - points[3].x, 2) + Math.pow(points[0].y - points[3].y, 2));

    // Create a transformation matrix
    Mat transformMatrix;
    {
      MatOfPoint2f srcPoints = new MatOfPoint2f(points);
      srcPoints.convertTo(srcPoints, CvType.CV_32F);

      MatOfPoint2f dstPoints = new MatOfPoint2f(
        new org.opencv.core.Point(0, 0),
        new org.opencv.core.Point(width - 1, 0),
        new org.opencv.core.Point(width - 1, height - 1),
        new org.opencv.core.Point(0, height - 1)
      );

      // Convert to float
      dstPoints.convertTo(dstPoints, CvType.CV_32F);

      // Get the perspective transform matrix
      transformMatrix = Imgproc.getPerspectiveTransform(srcPoints, dstPoints);
    }

    // Apply the perspective transformation
    Mat clippedImage = Mat.zeros((int) height, (int) width, image.type());
    Imgproc.warpPerspective(image, clippedImage, transformMatrix, clippedImage.size());

    return clippedImage;
  }
}
