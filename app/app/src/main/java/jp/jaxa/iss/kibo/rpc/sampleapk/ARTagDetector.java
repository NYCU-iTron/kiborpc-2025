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
import org.opencv.core.Core;


/**
 * Class to extract AR tag information from a given image.
 */
public class ARTagDetector {
  private final KiboRpcApi api;
  private final String TAG = this.getClass().getSimpleName();
  private final Dictionary arucoDictionary;
  public final float markerLength;
  private final Mat cameraMatrix;
  private final Mat distortionCoefficients;

  /**
   * Class to represent an ARTag.
   */
  public class ARTag {
    private int markerId;
    private Mat corner;

    public ARTag() {
      this.markerId = -1;
      this.corner = null;
    }

    public ARTag(int markerId, Mat corner) {
      this.markerId = markerId;
      this.corner = corner;
    }

    public int getMarkerId() { return markerId; }
    public Mat getCorner() { return corner; }
  }

  /**
   * Helper class to store results from pose estimation for a single AR tag.
   */
  public static class ARTagPoseEstimation {
    public final int markerId;
    public final Pose poseInCamera;
    public final Pose poseInWorld;
    public final Mat rvec;
    public final Mat tvec;

    public ARTagPoseEstimation(int markerId, Pose poseInCamera, Pose poseInWorld, Mat rvec, Mat tvec) {
      this.markerId = markerId;
      this.poseInCamera = poseInCamera;
      this.poseInWorld = poseInWorld;
      this.rvec = rvec;
      this.tvec = tvec;
    }
  }
  
  /**
   * Helper class to store the result of clipping an image.
   */
  public static class ClippedImageResult {
    public final Mat clippedImage;
    public final Mat transformMatrix;
    public final Mat inverseTransformMatrix;

    public ClippedImageResult(Mat clippedImage, Mat transformMatrix) {
      this.clippedImage = clippedImage;
      this.transformMatrix = transformMatrix;
      Mat invMat = new Mat();
      if (transformMatrix != null && !transformMatrix.empty()) {
        Core.invert(transformMatrix, invMat);
      }
      this.inverseTransformMatrix = invMat;
    }
  }


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

    Log.i(TAG, "Initialized ARTagDetector");
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
    markerIds.release();
    for(Mat corner : corners) corner.release();
    return results;
  }

  /**
   * Estimates poses of AR tags.
   * @param arResults List of detected ARTags.
   * @param currentRobotPoseInWorld Current robot pose in world coordinates.
   * @return Map of markerId to ARTagPoseEstimation object.
   */
  public Map<Integer, ARTagPoseEstimation> estimateTagPoses(List<ARTag> arResults, Pose currentRobotPoseInWorld) {
    Map<Integer, ARTagPoseEstimation> tagEstimations = new HashMap<>();

    for (ARTag arTag : arResults) {
      int markerId = arTag.getMarkerId();
      Mat corner = arTag.getCorner();

      Mat rvec = new Mat();
      Mat tvec = new Mat();

      // Estimate pose of the AR Tag in Camera Coordinates
      Aruco.estimatePoseSingleMarkers(Collections.singletonList(corner), markerLength, cameraMatrix, distortionCoefficients, rvec, tvec);

      double[] t_cam = tvec.get(0, 0);
      Point point_cam = new Point(t_cam[0], t_cam[1], t_cam[2]);

      double[] r_cam = rvec.get(0, 0);
      Mat rotMat_cam = new Mat();
      Calib3d.Rodrigues(new MatOfDouble(r_cam), rotMat_cam);
      // IMPORTANT: Corrected Quaternion order for gov.nasa.arc.astrobee.types.Quaternion(x,y,z,w)
      Quaternion quaternion_cam = rotationMatrixToQuaternion(rotMat_cam); 

      Pose tagPoseInCamera = new Pose(point_cam, quaternion_cam);
      Pose tagPoseInWorld = convertCameraToWorld(tagPoseInCamera, currentRobotPoseInWorld);
      
      tagEstimations.put(markerId, new ARTagPoseEstimation(markerId, tagPoseInCamera, tagPoseInWorld, rvec.clone(), tvec.clone()));
      
      // Release Mats to prevent memory leaks
      rotMat_cam.release();
    }
    // rvec and tvec are released implicitly if not cloned, or when ARTagPoseEstimation objects are GC'd if cloned.
    // Let's ensure rvec and tvec from estimatePoseSingleMarkers are cloned if stored, or released if not.
    // Here, cloning ensures they are independent.
    return tagEstimations;
  }


  public Pose guessResult(int areaId) {
    Pose tagPose; 

    if (areaId == 1) {
      tagPose = new Pose(new Point((10.42 + 11.48) / 2, -10.58, (4.82 + 5.57) / 2), new Quaternion(0.0f, 0.0f, 0.0f, 1.0f));
    } else if (areaId == 2) {
      tagPose = new Pose(new Point((10.3 + 11.55) / 2, (-9.25 + -8.5) / 2, 3.76203), new Quaternion(0.0f, 0.0f, 0.0f, 1.0f));
    } else if (areaId == 3) {
      tagPose = new Pose(new Point((10.3 + 11.55) / 2, (-8.4 + -7.45) / 2, 3.76203), new Quaternion(0.0f, 0.0f, 0.0f, 1.0f));
    } else if (areaId == 4) {
      tagPose = new Pose(new Point(9.866984, (-7.34 + -6.365) / 2, (4.32 + 5.57) / 2), new Quaternion(0.0f, 0.0f, 0.0f, 1.0f));
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
   * @param arResults List of detected ARTags.
   * @param undistortedImage The input image.
   * @return A map of markerId to ClippedImageResult.
   */
  public Map<Integer, ClippedImageResult> getClippedImages(List<ARTag> arResults, Mat undistortedImage) {
    Map<Integer, ClippedImageResult> clippedResultsMap = new HashMap<>();
    
    if (arResults.isEmpty()) {
      return clippedResultsMap;
    }

    for(ARTag arTag : arResults) {
      int markerId = arTag.getMarkerId();
      Mat corner = arTag.getCorner();

      ClippedImageResult clippedResult = clip(undistortedImage, corner);
      if (clippedResult.clippedImage.empty()) {
        Log.w(TAG, "Clipped image is empty for markerId: " + markerId);
        // Release transform matrices if they were created for an empty clipped image
        if (clippedResult.transformMatrix != null) clippedResult.transformMatrix.release();
        if (clippedResult.inverseTransformMatrix != null) clippedResult.inverseTransformMatrix.release();
        continue;
      }

      clippedResultsMap.put(markerId, clippedResult);
    }
    return clippedResultsMap;
  }

  /* ----------------------------- Tool Functions ----------------------------- */
  // Made public for access from VisionHandler if needed for coordinate transforms
  public Mat getCameraMatrix() {
    Mat camMatrix = new Mat(3, 3, CvType.CV_64F);
    double[] intrinsics = api.getNavCamIntrinsics()[0]; // Assuming this is [fx, fy, cx, cy, ...]
    camMatrix.put(0, 0, intrinsics[0]); // fx
    camMatrix.put(0, 1, 0);
    camMatrix.put(0, 2, intrinsics[2]); // cx
    camMatrix.put(1, 0, 0);
    camMatrix.put(1, 1, intrinsics[1]); // fy
    camMatrix.put(1, 2, intrinsics[3]); // cy
    camMatrix.put(2, 0, 0);
    camMatrix.put(2, 1, 0);
    camMatrix.put(2, 2, 1);
    return camMatrix;
  }

  // Made public for access from VisionHandler
  public Mat getDistortionCoefficients() {
    // Assuming api.getNavCamIntrinsics()[1] is an array of distortion coefficients [k1, k2, p1, p2, k3]
    double[] distCoeffsArray = api.getNavCamIntrinsics()[1];
    Mat distCoeffs = new Mat(1, distCoeffsArray.length, CvType.CV_64F);
    distCoeffs.put(0, 0, distCoeffsArray);
    return distCoeffs;
  }
  
  // Corrected Quaternion order for gov.nasa.arc.astrobee.types.Quaternion(x,y,z,w)
  private Quaternion rotationMatrixToQuaternion(Mat rotMat) {
    double[][] m = new double[3][3];
    for (int i = 0; i < 3; i++) {
      for (int j = 0; j < 3; j++) {
        m[i][j] = rotMat.get(i, j)[0];
      }
    }

    double trace = m[0][0] + m[1][1] + m[2][2];
    double w_scalar, x_vec, y_vec, z_vec;

    if (trace > 0) {
      double s = 0.5 / Math.sqrt(trace + 1.0);
      w_scalar = 0.25 / s;
      x_vec = (m[2][1] - m[1][2]) * s;
      y_vec = (m[0][2] - m[2][0]) * s;
      z_vec = (m[1][0] - m[0][1]) * s;
    } else {
      if (m[0][0] > m[1][1] && m[0][0] > m[2][2]) {
        double s = 2.0 * Math.sqrt(1.0 + m[0][0] - m[1][1] - m[2][2]);
        w_scalar = (m[2][1] - m[1][2]) / s;
        x_vec = 0.25 * s;
        y_vec = (m[0][1] + m[1][0]) / s;
        z_vec = (m[0][2] + m[2][0]) / s;
      } else if (m[1][1] > m[2][2]) {
        double s = 2.0 * Math.sqrt(1.0 + m[1][1] - m[0][0] - m[2][2]);
        w_scalar = (m[0][2] - m[2][0]) / s;
        x_vec = (m[0][1] + m[1][0]) / s;
        y_vec = 0.25 * s;
        z_vec = (m[1][2] + m[2][1]) / s;
      } else {
        double s = 2.0 * Math.sqrt(1.0 + m[2][2] - m[0][0] - m[1][1]);
        w_scalar = (m[1][0] - m[0][1]) / s;
        x_vec = (m[0][2] + m[2][0]) / s;
        y_vec = (m[1][2] + m[2][1]) / s;
        z_vec = 0.25 * s;
      }
    }
    
    // Normalize
    double norm = Math.sqrt(w_scalar * w_scalar + x_vec * x_vec + y_vec * y_vec + z_vec * z_vec);
    if (norm > 1e-9) {
      w_scalar /= norm;
      x_vec /= norm;
      y_vec /= norm;
      z_vec /= norm;
    } else {
      // Fallback to identity if norm is too small (e.g. zero rotation matrix)
      return new Quaternion(0f, 0f, 0f, 1f); 
    }
    // AstroBee's Quaternion is (x, y, z, w)
    return new Quaternion((float) x_vec, (float) y_vec, (float) z_vec, (float) w_scalar);
  }

  /**
   * Convert the pose in NavCam frame to World frame
   * Tag_Camera → Camera_Body → Body_World = Tag_World
   * 
   * @param tagInCVcam The tag pose in NavCam frame.
   * @return The pose in world frame. 
   */
  private Pose convertCameraToWorld(Pose tagInCVcam, Pose bodyInWorld) {
    // OpenCV camera pose in NavCam frame (Z forward, Y down, X right for OpenCV camera)
    // NavCam frame (X forward, Y left, Z up for Astrobee NavCam)
    // This transform rotates OpenCV's Z-forward to NavCam's X-forward, etc.
    // A common transform: (OpenCV X to NavCam Y, OpenCV Y to NavCam -Z, OpenCV Z to NavCam -X)
    // Or from NavCam to OpenCV: (NavCam X to OpenCV -Z, NavCam Y to OpenCV X, NavCam Z to OpenCV -Y)
    // The Quaternion (0.5, -0.5, 0.5, 0.5) corresponds to such a rotation (NavCam to OpenCV)
    // The Quaternion for CVcamInNavcam (OpenCV to NavCam) would be its inverse.
    // Inverse of (x,y,z,w) is (-x,-y,-z,w). So (-0.5, 0.5, -0.5, 0.5)
    // Let's re-verify the given quaternion: (0.5f, 0.5f, 0.5f, 0.5f) is not standard for this.
    // Standard OpenCV to ROS right-handed Z-up: (sqrt(0.5), -sqrt(0.5), 0, 0) then (0, sqrt(0.5), sqrt(0.5),0) etc.
    // The Kibo RPC documentation should specify this transform T_NavCam_CV<y_bin_456>.
    // Given: new Quaternion(0.5f, 0.5f, 0.5f, 0.5f) in the original code.
    // This is (x=0.5, y=0.5, z=0.5, w=0.5). Angle = 2*acos(0.5) = 120 deg. Axis = (1,1,1)/sqrt(3).
    // This corresponds to a rotation from (X Y Z) to (Y Z X).
    // If NavCam is (X_n, Y_n, Z_n) and CV Cam is (X_c, Y_c, Z_c)
    // This means X_c -> Y_n, Y_c -> Z_n, Z_c -> X_n. This is a common optical to body frame transform.
    Pose CVcamInNavcam = new Pose(
      new Point(0, 0, 0), // No translation
      new Quaternion(0.5f, 0.5f, 0.5f, 0.5f) // (X_cv, Y_cv, Z_cv) -> (Y_navcam, Z_navcam, X_navcam)
    );

    // Step 1: Convert tag pose from OpenCV camera frame to NavCam frame
    // tagInCVcam is (tag relative to CVcam_origin) expressed in CVcam_axes
    // We want (tag relative to NavCam_origin) expressed in NavCam_axes
    // This is T_NavCam_CVcam * P_tag_CVcam
    Pose tagInNavcam = composePoses(CVcamInNavcam, tagInCVcam);

    // NavCam pose in body frame (fixed offset and orientation)
    // This is T_Body_NavCam
    Pose navcamInBody = new Pose(
      new Point(0.1177, -0.0422, -0.0826), // NavCam origin in Body frame
      new Quaternion(0.0f, 0.0f, 0.0f, 1.0f)  // NavCam axes aligned with Body axes (identity rotation)
    );

    // Step 2: Convert tag pose from NavCam frame to body frame
    // This is T_Body_NavCam * P_tag_NavCam
    Pose tagInBody = composePoses(navcamInBody, tagInNavcam);

    // Step 3: Convert tag pose from body frame to world frame
    // bodyInWorld is T_World_Body
    // This is T_World_Body * P_tag_Body
    Pose tagInWorld = composePoses(bodyInWorld, tagInBody);

    return tagInWorld;
  }

  // Public static for use in other classes like VisionHandler
  public static Pose composePoses(Pose poseA, Pose poseB) {
    // poseA: Transform from frame C to frame A (T_AC)
    // poseB: Pose of point P in frame C (P_C)
    // Returns: Pose of point P in frame A (P_A)
    // P_A = T_AC * P_C
    Quaternion qa = poseA.getQuaternion(); // Orientation of C in A (q_AC)
    Point pa = poseA.getPoint();       // Origin of C in A (p_AC)

    Quaternion qb = poseB.getQuaternion(); // Orientation of P's frame in C (q_CP') (if P has its own frame) or P's orientation in C
    Point pb = poseB.getPoint();       // Position of P in C (p_CP)

    // 1. Rotate B's position vector (pb) by A's orientation (qa)
    //    pb_rotated_in_A = rotateByQuaternion(pb, qa)
    double[] rotated_pb_in_A = rotatePointByQuaternion(pb.getX(), pb.getY(), pb.getZ(), qa);
    
    // 2. Add A's position vector (pa)
    //    p_AP = p_AC + pb_rotated_in_A
    double final_x = pa.getX() + rotated_pb_in_A[0];
    double final_y = pa.getY() + rotated_pb_in_A[1];
    double final_z = pa.getZ() + rotated_pb_in_A[2];

    // 3. Combine orientations: q_AP' = q_AC * q_CP'
    Quaternion final_q = multiplyQuaternions(qa, qb);

    return new Pose(new Point(final_x, final_y, final_z), final_q);
  }

  // Public static for use
  public static double[] rotatePointByQuaternion(double x, double y, double z, Quaternion q) {
    // p' = q * p * q^-1
    // For a pure vector p=(0,x,y,z), this simplifies.
    // Using rotation matrix method for clarity here as it's common.
    float qx = q.getX();
    float qy = q.getY();
    float qz = q.getZ();
    float qw = q.getW();

    // Create rotation matrix from quaternion q
    // Ref: http://www.euclideanspace.com/maths/geometry/rotations/conversions/quaternionToMatrix/index.htm
    double r11 = 1 - 2*qy*qy - 2*qz*qz;
    double r12 = 2*qx*qy - 2*qz*qw;
    double r13 = 2*qx*qz + 2*qy*qw;
    double r21 = 2*qx*qy + 2*qz*qw;
    double r22 = 1 - 2*qx*qx - 2*qz*qz;
    double r23 = 2*qy*qz - 2*qx*qw;
    double r31 = 2*qx*qz - 2*qy*qw;
    double r32 = 2*qy*qz + 2*qx*qw;
    double r33 = 1 - 2*qx*qx - 2*qy*qy;

    double rx = r11 * x + r12 * y + r13 * z;
    double ry = r21 * x + r22 * y + r23 * z;
    double rz = r31 * x + r32 * y + r33 * z;

    return new double[]{rx, ry, rz};
  }

  // Public static for use
  public static Quaternion multiplyQuaternions(Quaternion q1, Quaternion q2) {
    // q_result = q1 * q2
    // q1 = (x1, y1, z1, w1), q2 = (x2, y2, z2, w2)
    float x1 = q1.getX(), y1 = q1.getY(), z1 = q1.getZ(), w1 = q1.getW();
    float x2 = q2.getX(), y2 = q2.getY(), z2 = q2.getZ(), w2 = q2.getW();

    // Hamilton product
    float rw = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2;
    float rx = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2;
    float ry = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2;
    float rz = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2;
    
    // AstroBee's Quaternion is (x, y, z, w)
    // The result here is (rx, ry, rz, rw)
    // No need to normalize here if inputs are unit quaternions, result will be unit.
    // However, to be safe from floating point errors, normalization is good.
    double norm = Math.sqrt(rw * rw + rx * rx + ry * ry + rz * rz);
    if (norm > 1e-9) {
        rw /= norm;
        rx /= norm;
        ry /= norm;
        rz /= norm;
    } else {
        return new Quaternion(0f, 0f, 0f, 1f); // Fallback
    }
    return new Quaternion(rx, ry, rz, rw);
  }


  /**
   * Clip the image to the area of interest.
   * @param image The input image.
   * @param corner The corners of the area of interest in the input image.
   * @return ClippedImageResult containing the clipped image and transformation matrix.
   */
  private ClippedImageResult clip(Mat image, Mat corner) {
    // `corner` contains the 4 corner points of the AR tag in the `image` (undistortedImage).
    // Each point is (x,y)
    // corner.get(0,0) -> top-left
    // corner.get(0,1) -> top-right
    // corner.get(0,2) -> bottom-right
    // corner.get(0,3) -> bottom-left

    // The h0,h1,v0,v1 parameters define a larger rectangle AROUND the AR tag.
    // These define the source quadrilateral in `image` that will be warped.
    org.opencv.core.Point[] srcQuadPoints = new org.opencv.core.Point[4];
    double[] c0 = corner.get(0, 0); // top-left of AR tag
    double[] c1 = corner.get(0, 1); // top-right of AR tag
    // double[] c2 = corner.get(0, 2); // bottom-right of AR tag (not directly used for h/v definition)
    double[] c3 = corner.get(0, 3); // bottom-left of AR tag

    // Vector from c0 to c1 (along top edge of AR tag)
    double dx_h = c1[0] - c0[0]; // delta x for horizontal-like edge
    double dy_h = c1[1] - c0[1]; // delta y for horizontal-like edge

    // Vector from c0 to c3 (along left edge of AR tag)
    double dx_v = c3[0] - c0[0]; // delta x for vertical-like edge
    double dy_v = c3[1] - c0[1]; // delta y for vertical-like edge

    // Parameters for extending the clipping box beyond the AR tag
    // h relates to horizontal-like direction, v to vertical-like
    double h0_factor = -4.4; // Extends "left" from c0 along c0-c1 direction
    double h1_factor = 0.0;  // Ends at c0 if v_factor is 0, or c3 if v_factor is 1, along c0-c1
                             // Correction: original h1 was for the "right" side of the clip area
                             // The original code implies h1_factor for points[1] and points[2]
                             // Let's rename to avoid confusion with AR tag corners.
                             // h_factor_p0 = -4.4 (for srcQuadPoints[0] and srcQuadPoints[3])
                             // h_factor_p1 = 0.0  (for srcQuadPoints[1] and srcQuadPoints[2])
                             // This is not right. Let's re-evaluate the original calculation.
                             // points[0] = new org.opencv.core.Point(h0 * (corner1[0] - corner0[0]) + v0 * (corner3[0] - corner0[0]) + corner0[0], h0 * (corner1[1] - corner0[1]) + v0 * (corner3[1] - corner0[1]) + corner0[1]);
                             // This means: src_top_left = c0 + h0*(c1-c0) + v0*(c3-c0)

    srcQuadPoints[0] = new org.opencv.core.Point(c0[0] + h0_factor * dx_h + (-0.5) * dx_v, c0[1] + h0_factor * dy_h + (-0.5) * dy_v); // Top-left of src quadrilateral
    srcQuadPoints[1] = new org.opencv.core.Point(c0[0] + (0.0) * dx_h + (-0.5) * dx_v, c0[1] + (0.0) * dy_h + (-0.5) * dy_v);   // Top-right
    srcQuadPoints[2] = new org.opencv.core.Point(c0[0] + (0.0) * dx_h + (3.0) * dx_v, c0[1] + (0.0) * dy_h + (3.0) * dy_v);    // Bottom-right
    srcQuadPoints[3] = new org.opencv.core.Point(c0[0] + h0_factor * dx_h + (3.0) * dx_v, c0[1] + h0_factor * dy_h + (3.0) * dy_v);   // Bottom-left


    // Calculate the width and height of the DESTINATION rectangle for the clipped image
    // This width/height will be the dimensions of the output `clippedImage` Mat
    // The original code calculates width/height from these srcQuadPoints, which might be skewed.
    // It should define a target rectangle size.
    // Let's make the destination rectangle have width and height based on the AR tag's markerLength, scaled.
    // For example, if we want the clipped image to represent an area 5x markerLength wide.
    // The original code calculates width/height based on the transformed points, this is fine.
    double warped_width = Math.sqrt(Math.pow(srcQuadPoints[0].x - srcQuadPoints[1].x, 2) + Math.pow(srcQuadPoints[0].y - srcQuadPoints[1].y, 2));
    double warped_height = Math.sqrt(Math.pow(srcQuadPoints[0].x - srcQuadPoints[3].x, 2) + Math.pow(srcQuadPoints[0].y - srcQuadPoints[3].y, 2));
    
    if (warped_width < 1 || warped_height < 1) {
        Log.w(TAG, "Calculated warped width or height is too small. Skipping clip.");
        return new ClippedImageResult(new Mat(), new Mat()); // Return empty if dimensions are invalid
    }

    MatOfPoint2f srcMatOfPoint2f = new MatOfPoint2f(srcQuadPoints);

    MatOfPoint2f dstMatOfPoint2f = new MatOfPoint2f(
        new org.opencv.core.Point(0, 0),                               // Top-left
        new org.opencv.core.Point(warped_width - 1, 0),                // Top-right
        new org.opencv.core.Point(warped_width - 1, warped_height - 1),// Bottom-right
        new org.opencv.core.Point(0, warped_height - 1)                // Bottom-left
    );

    Mat transformMatrix = Imgproc.getPerspectiveTransform(srcMatOfPoint2f, dstMatOfPoint2f);

    Mat clippedImageMat = Mat.zeros((int) warped_height, (int) warped_width, image.type());
    Imgproc.warpPerspective(image, clippedImageMat, transformMatrix, clippedImageMat.size());

    srcMatOfPoint2f.release();
    dstMatOfPoint2f.release();
    
    return new ClippedImageResult(clippedImageMat, transformMatrix);
  }
}
