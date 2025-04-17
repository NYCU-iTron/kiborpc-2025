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

public class ARTagDetector {
  private final KiboRpcApi api;
  private final String TAG = this.getClass().getSimpleName();
  private final Dictionary arucoDictionary;
  private final float markerLength;
  private final Mat cameraMatrix;
  private final Mat distortionCoefficients;

  public ARTagDetector(KiboRpcApi apiRef) {
    this.api = apiRef;
    markerLength = 0.05f;
    cameraMatrix = getCameraMatrix();
    distortionCoefficients = getDistortionCoefficients();
    arucoDictionary = Aruco.getPredefinedDictionary(Aruco.DICT_5X5_250);

    Log.i(TAG, "Initialized");
  }

  public Map detectFromImage(Mat undistortedImage) {
    List<Mat> corners = new ArrayList<>();
    Mat markerIds = new Mat();
    Aruco.detectMarkers(undistortedImage, arucoDictionary, corners, markerIds);
  
    Map<Integer, Pose> result = new HashMap<>();

    if (markerIds.empty()) {
      Log.w(TAG, "No AR Tag Found");
      return result;
    }

    Mat rvec = new Mat();
    Mat tvec = new Mat();

    for (int i = 0; i < markerIds.rows(); i++) {
      // Get marker ID
      int id = (int) markerIds.get(i, 0)[0];

      // Get corner 
      Mat corner = corners.get(i);

      // Estimate pose
      Aruco.estimatePoseSingleMarkers(Collections.singletonList(corner), markerLength, cameraMatrix, distortionCoefficients, rvec, tvec);

      // Extract shift matrix
      double[] t = tvec.get(0, 0);
      Point position = new Point(t[0], t[1], t[2]);

      // Extract rotation matrix
      double[] r = rvec.get(0, 0);
      Mat rotMat = new Mat();
      Calib3d.Rodrigues(new MatOfDouble(r), rotMat);

      Quaternion orientation = rotationMatrixToQuaternion(rotMat);

      result.put(id, new Pose(position, orientation));
    }

    return result;
  }

  public Mat getCameraMatrix() {
    Mat cameraMatrix = new Mat(3, 3, CvType.CV_64F);
    cameraMatrix.put(0, 0, api.getNavCamIntrinsics()[0]);
    return cameraMatrix;
  }

  public Mat getDistortionCoefficients() {
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

    return new Quaternion((float)w, (float)x, (float)y, (float)z);
  }
}