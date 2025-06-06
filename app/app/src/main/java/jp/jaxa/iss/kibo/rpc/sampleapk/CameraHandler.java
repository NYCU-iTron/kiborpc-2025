package jp.jaxa.iss.kibo.rpc.sampleapk;

import jp.jaxa.iss.kibo.rpc.api.KiboRpcApi;

import android.util.Log;

import org.opencv.calib3d.Calib3d;
import org.opencv.core.CvType;
import org.opencv.core.Mat;


/**
 * Class to handle the camera tasks of the robot.
 */
public class CameraHandler {
  private final KiboRpcApi api;
  private final String TAG = this.getClass().getSimpleName();
  private final Mat cameraMatrix;
  private final Mat distortionCoefficients;

  /**
   * Constructor
   * 
   * @param apiRef API reference.
   */
  public CameraHandler(KiboRpcApi apiRef) {
    this.api = apiRef;
    cameraMatrix = getCameraMatrix();
    distortionCoefficients = getDistortionCoefficients();
    
    Log.i(TAG, "Initialized");
  }

  public Mat captureImage(int areaId) {
    switch (areaId) {
      case 1:
        api.flashlightControlFront(0.30f);
        break;
      case 2:
        api.flashlightControlFront(0.3f);
        break;
      case 3:
        api.flashlightControlFront(0.3f);
        break;
      case 4:
        api.flashlightControlFront(0.35f);
        break;
      case 5:
        api.flashlightControlFront(0.3f);
        break;
      default:
        api.flashlightControlFront(0.01f);
        break;
    }

    // api.flashlightControlFront(0.35f);
    Mat image = api.getMatNavCam();
    api.flashlightControlFront(0.00f);
    return image;
  }

  public Mat getUndistortedImage(Mat rawImage) {
    Mat undistorted = new Mat();
    Calib3d.undistort(rawImage, undistorted, cameraMatrix, distortionCoefficients);
    return undistorted;
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
}