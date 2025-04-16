package jp.jaxa.iss.kibo.rpc.sampleapk;

import jp.jaxa.iss.kibo.rpc.api.KiboRpcApi;

import android.util.Log;

import org.opencv.calib3d.Calib3d;
import org.opencv.core.CvType;
import org.opencv.core.Mat;


public class CameraHandler {
  private final KiboRpcApi api;
  private final String TAG = this.getClass().getSimpleName();

  public CameraHandler(KiboRpcApi apiRef) {
    this.api = apiRef;
    Log.i(TAG, "Initialized");
  }

  public Mat captureImage() {
    return api.getMatNavCam();
  }

  public Mat getUndistortedImage(Mat rawImage) {
    Mat undistorted = new Mat();
    Calib3d.undistort(rawImage, undistorted, getCameraMatrix(), getDistortionCoefficients());
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

  public void saveImage(Mat image, String filename) {
    api.saveMatImage(image, filename);
  }
}