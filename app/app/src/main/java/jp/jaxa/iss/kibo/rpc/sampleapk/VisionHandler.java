package jp.jaxa.iss.kibo.rpc.sampleapk;

import jp.jaxa.iss.kibo.rpc.api.KiboRpcApi;

import gov.nasa.arc.astrobee.types.Point;
import gov.nasa.arc.astrobee.types.Quaternion;
import gov.nasa.arc.astrobee.Result;

import java.util.List;
import java.util.ArrayList;

import android.util.Log;
import android.content.Context;

import org.opencv.calib3d.Calib3d;
import org.opencv.core.CvType;
import org.opencv.core.Mat;


/**
 * Class to handle the vision tasks of the robot and also communicate with navigator if needed.
 */
public class VisionHandler {
  private final String TAG = this.getClass().getSimpleName();
  private final CameraHandler cameraHandler;
  private final ItemDetector itemDetector;
  private final ARTagDetector arTagDetector;
  private final ItemManager itemManager;

  /**
   * Constructor
   * 
   * @param context Context reference
   * @param apiRef API reference
   */
  public VisionHandler(Context context, KiboRpcApi apiRef) {
    this.cameraHandler = new CameraHandler(apiRef);
    this.itemDetector = new ItemDetector(context, apiRef);
    this.arTagDetector = new ARTagDetector(apiRef);
    this.itemManager = new ItemManager(apiRef);

    Log.i(TAG, "Initialized");
  }

  public void inspectArea(int areaId) {

  // TODO :
  // raw image (Mat) ← api.getMatNavCam()
  //   ↓
  // undistorted image ← CameraHandler.getUndistortedImage()
  //   ↓
  // clipped image ← ImageUtil.clipAR(undistorted)
  //   ↓
  // bitmap image ← matToBitmap(clipped)
  //   ↓
  // item detection → boundingBoxes → drawBoxes(bitmap)

  }
}