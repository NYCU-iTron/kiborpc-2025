package jp.jaxa.iss.kibo.rpc.sampleapk;

import jp.jaxa.iss.kibo.rpc.api.KiboRpcApi;

import gov.nasa.arc.astrobee.types.Point;
import gov.nasa.arc.astrobee.types.Quaternion;
import gov.nasa.arc.astrobee.Result;

import java.util.List;
import java.util.Map;
import java.util.ArrayList;

import android.util.Log;
import android.content.Context;

import org.opencv.calib3d.Calib3d;
import org.opencv.core.CvType;
import org.opencv.core.Mat;


/**
 * Class to handle the vision tasks of the robot and interact with navigator class.
 * 
 * @todo implement yolo model in VisionHandler::inspectArea() to return proper item.
 */
public class VisionHandler {
  private final String TAG = this.getClass().getSimpleName();
  private final CameraHandler cameraHandler;
  private final ItemDetector itemDetector;
  private final ARTagDetector arTagDetector;
  private KiboRpcApi api;

  private Pose currentPose = null;

  /**
   * Constructor
   * 
   * @param context Context reference.
   * @param apiRef API reference.
   * 
   * Example of using the VisionHandler constructor:
   * @code
   * VisionHandler visionHandler = new VisionHandler(getApplicationContext(), api);
   * @endcode
   */
  public VisionHandler(Context context, KiboRpcApi apiRef) {
    cameraHandler = new CameraHandler(apiRef);
    itemDetector = new ItemDetector(context, apiRef);
    arTagDetector = new ARTagDetector(apiRef);
    api = apiRef;

    Log.i(TAG, "Initialized");
  }

  /**
   * Get the current pose (should be measured by navigator class).
   * 
   * @param pose Current pose.
   */
  public void getCurrentPose(Pose pose) {
    currentPose = pose;
  }

  /**
   * Capture and analyze the image from NavCam after arriving target pose of the area.
   * 
   * @note NOTE : You should call getCurrentPose() to update the currentPose before using this function.
   * 
   * Example:
   * @code
   * Navigator navigator = new Navigator(api);
   * VisionHandler visionHandler = new VisionHandler(getApplicationContext(), api);
   * 
   * // Remember to call getCurrentPose() before using inspectArea()
   * visionHandler.getCurrentPose(navigator.getCurrentPose());
   * visionHandler.inspectArea();
   * @endcode
   */
  public Item inspectArea(int area) {
    Mat rawImage = cameraHandler.captureImage();
    Mat undistortedImage = cameraHandler.getUndistortedImage(rawImage);
    Map<Integer, Pose> arResult = arTagDetector.detectFromImage(undistortedImage);

    Log.i(TAG, "Current Body Pose in World: " + currentPose.toString());
    for (Map.Entry<Integer, Pose> entry : arResult.entrySet()) {
      Integer id = entry.getKey();
      Pose pose = entry.getValue();
      Pose poseWorld = arTagDetector.convertCameraToWorld(pose, currentPose);
      Log.i(TAG, "ID: " + id + ", Pose in Cam: " + pose.toString());
      Log.i(TAG, "ID: " + id + ", Pose in poseWorld: " + poseWorld.toString());
    }

    /**
     * @todo implement yolo model to detect the item in the image.
     */
    Item detectedItem = new Item(area, 11, "crystal", 1, new Pose());
    return detectedItem;
  }

  public Item recognizeTreasure() {
    this.api.notifyRecognitionItem();
    
    /**
     * @todo implement yolo model to detect the item in the treasure image.
     */
    Item detectedItem = new Item();
    return detectedItem;
  }

  public void captureTreasureImage() {
    this.api.takeTargetItemSnapshot();
  }
}