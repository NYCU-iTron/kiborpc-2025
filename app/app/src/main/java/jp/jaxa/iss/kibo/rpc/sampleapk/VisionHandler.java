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
  private KiboRpcApi api;
  private final String TAG = this.getClass().getSimpleName();
  private boolean DEBUG = true;

  private final CameraHandler cameraHandler;
  private final ItemDetector itemDetector;
  private final ARTagDetector arTagDetector;
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
  public Item[] inspectArea(int area) {
    Mat rawImage = cameraHandler.captureImage();
    if (DEBUG) api.saveMatImage(rawImage, String.format("area%d_raw.png", area));

    Mat undistortedImage = cameraHandler.getUndistortedImage(rawImage);
    if (DEBUG) api.saveMatImage(undistortedImage, String.format("area%d_undistorted.png", area));

    // Detect AR tag pose
    Map<Integer, Pose> arResult = arTagDetector.detect(undistortedImage);
    Pose tagPose = arTagDetector.filterResult(arResult, area, currentPose);

    // Detect item
    List<float[]> detectResult = itemDetector.detect(undistortedImage);
    Item[] detectedItemArray = itemDetector.filterResult(detectResult, area, tagPose);
    if (DEBUG) itemDetector.drawBoundingBoxes(undistortedImage, detectResult, area);

    return detectedItemArray;
  }

  public Item recognizeTreasure() {
    Mat rawImage = cameraHandler.captureImage();
    if (DEBUG) api.saveMatImage(rawImage, "treasure_raw.png");

    Mat undistortedImage = cameraHandler.getUndistortedImage(rawImage);
    if (DEBUG) api.saveMatImage(rawImage, "treasure_undistorted.png");

    List<float[]> detectResult = itemDetector.detect(undistortedImage);
    Item[] detectedItemArray = itemDetector.filterResult(detectResult, 5, new Pose());
    if (DEBUG) itemDetector.drawBoundingBoxes(undistortedImage, detectResult, 5);

    Item detectedItem = detectedItemArray[0]; // This array is expected to be [treasureItem, landmarkItem]
    this.api.notifyRecognitionItem();

    return detectedItem;
  }

  public void captureTreasureImage() {
    this.api.takeTargetItemSnapshot();
  }
}