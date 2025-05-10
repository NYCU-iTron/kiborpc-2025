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
  public List<Item> inspectArea(int areaId) {
    Mat rawImage = cameraHandler.captureImage();
    if (DEBUG) api.saveMatImage(rawImage, String.format("area%d_raw.png", areaId));
    Log.i(TAG, "Get raw image");

    // Get undistorted image
    Mat undistortedImage = cameraHandler.getUndistortedImage(rawImage);
    if (DEBUG) api.saveMatImage(undistortedImage, String.format("area%d_undistorted.png", areaId));
    Log.i(TAG, "Get undistorted image.");

    // Get tag pose and clipped image
    List<ARTag> arResults = arTagDetector.detect(undistortedImage);
    Map<Integer, Pose> tagPoses = arTagDetector.filterResult(arResults, currentPose);
    Map<Integer, Mat> clippedImages = arTagDetector.getclippedImages(arResults, undistortedImage);
    
    int markerId = areaId + 100;
    Pose tagPose = tagPoses.get(markerId);
    Mat clippedImage = clippedImages.get(markerId);

    List<float[]> itemResults = new ArrayList<>();
    List<Item> itemList = new ArrayList<>();
    
    if (clippedImage == null) {
      Log.w(TAG, "No clipped image found.");
      return itemList;
    }

    if (DEBUG) api.saveMatImage(clippedImage, String.format("area%d_clipped.png", areaId));
    Log.i(TAG, "Get clipped image.");

    // Detect item
    itemResults = itemDetector.detect(clippedImage, ItemDetector.ModelType.CLIPPED);
    itemList = itemDetector.filterResult(itemResults, areaId, tagPose);
    if (DEBUG) itemDetector.drawBoundingBoxes(clippedImage, itemResults, areaId);

    return itemList;
  }

  public Item recognizeTreasure() {
    int areaId = 0;

    // Get raw image
    Mat rawImage = cameraHandler.captureImage();
    if (DEBUG) api.saveMatImage(rawImage, "treasure_raw.png");
    
    // Get undistorted image
    Mat undistortedImage = cameraHandler.getUndistortedImage(rawImage);
    if (DEBUG) api.saveMatImage(rawImage, "treasure_undistorted.png");

    // Detect item
    List<float[]> itemResults = itemDetector.detect(undistortedImage, ItemDetector.ModelType.ENV);
    List<Item> itemList = itemDetector.filterResult(itemResults, areaId, new Pose());
    if (DEBUG) itemDetector.drawBoundingBoxes(undistortedImage, itemResults, areaId);

    Item treasureItem = itemList.get(0); // This array is expected to be [treasureItem, landmarkItem]

    return treasureItem;
  }

  public void captureTreasureImage() {
    this.api.takeTargetItemSnapshot();
  }

  public List<Item> guessResult(int areaId) {
    Pose tagPose = arTagDetector.guessResult(areaId);
    List<Item> guessItemArray = itemDetector.guessResult(areaId, tagPose);
    return guessItemArray;
  }
}
