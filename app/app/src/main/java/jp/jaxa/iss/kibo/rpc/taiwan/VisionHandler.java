package jp.jaxa.iss.kibo.rpc.taiwan;

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
import org.opencv.core.Core;
import org.opencv.core.Core.MinMaxLocResult;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

/**
 * Class to handle the vision tasks of the robot and interact with navigator class.
 *
 * @todo implement yolo model in VisionHandler::inspectArea() to return proper item.
 */
public class VisionHandler {
    private KiboRpcApi api;
    private final String TAG = this.getClass().getSimpleName();

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
        itemDetector = new ItemDetector(apiRef, context);
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
        // Get raw image
        Mat rawImage = cameraHandler.captureImage(areaId);

        // Get undistorted image
        Mat undistortedImage = cameraHandler.getUndistortedImage(rawImage);

        // Get tag pose and clipped image
        List<ARTagDetector.ARTag> arResults = arTagDetector.detect(undistortedImage);
        Map<Integer, Pose> tagPoses = arTagDetector.filterResult(arResults, currentPose);
        Map<Integer, Mat> clippedImages = arTagDetector.getclippedImages(arResults, undistortedImage);

        int markerId = areaId + 100;
        Pose tagPose = tagPoses.get(markerId);

        // Check clipped image
        Mat clippedImage = clippedImages.get(markerId);
        if (clippedImage == null) {
            Log.w(TAG, "No clipped image found.");
            return new ArrayList<>();
        }

        // Check overexposure
        // if (isOverexposed(clippedImage)) {
        //     Log.w(TAG, "Clipped image is overexposed.");
        //     return new ArrayList<>();
        // }

        api.saveMatImage(undistortedImage, String.format("area%d_undistorted.png", areaId));
        api.saveMatImage(clippedImage, String.format("area%d_clipped.png", areaId));

        // Detect item
        List<ItemDetector.Detection> results = itemDetector.detect(clippedImage);
        List<Item> itemList = itemDetector.filterResult(results, areaId, tagPose);
        Mat imageBbox = itemDetector.drawBoundingBoxes(clippedImage, results, areaId);
        api.saveMatImage(imageBbox, String.format("area%d_bbox.png", areaId));

        return itemList;
    }

    public Item recognizeTreasure() {
        // Get raw image
        Mat rawImage = cameraHandler.captureImage(0);

        // Get undistorted image
        Mat undistortedImage = cameraHandler.getUndistortedImage(rawImage);
        api.saveMatImage(rawImage, "treasure_undistorted.png");

        // Get tag pose and clipped image
        List<ARTagDetector.ARTag> arResults = arTagDetector.detect(undistortedImage);
        Map<Integer, Pose> tagPoses = arTagDetector.filterResult(arResults, currentPose);
        Map<Integer, Mat> clippedImages = arTagDetector.getclippedImages(arResults, undistortedImage);

        int areaId = 0;
        int markerId = 100;
        Pose tagPose = tagPoses.get(markerId);

        // Check clipped image
        Mat clippedImage = clippedImages.get(markerId);
        if (clippedImage == null) {
            Log.w(TAG, "No clipped image found.");
            return new Item();
        }
        api.saveMatImage(clippedImage, String.format("area%d_clipped.png", areaId));

        // Detect item
        List<ItemDetector.Detection> results = itemDetector.detect(clippedImage);
        List<Item> itemList = itemDetector.filterResult(results, areaId, tagPose);
        Mat imageBbox = itemDetector.drawBoundingBoxes(clippedImage, results, areaId);
        api.saveMatImage(imageBbox, String.format("area%d_bbox.png", areaId));

        // This array is expected to be [treasureItem, landmarkItem]
        Item treasureItem = itemList.get(0);

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

     /**
     * Analyzes the Mat (image) quality to detect overexposure.
     *
     * @param img The OpenCV Mat object (BGR or Gray).
     * @return true if the image is considered overexposed, false otherwise.
     */
    public boolean isOverexposed(Mat img) {
        Mat gray = new Mat();

        // Convert to grayscale if necessary
        if (img.channels() > 1) {
            Imgproc.cvtColor(img, gray, Imgproc.COLOR_BGR2GRAY);
        } else {
            img.copyTo(gray);
        }

        // Calculate mean brightness
        double meanBrightness = Core.mean(gray).val[0];

        // Find min and max values
        MinMaxLocResult mmr = Core.minMaxLoc(gray);
        double minVal = mmr.minVal;

        Log.i(TAG, "Image Brightness Analysis: Mean = " + meanBrightness + ", Min = " + minVal);

        boolean overexposed = false;

        // 1. Extreme overexposure (almost completely white)
        if (meanBrightness > 250) {
            Log.w(TAG, "Overexposed: Too bright overall (Mean > 250)");
            overexposed = true;
        }
        // 2. Loss of black details (darkest point is too bright)
        //    Ensure it's not just a small pattern issue by checking mean > 230
        else if (minVal > 80 && meanBrightness > 230) {
            Log.w(TAG, String.format("Overexposed: Black details lost (Min %.1f > 80 & Mean > 230)", minVal));
            overexposed = true;
        }

        gray.release(); // Release native memory
        return overexposed;
    }
}
