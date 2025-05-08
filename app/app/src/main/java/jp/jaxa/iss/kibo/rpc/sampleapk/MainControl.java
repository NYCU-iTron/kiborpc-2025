package jp.jaxa.iss.kibo.rpc.sampleapk;

import jp.jaxa.iss.kibo.rpc.api.KiboRpcApi;

import gov.nasa.arc.astrobee.types.Point;
import gov.nasa.arc.astrobee.types.Quaternion;

import org.opencv.core.Mat;

import android.util.Log;
import android.content.Context;

/**
 * @brief MainControl class for planning the mission.
 * 
 * @param context Context reference.
 * @param api API reference.
 * 
 * Example of using the MainControl class:
 * @code
 * MainControl mainControl = new MainControl(getApplicationContext(), api);
 * mainControl.method1();
 * @endcode
 */
public class MainControl {
    private final String TAG = this.getClass().getSimpleName();

    private Context context;
    private KiboRpcApi api;
    private Navigator navigator;
    private VisionHandler visionHandler;
    private ItemManager itemManager;

    public MainControl(Context context, KiboRpcApi api) {
        this.context = context;
        this.api = api;
        this.navigator = new Navigator(api);
        this.visionHandler = new VisionHandler(context, api);
        this.itemManager = new ItemManager(api);
    }

    /**
     * @brief Main method to control the mission.
     * 
     * This method is called when the mission starts.
     * It will call other methods to explore all areas, meet astronauts, and find treasures.
     * 
     * @note This method is intended to be called once with no following code.
     * 
     * @todo maybe a more optimal pose can be used to take the picture of the treasure?
     */
    public void method1() {
        api.startMission();
        exploreAllAreas();
        Item treasureItem = meetAstronaut();
        findAndCaptureTreasure(treasureItem);
    }

    /**
     * @brief First part of the mission to explore all areas.
     */
    private void exploreAllAreas() {
        for(int areaId = 1; areaId <= 4; areaId++) {
            navigator.navigateToArea(areaId);
            visionHandler.getCurrentPose(navigator.getCurrentPose());
            Item[] areaItems = visionHandler.inspectArea(areaId);

            int retryMax = 5;
            for (int retry = 1; retry <= retryMax - 1; retry++) {
                Log.i(TAG, "Exploring area " + areaId + " (try " + retry + ")");
                
                if (containsLandmark(areaItems)) {
                    // Treasure Item
                    if (areaItems[0].getItemId() / 10 == 1) {
                        itemManager.storeTreasureInfo(areaItems[0]);
                        Log.i(TAG, "Area " + areaId + ": Found treasure " + areaItems[0].getItemName());
                    } else {
                        Log.w(TAG, "Area " + areaId + ": Treasure not found.");
                    }
                    
                    // Landmark Item
                    if (areaItems[1].getItemId() / 10 == 2) {
                        itemManager.setAreaInfo(areaItems[1]);
                        Log.i(TAG, "Area " + areaId + ": Found landmark " + areaItems[1].getItemName());
                    } else {
                        Log.w(TAG, "Area " + areaId + ": Landmark not found."); // Should never happen
                    }

                    break;
                } else {
                    if (retry == retryMax - 1) {
                        Log.w(TAG, "No landmark found, leaving to fate.");
                        areaItems = visionHandler.guessResult(areaId);
                    } else {
                        Log.w(TAG, "No landmark found, pause system for a short time then retry.");
                        
                        try {
                            Thread.sleep(200);
                        } catch (InterruptedException e) {
                            Log.w(TAG, "Fail to sleep thread" + e);
                        }

                        areaItems = visionHandler.inspectArea(areaId);
                    }
                }
            }
        }
    }

    /**
     * @brief Second part of the mission to meet astronauts.
     */
    private Item meetAstronaut() {
        int areaId = 0;

        // See the real treasure
        navigator.navigateToArea(areaId);
        api.reportRoundingCompletion();
        
        // Recognize the treasure
        Item[] areaItems = visionHandler.inspectArea(areaId);
        Item treasureItem = null;

        int retryMax = 20;
        for (int retry = 1; retry <= retryMax; retry++) {
            if (containsTreasure(areaItems)) {
                treasureItem = areaItems[0];                
                break;
            } else {
                if (retry == retryMax - 1) {
                    Log.w(TAG, "No treasure found, leaving to fate.");
                    areaItems = visionHandler.guessResult(areaId);
                } else {
                    Log.w(TAG, "No treasure found, pause system for a short time then retry.");
                        
                    try {
                        Thread.sleep(200);
                    } catch (InterruptedException e) {
                        Log.w(TAG, "Fail to sleep thread" + e);
                    }

                    areaItems = visionHandler.inspectArea(areaId);
                }
            }
        }

        Log.i(TAG, "Treasure recognized: " + treasureItem.getItemName());
        api.notifyRecognitionItem();

        // Find the area of the treasure and the treasure item
        treasureItem = itemManager.getTreasureInfo(treasureItem);
        Log.i(TAG, "Treasure area: " + treasureItem.getAreaId());

        return treasureItem;
    }

    /**
     * @brief Third part of the mission to find and capture treasure.
     */
    private void findAndCaptureTreasure(Item treasureItem) {
        navigator.navigateToTreasure(treasureItem);
        Log.i(TAG, "Navigating to treasure " + treasureItem.getItemName() + " at " + treasureItem.getAreaId());
        // Capture the treasure image
        visionHandler.captureTreasureImage();
    }

    private boolean containsLandmark(Item[] items) {
        for (Item item : items) {
            if (item.getItemId() / 10 == 2) return true;
        }
        return false;
    }

    private boolean containsTreasure(Item[] items) {
        for (Item item : items) {
            if (item.getItemId() / 10 == 1) return true;
        }
        return false;
    }
}