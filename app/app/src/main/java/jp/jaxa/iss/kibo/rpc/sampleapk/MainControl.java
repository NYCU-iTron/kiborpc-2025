package jp.jaxa.iss.kibo.rpc.sampleapk;

import jp.jaxa.iss.kibo.rpc.api.KiboRpcApi;

import gov.nasa.arc.astrobee.types.Point;
import gov.nasa.arc.astrobee.types.Quaternion;

import java.util.List;

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

        // Explore all areas
        for(int areaId = 1; areaId <= 4; areaId++) {
            handleSingleArea(areaId);
        }

        Item treasureItem = meetAstronaut();
        findAndCaptureTreasure(treasureItem);
    }

    public void method2() {
        api.startMission();

        // Explore all areas
        handleSingleArea(1);
        handleCombinedArea();
        handleSingleArea(4);

        Item treasureItem = meetAstronaut();
        findAndCaptureTreasure(treasureItem);
    }

    private void handleSingleArea(int areaId) {
        navigator.navigateToArea(areaId);
        visionHandler.getCurrentPose(navigator.getCurrentPose());
        List<Item> areaItems = null;

        int retryMax = 5;
        for (int retry = 1; retry <= retryMax; retry++) {
            Log.i(TAG, "Exploring area " + areaId + " (try " + retry + ")");
            areaItems = visionHandler.inspectArea(areaId);

            if (containsLandmark(areaItems)) break;

            Log.w(TAG, "No landmark found, pause system for a short time then retry.");
            try {
                Thread.sleep(200);
            } catch (Exception e) {
                Log.w(TAG, "Fail to sleep thread" + e);
            }
        }

        if (areaItems == null || !containsLandmark(areaItems)) {
            Log.w(TAG, "No landmark found after retries, leaving to fate.");
            areaItems = visionHandler.guessResult(areaId);
        }

        // Treasure Item
        Item treasureItem = areaItems.get(0); 
        if (treasureItem.getItemId() / 10 == 1) {
            itemManager.storeTreasureInfo(treasureItem);
            Log.i(TAG, "Area " + areaId + ": Found treasure " + treasureItem.getItemName());
        } else {
            Log.w(TAG, "Area " + areaId + ": No treasure.");
        }
        
        // Landmark Item
        Item landmarkItem = areaItems.get(1);
        if (landmarkItem.getItemId() / 10 == 2) {
            itemManager.setAreaInfo(landmarkItem);
            Log.i(TAG, "Area " + areaId + ": Found landmark " + landmarkItem.getItemName());
        } else {
            // Should never end up here
            Log.w(TAG, "Area " + areaId + ": No landmark.");
        }
    }

    private void handleCombinedArea() {
        navigator.navigateToArea(5);
        visionHandler.getCurrentPose(navigator.getCurrentPose());
        List<Item> area2Items = null;
        List<Item> area3Items = null;
        
        boolean success2 = false;
        boolean success3 = false;

        int retryMax = 5;
        for (int retry = 1; retry <= retryMax; retry++) {
            Log.i(TAG, "Exploring area 2 and 3 together (try " + retry + ")");
            
            if (success2 == true && success3 == true) break;

            if(success2 == false)
                area2Items = visionHandler.inspectArea(2);
            if(success3 == false)
                area3Items = visionHandler.inspectArea(3);
            
            // Check area 2 items
            if (success2 == false && containsLandmark(area2Items)) {
                // Treasure Item
                Item treasureItem = area2Items.get(0);
                if (treasureItem.getItemId() / 10 == 1) {
                    itemManager.storeTreasureInfo(treasureItem);
                    Log.i(TAG, "Area 2: Found treasure " + treasureItem.getItemName());
                } else {
                    Log.w(TAG, "Area 2: No treasure.");
                }
                
                // Landmark Item
                Item landmarkItem = area2Items.get(1);
                if (landmarkItem.getItemId() / 10 == 2) {
                    itemManager.setAreaInfo(landmarkItem);
                    Log.i(TAG, "Area 2: Found landmark " + landmarkItem.getItemName());
                } else {
                    Log.w(TAG, "Area 2: No landmark.");
                }

                success2 = true;
            }

            // Check area 3 items
            if (success3 == false && containsLandmark(area3Items)) {
                // Treasure Item
                Item treasureItem = area3Items.get(0); 
                if (treasureItem.getItemId() / 10 == 1) {
                    itemManager.storeTreasureInfo(treasureItem);
                    Log.i(TAG, "Area 3: Found treasure " + treasureItem.getItemName());
                } else {
                    Log.w(TAG, "Area 3: No treasure.");
                }
                
                // Landmark Item
                Item landmarkItem = area3Items.get(1);
                if (landmarkItem.getItemId() / 10 == 2) {
                    itemManager.setAreaInfo(landmarkItem);
                    Log.i(TAG, "Area 3: Found landmark " + landmarkItem.getItemName());
                } else {
                    // Should never end up here
                    Log.w(TAG, "Area 3: No landmark.");
                }

                success3 = true;               
            }

            Log.w(TAG, "Exploration not completed, pause system for a short time then retry.");

            try {
                Thread.sleep(200);
            } catch (Exception e) {
                Log.w(TAG, "Fail to sleep thread" + e);
            }
        }

        if(success2 == false) {
            handleSingleArea(2);
        }
        if(success3 == false) {
            handleSingleArea(3);
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
        List<Item> areaItems = null;
        Item treasureItem = null;

        int retryMax = 20;
        for (int retry = 1; retry <= retryMax; retry++) {
            areaItems = visionHandler.inspectArea(areaId);
            if (containsTreasure(areaItems)) {
                treasureItem = areaItems.get(0);
                break;
            } else {
                Log.w(TAG, "No treasure found, pause system for a short time then retry.");
                    
                try {
                    Thread.sleep(500);
                } catch (InterruptedException e) {
                    Log.w(TAG, "Fail to sleep thread" + e);
                }

                areaItems = visionHandler.inspectArea(areaId);
            }
            
        }

        if (treasureItem == null) {
            Log.w(TAG, "No treasure found, leaving to fate.");
            areaItems = visionHandler.guessResult(areaId);
            treasureItem = areaItems.get(0);
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

    private boolean containsLandmark(List<Item> items) {
        for (Item item : items) {
            if (item.getItemId() / 10 == 2) return true;
        }
        return false;
    }

    private boolean containsTreasure(List<Item> items) {
        for (Item item : items) {
            if (item.getItemId() / 10 == 1) return true;
        }
        return false;
    }
}