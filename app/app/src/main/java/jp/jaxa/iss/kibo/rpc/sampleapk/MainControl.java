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
        // Exploring area 1
        navigator.navigateToArea(1);
        visionHandler.getCurrentPose(navigator.getCurrentPose());
        Item[] area1Items = visionHandler.inspectArea(1);
        for (Item item : area1Items) {
            if (item.getItemId() / 10 == 1) { // Treasure Item
                itemManager.storeTreasureInfo(item);
                Log.i(TAG, "Area 1: Found treasure " + item.getItemName());
            } else if (item.getItemId() / 10 == 2) { // Landmark Item
                itemManager.setAreaInfo(item);
                Log.i(TAG, "Area 1: Found landmark " + item.getItemName());
            } else {
                Log.w(TAG, "Unknown item ID: " + item.getItemId());
            }
        }

        // Exploring area 2
        navigator.navigateToArea(2);
        visionHandler.getCurrentPose(navigator.getCurrentPose());
        Item[] area2Items = visionHandler.inspectArea(2);
        for (Item item : area2Items) {
            if (item.getItemId() / 10 == 1) { // Treasure Item
                itemManager.storeTreasureInfo(item);
                Log.i(TAG, "Area 2: Found treasure " + item.getItemName());
            } else if (item.getItemId() / 10 == 2) { // Landmark Item
                itemManager.setAreaInfo(item);
                Log.i(TAG, "Area 2: Found landmark " + item.getItemName());
            } else {
                Log.w(TAG, "Unknown item ID: " + item.getItemId());
            }
        }

        // Exploring area 3
        navigator.navigateToArea(3);
        visionHandler.getCurrentPose(navigator.getCurrentPose());
        Item[] area3Items = visionHandler.inspectArea(3);
        for (Item item : area3Items) {
            if (item.getItemId() / 10 == 1) { // Treasure Item
                itemManager.storeTreasureInfo(item);
                Log.i(TAG, "Area 3: Found treasure " + item.getItemName());
            } else if (item.getItemId() / 10 == 2) { // Landmark Item
                itemManager.setAreaInfo(item);
                Log.i(TAG, "Area 3: Found landmark " + item.getItemName());
            } else {
                Log.w(TAG, "Unknown item ID: " + item.getItemId());
            }
        }

        // Exploring area 4
        navigator.navigateToArea(4);
        visionHandler.getCurrentPose(navigator.getCurrentPose());
        Item[] area4Items = visionHandler.inspectArea(4);
        for (Item item : area4Items) {
            if (item.getItemId() / 10 == 1) { // Treasure Item
                itemManager.storeTreasureInfo(item);
                Log.i(TAG, "Area 4: Found treasure " + item.getItemName());
            } else if (item.getItemId() / 10 == 2) { // Landmark Item
                itemManager.setAreaInfo(item);
                Log.i(TAG, "Area 4: Found landmark " + item.getItemName());
            } else {
                Log.w(TAG, "Unknown item ID: " + item.getItemId());
            }
        }
    }

    /**
     * @brief Second part of the mission to meet astronauts.
     */
    private Item meetAstronaut() {
        // See the real treasure
        navigator.navigateToArea(5);
        this.api.reportRoundingCompletion();
        // Recognize the treasure
        Item treasureItem = visionHandler.recognizeTreasure();
        Log.i(TAG, "Treasure recognized: " + treasureItem.getItemName());
        // Find the area of the treasure and the treasure item
        treasureItem = itemManager.getTreasureInfo(treasureItem);
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
}