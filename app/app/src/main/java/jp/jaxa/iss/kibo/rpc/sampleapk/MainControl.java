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
        navigator.navigateToArea1();
        visionHandler.getCurrentPose(navigator.getCurrentPose());
        Item[] area1Items = visionHandler.inspectArea(1);
        for (Item item : area1Items) {
            if (item.getItemId() / 10 == 1) { // Treasure Item
                itemManager.storeTreasureInfo(item);
            } else if (item.getItemId() / 10 == 2) { // Landmark Item
                itemManager.setAreaInfo(item);
            } else {
                Log.w(TAG, "Unknown item ID: " + item.getItemId());
            }
        }

        // Exploring area 2
        navigator.navigateToArea2();
        visionHandler.getCurrentPose(navigator.getCurrentPose());
        Item[] area2Items = visionHandler.inspectArea(2);
        for (Item item : area2Items) {
            if (item.getItemId() / 10 == 1) { // Treasure Item
                itemManager.storeTreasureInfo(item);
            } else if (item.getItemId() / 10 == 2) { // Landmark Item
                itemManager.setAreaInfo(item);
            } else {
                Log.w(TAG, "Unknown item ID: " + item.getItemId());
            }
        }

        // Exploring area 3
        navigator.navigateToArea3();
        visionHandler.getCurrentPose(navigator.getCurrentPose());
        Item[] area3Items = visionHandler.inspectArea(3);
        for (Item item : area3Items) {
            if (item.getItemId() / 10 == 1) { // Treasure Item
                itemManager.storeTreasureInfo(item);
            } else if (item.getItemId() / 10 == 2) { // Landmark Item
                itemManager.setAreaInfo(item);
            } else {
                Log.w(TAG, "Unknown item ID: " + item.getItemId());
            }
        }

        // Exploring area 4
        navigator.navigateToArea4();
        visionHandler.getCurrentPose(navigator.getCurrentPose());
        Item[] area4Items = visionHandler.inspectArea(4);
        for (Item item : area4Items) {
            if (item.getItemId() / 10 == 1) { // Treasure Item
                itemManager.storeTreasureInfo(item);
            } else if (item.getItemId() / 10 == 2) { // Landmark Item
                itemManager.setAreaInfo(item);
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
        navigator.navigateToReport();
        this.api.reportRoundingCompletion();
        // Recognize the treasure
        Item treasureItem = visionHandler.recognizeTreasure();
        return treasureItem;
    }

    /**
     * @brief Third part of the mission to find and capture treasure.
     */
    private void findAndCaptureTreasure(Item treasureItem) {
        // maybe a more optimal pose can be used to take the picture of the treasure?
        Item treasureInfo = itemManager.getTreasureInfo(treasureItem);
        int treasureArea = treasureInfo.getAreaId();
        switch (treasureArea) {
            case 1:
                navigator.navigateToArea1();
                break;
            case 2:
                navigator.navigateToArea2();
                break;
            case 3:
                navigator.navigateToArea3();
                break;
            case 4:
                navigator.navigateToArea4();
                break;
            default:
                // Handle error
                Log.w(TAG, "No treasure found in any area.");
                // guessing the treasure is in area 1
                navigator.navigateToArea1();
                break;
        }
        // Capture the treasure image
        visionHandler.captureTreasureImage();
    }
}