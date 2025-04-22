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

    public MainControl(Context context, KiboRpcApi api) {
        this.context = context;
        this.api = api;
        this.navigator = new Navigator(api);
        this.visionHandler = new VisionHandler(context, api);
    }

    /**
     * @brief Main method to control the mission.
     * 
     * This method is called when the mission starts.
     * It will call other methods to explore all areas, meet astronauts, and find treasures.
     * 
     * @note This method is intended to be called once with no following code.
     */
    public void method1() {
        api.startMission();
        exploreAllAreas();
        int area = meetAstronaut();
        findAndCaptureTreasure(area);
    }

    /**
     * @brief First part of the mission to explore all areas.
     */
    private void exploreAllAreas() {
        // Exploring area 1
        navigator.navigateToArea1();
        visionHandler.getCurrentPose(navigator.getCurrentPose());
        visionHandler.inspectArea();

        // Exploring area 2
        navigator.navigateToArea2();
        visionHandler.getCurrentPose(navigator.getCurrentPose());
        visionHandler.inspectArea();

        // Exploring area 3
        navigator.navigateToArea3();
        visionHandler.getCurrentPose(navigator.getCurrentPose());
        visionHandler.inspectArea();

        // Exploring area 4
        navigator.navigateToArea4();
        visionHandler.getCurrentPose(navigator.getCurrentPose());
        visionHandler.inspectArea();
    }

    /**
     * @brief Second part of the mission to meet astronauts.
     */
    private int meetAstronaut() {
        // See the real treasure
        navigator.navigateToReport();
        // // Recognize the treasure
        // Item item = visionHandler.recognizeTreasure();
        // // Compare the treasure with other items
        int treasureArea = 1; // some number
        return treasureArea;
    }

    /**
     * @brief Third part of the mission to find and capture treasure.
     */
    private void findAndCaptureTreasure(int treasureArea) {
        // // maybe a more optimal pose can be used to take the picture of the treasure?
        // switch (treasureArea) {
        //     case 1:
        //         navigator.navigateToTreasureArea1();
        //         break;
        //     case 2:
        //         navigator.navigateToTreasureArea2();
        //         break;
        //     case 3:
        //         navigator.navigateToTreasureArea3();
        //         break;
        //     case 4:
        //         navigator.navigateToTreasureArea4();
        //         break;
        //     default:
        //         // Handle error
        //         Log.w(TAG, "No treasure found in any area.");
        //         break;
        // }
        // // Capture the treasure image
        // visionHandler.captureTreasureImage();
    }
}