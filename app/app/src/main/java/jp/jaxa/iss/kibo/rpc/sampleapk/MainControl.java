package jp.jaxa.iss.kibo.rpc.sampleapk;

import jp.jaxa.iss.kibo.rpc.api.KiboRpcApi;

import gov.nasa.arc.astrobee.types.Point;
import gov.nasa.arc.astrobee.types.Quaternion;

import org.opencv.core.Mat;

/**
 * @todo complete logic for exploring all areas, meeting astronauts, and finding treasures.
 */
public class MainControl {
    private KiboRpcApi api;

    public MainControl(KiboRpcApi api) {
        this.api = api;
    }

    public void method1() {
        api.startMission();
        exploreAllAreas();
        meetAstronaut();
        findAndCaptureTreasure();
    }

    private void exploreAllAreas() {
        
    }

    private void meetAstronaut() {
        
    }

    private void findAndCaptureTreasure() {
        
    }
}