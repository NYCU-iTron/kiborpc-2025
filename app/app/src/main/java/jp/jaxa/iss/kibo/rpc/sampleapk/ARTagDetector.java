package jp.jaxa.iss.kibo.rpc.sampleapk;

import jp.jaxa.iss.kibo.rpc.api.KiboRpcApi;

import java.util.List;
import java.util.ArrayList;

import android.util.Log;

import org.opencv.core.Mat;
import org.opencv.aruco.Aruco;
import org.opencv.aruco.Dictionary;

public class ARTagDetector {
  private final KiboRpcApi api;
  private final String TAG = this.getClass().getSimpleName();

  public ARTagDetector(KiboRpcApi apiRef) {
    this.api = apiRef;
    Log.i(TAG, "Initialized");
  }

  // TODO : return the result of detection
  public void detectFromImage(Mat image) {
    Dictionary dictionary = Aruco.getPredefinedDictionary(Aruco.DICT_5X5_250);
    List<Mat> corners = new ArrayList<>();
    Mat markerIds = new Mat();
    Aruco.detectMarkers(image, dictionary, corners, markerIds);
  }
}