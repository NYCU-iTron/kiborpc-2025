package jp.jaxa.iss.kibo.rpc.sampleapk;

import jp.jaxa.iss.kibo.rpc.api.KiboRpcApi;

import android.content.Context;
import android.util.Log;

/**
 * Class to detect the item from provided image using yolo model
 */
public class ItemDetector {
  private final KiboRpcApi api;
  private final Context context;
  private final String TAG = this.getClass().getSimpleName();

  public ItemDetector(Context context, KiboRpcApi apiRef) {
    this.api = apiRef;
    this.context = context;
    Log.i(TAG, "Initialized");
  }
}