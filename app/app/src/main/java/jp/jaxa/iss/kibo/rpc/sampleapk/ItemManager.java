package jp.jaxa.iss.kibo.rpc.sampleapk;

import jp.jaxa.iss.kibo.rpc.api.KiboRpcApi;

import java.util.Map;
import java.util.HashMap;

import android.util.Log;


/**
 * Class to manage the detected item information
 */
public class ItemManager {
  private final KiboRpcApi api;
  private final String TAG = this.getClass().getSimpleName();
  private final Map<Integer, Item> treasureMap = new HashMap<>();

  public ItemManager(KiboRpcApi apiRef) {
    this.api = apiRef;
    Log.i(TAG, "Initialized");
  }

  public void setAreaInfo(Item item) {
    api.setAreaInfo(item.getAreaId(), item.getItemName(), item.getItemCount());
  }

  public void storeTreasureInfo(Item item) {
    treasureMap.put(item.getItemId(), item);
  }

  public Item getTreasureInfo(int itemId) {
    return treasureMap.get(itemId);
  }
}