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

  /**
   * Constructor
   * 
   * @param apiRef API reference.
   * 
   * Example of using the ItemManager constructor:
   * @code
   * ItemManager itemManager = new ItemManager(api);
   * @endcode
   */
  public ItemManager(KiboRpcApi apiRef) {
    this.api = apiRef;
    Log.i(TAG, "Initialized");
  }

  /**
   * Set the area information of the item. Often used with the storeTreasureInfo method.
   * 
   * @param item Item object containing area information. See Item class for details.
   * 
   * Example of recording area information:
   * @code
   * Item item = new Item(areaId, itemId, itemName, itemCount, itemPose);
   * itemManager.setAreaInfo(item);
   * itemManager.storeTreasureInfo(item);
   * @endcode
   */
  public void setAreaInfo(Item item) {
    api.setAreaInfo(item.getAreaId(), item.getItemName(), item.getItemCount());
  }

  /**
   * Store the treasure information in the map. Often used with the setAreaInfo method.
   * 
   * @param item Item object containing treasure information. See Item class for details.
   * 
   * See @ref setAreaInfo for an example of how to use this method.
   */
  public void storeTreasureInfo(Item item) {
    treasureMap.put(item.getItemId(), item);
  }

  /**
   * Get the treasure information from the map using the item ID.
   * 
   * @param itemId Item ID of the treasure.
   * @return Item object containing treasure information. See Item class for details.
   */
  public Item getTreasureInfo(int itemId) {
    if (treasureMap.containsKey(itemId)) {
      return treasureMap.get(itemId);
    }
    else {
      return new Item();
    }
  }

  /**
   * Get the treasure information from the map using the item object.
   * 
   * @param item Item object containing treasure information. See Item class for details.
   * @return Item object containing treasure information. See Item class for details.
   */
  public Item getTreasureInfo(Item item) {
    if (treasureMap.containsKey(item.getItemId())) {
      return treasureMap.get(item.getItemId());
    }
    else {
      return new Item();
    }
  }
}
