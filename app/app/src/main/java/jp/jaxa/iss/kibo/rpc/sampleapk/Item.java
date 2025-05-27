package jp.jaxa.iss.kibo.rpc.sampleapk;


/**
 * Class to represent a single detected item
 */
public class Item {
  private final int areaId;
  private final int itemId;
  private final String itemName;
  private final int itemCount;
  private final Pose itemPose;

  /**
   * Default constructor
   */
  public Item() {
    this.areaId = -1;
    this.itemId = -1;
    this.itemName = "";
    this.itemCount = 0;
    this.itemPose = new Pose();
  }

  /**
   * Constructor
   * 
   * @param areaId Area ID of the item. See following figure for the area ID for each area.
   * @image html area_id.png width=50%
   * @param itemId Item ID of the item. See following figure for the item ID for each item.
   * @param itemName Name of the item. See following figure for the item name for each item.
   * @image html item_name_id.png width=40%
   * @param itemCount Number of items detected.
   * @param itemPose Pose of the item. See Pose class for details.
   */
  public Item(int areaId, int itemId, String itemName, int itemCount, Pose itemPose) {
    this.areaId = areaId;
    this.itemId = itemId;
    this.itemName = itemName;
    this.itemCount = itemCount;
    this.itemPose = itemPose;
  }

  // Getters
  public int getAreaId() { return areaId; }
  public int getItemId() { return itemId; }
  public String getItemName() { return itemName; }
  public int getItemCount() { return itemCount; }
  public Pose getItemPose() { return itemPose; }
  
  public String toString() {
    return "Item {" +
            "itemId=" + itemId +
            ", areaId=" + areaId +
            ", itemCount=" + itemCount +
            ", itemPose=" + itemPose +
            '}';
  }
}
