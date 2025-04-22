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
   * Constructor
   * 
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