package jp.jaxa.iss.kibo.rpc.sampleapk;

import gov.nasa.arc.astrobee.types.Point;
import gov.nasa.arc.astrobee.types.Quaternion;


/**
 * Class to represent a 3D pose consisting of a position and orientation.
 * 
 * @param point The position in 3D space.
 * @param quaternion The orientation represented as a quaternion.
 */
public class Pose {
  private final Point point;
  private final Quaternion quaternion;

  public Pose() {
    this.point = new Point();
    this.quaternion = new Quaternion();
  }

  public Pose(Point point, Quaternion quaternion) {
    this.point = point;
    this.quaternion = quaternion;
  }

  public Point getPoint() {
    return point;
  }

  public Quaternion getQuaternion() {
    return quaternion;
  }

  public String toString() {
    return "Pose{point=" + point + ", quaternion=" + quaternion + "}";
  }
}
