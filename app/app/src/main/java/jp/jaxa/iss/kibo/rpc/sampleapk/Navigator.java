import jp.jaxa.iss.kibo.rpc.api.KiboRpcService;

import gov.nasa.arc.astrobee.types.Point;
import gov.nasa.arc.astrobee.types.Quaternion;
import gov.nasa.arc.astrobee.Kinematics;
import gov.nasa.arc.astrobee.Result;


public class Pose {
  private Point point;
  private Quaternion quaternion;

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
}

public class Navigator extends KiboRpcService {

  public Pose getCurrentPose() {
    // TODO : Deal with measure error and low confidence
    Kinemaics kinemaics = api.getRobotKinematics();
    return new Pose(kinemaics.getPosition(), kinemaics.getOrientation());
  }

  public void moveToTarget(Pose target) {
    Result result = api.moveTo(target.getPoint(), target.getQuaternion());
  }

  public static List<Pose> interpolate(Pose start, Pose end) {
    float linearUnit = 0.08f;

    float dx = targetPoint.getX() - currentPoint.getX();
    float dy = targetPoint.getY() - currentPoint.getY();
    float dz = targetPoint.getZ() - currentPoint.getZ();

    float distance = Math.sqrt(dx * dx + dy * dy + dz * dz);
    int numSteps = (int) (distance / linearUnit);
    List<Pose> poses = new ArrayList<>();

    if (numSteps == 0) {
      poses.add(end);
      return poses;
    }
    
    for (int i = 0; i <= numSteps; i++) {
      float t = (float) i / numSteps;
      float x = currentPoint.getX() + t * dx;
      float y = currentPoint.getY() + t * dy;
      float z = currentPoint.getZ() + t * dz;

      // TODO : Interpolate quaternion using SLERP
      poses.add(new Pose(new Point(x, y, z), current.getQuaternion()));
    }

    return poses;
  }

  public static Pose getPoseToFaceTarget(Pose current, Pose target) {
    Point currentPoint = current.getPoint();
    Point targetPoint = target.getPoint();

    // Calculate direction vector
    float dx = targetPoint.getX() - currentPoint.getX();
    float dy = targetPoint.getY() - currentPoint.getY();
    float dz = targetPoint.getZ() - currentPoint.getZ();

    // Normalize the direction vector
    float length = Math.sqrt(dx * dx + dy * dy + dz * dz);
    if (length == 0) {
      return current; // No movement needed
    }

    dx /= length;
    dy /= length;
    dz /= length;

    // Default direction vector
    float ux = 1f;
    float uy = 0f;
    float uz = 0f;

    // Dot product to find angle
    float dot = ux * vx + uy * vy + uz * vz;
    float theta = (float) Math.acos(dot);

    // Special cases
    if (Math.abs(dot - 1) < 1e-6) { // Already aligned with default direction
      return current;
    } else if (Math.abs(dot + 1) < 1e-6) { // 180° opposite to default direction
      // Choose an arbitrary perpendicular axis (e.g., x-axis or y-axis)
      return new Pose(currentPoint, new Quaternion(1, 0, 0, 0)); // If aligned with x, rotate around x
    }

    // Cross product to find rotation axis
    float axisX = uy * vz - uz * vy;
    float axisY = uz * vx - ux * vz;
    float axisZ = ux * vy - uy * vx;
    float axisMagnitude = (float) Math.sqrt(axisX * axisX + axisY * axisY + axisZ * axisZ);

    // Avoid division by zero
    if (axisMagnitude < 1e-6) {
      return new Pose(currentPoint, new Quaternion(0, 0, 0, 1));
    }

    axisX /= axisMagnitude;
    axisY /= axisMagnitude;
    axisZ /= axisMagnitude;

    // Calculate quaternion
    float halfTheta = theta / 2;
    float sinHalfTheta = (float) Math.sin(halfTheta);
    float qx = axisX * sinHalfTheta;
    float qy = axisY * sinHalfTheta;
    float qz = axisZ * sinHalfTheta;
    float qw = (float) Math.cos(halfTheta);

    return new Pose(currentPoint, new Quaternion(qx, qy, qz, qw));
  }
}
