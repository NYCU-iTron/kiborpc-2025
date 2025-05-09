package jp.jaxa.iss.kibo.rpc.sampleapk;

import org.opencv.core.Mat;


public class ARTag {
  private int markerId;
  private Mat corner;

  public ARTag() {
    this.markerId = -1;
    this.corner = null;
  }

  public ARTag(int markerId, Mat corner) {
    this.markerId = markerId;
    this.corner = corner;
  }

  public int getMarkerId() { return markerId; }
  public Mat getCorner() { return corner; }
}