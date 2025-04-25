package jp.jaxa.iss.kibo.rpc.sampleapk;

import jp.jaxa.iss.kibo.rpc.api.KiboRpcApi;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Color;
import android.util.Log;

import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.support.common.FileUtil;

import java.io.IOException;
import java.nio.MappedByteBuffer;
import java.util.ArrayList;
import java.util.List;
import java.util.HashMap;
import java.util.Map;
import java.util.Collections;
import java.util.Comparator;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.Random;

import org.opencv.core.Mat;
import org.opencv.android.Utils;
import org.opencv.core.Core;
import org.opencv.core.Scalar;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.imgproc.Imgproc;
import org.opencv.utils.Converters;


/**
 * Class to detect items from provided image using YOLO TFLite model
 */
public class ItemDetector {
  private final KiboRpcApi api;
  private final Context context;
  private final String TAG = this.getClass().getSimpleName();
  private boolean DEBUG = true;  

  private Interpreter tflite;
  private List<String> labels;
  private static final float CONFIDENCE_THRESHOLD = 0.85f;
  private static final float NMS_IOU_THRESHOLD = 0.95f;
  private Map<Integer, String> idToNameMap = new HashMap<>();
  private Random rand;

  public ItemDetector(Context context, KiboRpcApi apiRef) {
    this.api = apiRef;
    this.context = context;

    initializeItemMappings();
    rand = new Random();

    try {
      setupModel();
    } catch (IOException e) {
      Log.e(TAG, "Failed to load model or labels", e);
    }

    Log.i(TAG, "Initialized");
  }

  private void initializeItemMappings() {
    // Treasure
    idToNameMap.put(11, "crystal");
    idToNameMap.put(12, "diamond");
    idToNameMap.put(13, "emerald");

    // Landmark
    idToNameMap.put(21, "coin");
    idToNameMap.put(22, "compass");
    idToNameMap.put(23, "coral");
    idToNameMap.put(24, "fossil");
    idToNameMap.put(25, "key");
    idToNameMap.put(26, "letter");
    idToNameMap.put(27, "shell");
    idToNameMap.put(28, "Treasure_box");
  }

  /**
   * Detects items in the given undistorted image using a TensorFlow Lite model.
   *
   * @param undistortImage The input image as a Mat object, which has been undistorted.
   * @return A list of detected items, or an empty list if the TensorFlow Lite model is not initialized.
   */
  public List<float[]> detect(Mat undistortImage) {
    if (tflite == null) {
      Log.w(TAG, "TFLite model not loaded.");
      return new ArrayList<>();
    }

    // Get input tensor shape
    int[] inputShape = tflite.getInputTensor(0).shape();
    int inputWidth = inputShape[2]; // 640
    int inputHeight = inputShape[1]; // 640

    // Preprocess image
    Object[] preprocessResult = preprocess(undistortImage, inputWidth, inputHeight);
    ByteBuffer inputBuffer = (ByteBuffer) preprocessResult[0];
    int padPixelX = (int) preprocessResult[1];
    int padPixelY = (int) preprocessResult[2];
    float rate = (float) preprocessResult[3];

    Log.i(TAG, "padPixelX: " + padPixelX);
    Log.i(TAG, "padPixelY: " + padPixelY);
    Log.i(TAG, "rate: " + rate);

    // Run inference
    int[] outputShape = tflite.getOutputTensor(0).shape(); // [1, 15, 8400]
    float[][][] output = new float[outputShape[0]][outputShape[1]][outputShape[2]];
    tflite.run(inputBuffer, output);

    // Parse the predictions
    List<float[]> detections = parsePredictions(output[0], padPixelX, padPixelY, rate);

    // Apply Non-Maximum Suppression (NMS)
    List<float[]> results = applyNonMaximumSuppression(detections);

    return results;
  }

  /* ----------------------------- Tool Functions ----------------------------- */

  /**
   * Setup model
   */
  private void setupModel() throws IOException {
    MappedByteBuffer modelBuffer = FileUtil.loadMappedFile(context, "best.tflite");
    tflite = new Interpreter(modelBuffer);
    labels = FileUtil.loadLabels(context, "labels.txt");
  }

  /**
   * Preprocesses the input image with letterbox resizing and padding.
   * Returns [inputBuffer, padX, padY].
   */
  private Object[] preprocess(Mat undistortImage, int inputW, int inputH) {
    // Convert Mat to Bitmap
    Bitmap bitmap = Bitmap.createBitmap(undistortImage.cols(), undistortImage.rows(), Bitmap.Config.ARGB_8888);
    Utils.matToBitmap(undistortImage, bitmap);

    // Letterbox resize
    int origW = bitmap.getWidth();
    int origH = bitmap.getHeight();
    float rate = Math.min((float) inputW / origW, (float) inputH / origH);
    int newW = Math.round(origW * rate);
    int newH = Math.round(origH * rate);
    Bitmap resized = Bitmap.createScaledBitmap(bitmap, newW, newH, true);

    // Add padding
    int dw = (inputW - newW) / 2;
    int dh = (inputH - newH) / 2;
    Bitmap padded = Bitmap.createBitmap(inputW, inputH, Bitmap.Config.ARGB_8888);
    Canvas canvas = new Canvas(padded);
    canvas.drawColor(Color.rgb(114, 114, 114)); // Gray padding
    canvas.drawBitmap(resized, dw, dh, null);

    // Prepare input buffer (float32)
    ByteBuffer inputBuffer = ByteBuffer.allocateDirect(1 * inputW * inputH * 3 * 4);
    inputBuffer.order(ByteOrder.nativeOrder());
    int[] intValues = new int[inputW * inputH];
    padded.getPixels(intValues, 0, inputW, 0, 0, inputW, inputH);

    for (int pixel : intValues) {
      inputBuffer.putFloat(((pixel >> 16) & 0xFF) / 255.0f); // R
      inputBuffer.putFloat(((pixel >> 8) & 0xFF) / 255.0f);  // G
      inputBuffer.putFloat((pixel & 0xFF) / 255.0f);         // B
    }

    return new Object[]{inputBuffer, (int) dw, (int) dh, (float) rate};
  }

  private List<float[]> parsePredictions(float[][] preds, int padPixelX, int padPixelY, float rate) {
    List<float[]> detections = new ArrayList<>();
    for (int i = 0; i < preds[0].length; i++) {
      float cx = (preds[0][i] - padPixelX) / rate;
      float cy = (preds[1][i] - padPixelY) / rate;
      float w = preds[2][i] / rate;
      float h = preds[3][i] / rate;

      float x1 = cx - w / 2;
      float y1 = cy - h / 2;
      float x2 = cx + w / 2;
      float y2 = cy + h / 2;

      float conf = preds[4][i];

      // Calculate max probability and class
      float maxProb = -1f;
      int classId = -1;
      for (int j = 5; j < preds.length; j++) {
        float prob = preds[j][i] * conf;
        if (prob > maxProb) {
          maxProb = prob;
          classId = j - 5;
        }
      }
      Log.i(TAG,"Class " + i + " Prob: " + maxProb);

      // Add to detections if above threshold
      if (conf > CONFIDENCE_THRESHOLD && maxProb > CONFIDENCE_THRESHOLD) {
        detections.add(new float[]{x1, y1, x2, y2, maxProb, classId});
      }
    }
    return detections;
  }

  private List<float[]> applyNonMaximumSuppression(List<float[]> detections) {
    List<float[]> filteredDetections = new ArrayList<>();
    while (!detections.isEmpty()) {
      float[] best = detections.get(0);
      for (float[] det : detections) {
        if (det[4] > best[4]) best = det;
      }
      filteredDetections.add(best);
      detections.remove(best);

      // Remove overlapping boxes
      List<float[]> toRemove = new ArrayList<>();
      for (float[] det : detections) {
        float iou = calculateIoU(best, det);
        if (iou > NMS_IOU_THRESHOLD) toRemove.add(det);
      }
      detections.removeAll(toRemove);
    }
    return filteredDetections;
  }

  /**
   * Calculates IoU between two bounding boxes [x1, y1, x2, y2].
   */
  private float calculateIoU(float[] box1, float[] box2) {
    float x1 = Math.max(box1[0], box2[0]);
    float y1 = Math.max(box1[1], box2[1]);
    float x2 = Math.min(box1[2], box2[2]);
    float y2 = Math.min(box1[3], box2[3]);

    float interArea = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
    float box1Area = (box1[2] - box1[0]) * (box1[3] - box1[1]);
    float box2Area = (box2[2] - box2[0]) * (box2[3] - box2[1]);

    return interArea / (box1Area + box2Area - interArea);
  }

  public void drawBoundingBoxes(Mat undistortImage, List<float[]> detectResult, int area) {
    for (float[] det : detectResult) {
      float x1 = det[0];
      float y1 = det[1];
      float x2 = det[2];
      float y2 = det[3];

      // Create a rectangle to represent the bounding box
      Rect rect = new Rect(new Point((int) x1, (int) y1), new Point((int) x2, (int) y2));

      // Draw the rectangle on the image (color and thickness can be customized)
      Scalar color = new Scalar(255, 0, 0);  // Red color for the rectangle
      int thickness = 2;
      Imgproc.rectangle(undistortImage, rect, color, thickness);
    }

    api.saveMatImage(undistortImage, String.format("area%d_bbox.png", area));
  }

  public Item[] filterResult(List<float[]> filteredDetections, int area, Pose tagPose) {
    int treasureId = -1;
    int landmarkId = -1;
    float treasureMaxConfidence = -1.0f;
    float landmarkMaxConfidence = -1.0f;
    int landmarkCount = 0;

    Map<Integer, Integer> itemCountMap = new HashMap<>();
    Item[] results = new Item[2];

    if (filteredDetections.isEmpty()) {
      Log.w(TAG, "No detection found. Leave it to fate.");
      treasureId = rand.nextInt(3) + 11;
      landmarkId = rand.nextInt(8) + 21;
      landmarkCount = rand.nextInt(3) + 1;
    } else {
      for (float[] det : filteredDetections) {
        String label = labels.get((int) det[5]);
        int itemId = Integer.parseInt(label);

        // Count the duplicate
        itemCountMap.put(itemId, itemCountMap.getOrDefault(itemId, 0) + 1);

        // Record the Treasure item and Landmark item with the max confidence respectively
        if (itemId / 10 == 1) { // Treasure Item
          if (det[4] > treasureMaxConfidence) {
            treasureMaxConfidence = det[4];
            treasureId = itemId;
          }
        } else if (itemId / 10 == 2) { // Landmark Item
          if (det[4] > landmarkMaxConfidence) {
            landmarkMaxConfidence = det[4];
            landmarkId = itemId;
          }
        } else {
          Log.w(TAG, "Unknown item ID: " + itemId);
        }
      }

      // Handle no treasure or landmark case
      if (treasureId == -1) treasureId = rand.nextInt(3) + 11;
      if (landmarkId == -1) landmarkId = rand.nextInt(8) + 21;
      landmarkCount = itemCountMap.getOrDefault(landmarkId, 1);
    }

    results[0] = new Item(area, treasureId, idToNameMap.get(treasureId), 1, tagPose);
    results[1] = new Item(area, landmarkId, idToNameMap.get(landmarkId), landmarkCount, tagPose);

    return results;
  }
}