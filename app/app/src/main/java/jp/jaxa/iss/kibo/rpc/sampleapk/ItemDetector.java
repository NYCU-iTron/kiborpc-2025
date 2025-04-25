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

import org.opencv.core.Mat;
import org.opencv.android.Utils;

/**
 * Class to detect items from provided image using YOLO TFLite model
 */
public class ItemDetector {
  private final KiboRpcApi api;
  private final Context context;
  private final String TAG = this.getClass().getSimpleName();

  private Interpreter tflite;
  private List<String> labels;
  private static final float CONFIDENCE_THRESHOLD = 0.2f;
  private static final float NMS_IOU_THRESHOLD = 0.95f;

  public ItemDetector(Context context, KiboRpcApi apiRef) {
    this.api = apiRef;
    this.context = context;

    try {
      setupModel();
    } catch (IOException e) {
      Log.e(TAG, "Failed to load model or labels", e);
    }

    Log.i(TAG, "Initialized");
  }

  /**
   * Detects items in the given undistorted image using a TensorFlow Lite model.
   *
   * @param undistortImage The input image as a Mat object, which has been undistorted.
   * @return A list of detected items, or an empty list if the TensorFlow Lite model is not initialized.
   */
  public List<Item> detect(Mat undistortImage) {
    if (tflite == null) return Collections.emptyList();

    // Get input tensor shape
    int[] inputShape = tflite.getInputTensor(0).shape();
    int height = inputShape[1]; // 640
    int width = inputShape[2]; // 640

    // Preprocess image
    Object[] preprocessResult = preprocess(undistortImage, width, height);
    ByteBuffer inputBuffer = (ByteBuffer) preprocessResult[0];
    float padX = (float) preprocessResult[1];
    float padY = (float) preprocessResult[2];

    // Run inference
    int[] outputShape = tflite.getOutputTensor(0).shape(); // [1, 15, 8400]
    float[][][] output = new float[outputShape[0]][outputShape[1]][outputShape[2]];
    tflite.run(inputBuffer, output);

    // Postprocess and return results
    return postprocess(output[0], undistortImage.cols(), undistortImage.rows(), width, height, padX, padY);
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
    float r = Math.min((float) inputW / origW, (float) inputH / origH);
    int newW = Math.round(origW * r);
    int newH = Math.round(origH * r);
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

    return new Object[]{inputBuffer, (float) dw / inputW, (float) dh / inputH};
  }

  /**
   * Processes the predictions from the TensorFlow Lite model to extract detected items.
   *
   * @param preds A 2D array of predictions [features, predictions].
   * @param origW The width of the original image.
   * @param origH The height of the original image.
   * @param inputW The width of the model input.
   * @param inputH The height of the model input.
   * @param padX The left padding ratio.
   * @param padY The top padding ratio.
   * @return A list of detected items.
   */
  private List<Item> postprocess(float[][] preds, int origW, int origH, int inputW, int inputH, float padX, float padY) {
    List<Item> results = new ArrayList<>();
    Map<String, Integer> itemCountMap = new HashMap<>();
    int itemId = 0;

    // Collect detections and track top 3 for debugging
    List<float[]> detections = new ArrayList<>();
    List<float[]> topCandidates = new ArrayList<>(); // For top 3 candidates
    for (int i = 0; i < preds[0].length; i++) {
      float cx = preds[0][i] - padX;
      float cy = preds[1][i] - padY;
      float w = preds[2][i];
      float h = preds[3][i];
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

      // Store top 3 candidates regardless of threshold
      float scale = Math.max(origW, origH);
      float x1 = (cx - w / 2) * scale;
      float y1 = (cy - h / 2) * scale;
      float x2 = (cx + w / 2) * scale;
      float y2 = (cy + h / 2) * scale;
      topCandidates.add(new float[]{x1, y1, x2, y2, maxProb, classId});

      // Add to detections if above threshold
      if (conf > CONFIDENCE_THRESHOLD && maxProb > CONFIDENCE_THRESHOLD) {
        detections.add(new float[]{x1, y1, x2, y2, maxProb, classId});
      }
    }

    // Simple NMS
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

    // Log detections or top 3 candidates if no detections
    if (filteredDetections.isEmpty()) {
      Log.w(TAG, "No detections found. Logging top 3 candidates by confidence:");
      
      // Sort candidates by score (descending) using Comparator for Java 7
      Collections.sort(topCandidates, new Comparator<float[]>() {
        @Override
        public int compare(float[] a, float[] b) {
          return Float.compare(b[4], a[4]); // Descending order
        }
      });

      // Log top 3 (or fewer if less than 3)
      for (int i = 0; i < Math.min(3, topCandidates.size()); i++) {
        float[] cand = topCandidates.get(i);
        String label = labels.get((int) cand[5]);
        Log.i(TAG, String.format("Candidate %d: %s, Confidence: %.2f, BBox: [%.1f, %.1f, %.1f, %.1f]",
                i + 1, label, cand[4], cand[0], cand[1], cand[2], cand[3]));
      }
    } else {
      for (float[] det : filteredDetections) {
        String label = labels.get((int) det[5]);
        itemCountMap.put(label, itemCountMap.getOrDefault(label, 0) + 1);
        Log.i(TAG, String.format("Detected item: %s, Confidence: %.2f, BBox: [%.1f, %.1f, %.1f, %.1f]",
                label, det[4], det[0], det[1], det[2], det[3]));

        // TODO:  Create item
        // Item item = new Item(...);
        // results.add(item);
      }
    }

    return results;
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
}