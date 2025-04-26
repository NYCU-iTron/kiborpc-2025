package jp.jaxa.iss.kibo.rpc.sampleapk;

import jp.jaxa.iss.kibo.rpc.api.KiboRpcApi;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Color;
import android.util.Log;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.support.common.FileUtil;
import org.tensorflow.lite.support.image.TensorImage;
import org.tensorflow.lite.support.image.ImageProcessor;
import org.tensorflow.lite.support.common.ops.NormalizeOp;
import org.tensorflow.lite.support.common.ops.CastOp;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

import java.io.IOException;
import java.io.InputStream;
import java.nio.MappedByteBuffer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.HashMap;
import java.util.Map;
import java.util.Collections;
import java.util.Comparator;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.Random;
import java.util.Comparator;
import java.util.Iterator;

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

  private Interpreter interpreter;
  private List<String> labels;
  private final ImageProcessor imageProcessor;
  private Map<Integer, String> idToNameMap = new HashMap<>();
  private Random rand; // Deal with no item detected

  private int tensorWidth;
  private int tensorHeight;
  private int numChannel;
  private int numElements;

  private static final float CONFIDENCE_THRESHOLD = 0.7f;
  private static final float NMS_IOU_THRESHOLD = 0.95f;
  private static final float IOU_THRESHOLD = 0.4F;
  private static final float INPUT_MEAN = 0.0F;
  private static final float INPUT_STANDARD_DEVIATION = 255.0F;
  private static final DataType INPUT_IMAGE_TYPE = DataType.FLOAT32;

  /**
   * Constructor
   */
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

    this.imageProcessor = new ImageProcessor.Builder()
        .add(new NormalizeOp(INPUT_MEAN, INPUT_STANDARD_DEVIATION))
        .add(new CastOp(INPUT_IMAGE_TYPE))
        .build();

    Log.i(TAG, "Initialized");
  }

  /**
   * Detects items in the given undistorted image using a TensorFlow Lite model.
   *
   * @param undistortImage The input image as a Mat object, which has been undistorted.
   * @return A list of detected items, or an empty list if the TensorFlow Lite model is not initialized.
   */
  public List<float[]> detect(Mat undistortImage) {
    if (interpreter == null) {
      Log.w(TAG, "TFLite model not loaded, detect operation aborted.");
      return new ArrayList<>();
    }

    ByteBuffer imageBuffer = preprocess(undistortImage);
    float[] outputArray = runReference(imageBuffer);
    List<float[]> detections = parsePredictions(outputArray);
    List<float[]> results = applyNonMaximumSuppression(detections);

    return results;
  }

  /**
   * Filter the Result
   */
  public Item[] filterResult(List<float[]> detectResult, int area, Pose tagPose) {
    int treasureId = -1;
    int landmarkId = -1;
    float treasureMaxConfidence = -1.0f;
    float landmarkMaxConfidence = -1.0f;
    int landmarkCount = 0;

    Map<Integer, Integer> itemCountMap = new HashMap<>();
    Item[] results = new Item[2];

    if (detectResult.isEmpty()) {
      Log.w(TAG, "No detection found. Leave it to fate.");
      treasureId = rand.nextInt(3) + 11;
      landmarkId = rand.nextInt(8) + 21;
      landmarkCount = rand.nextInt(3) + 1;
    } else {
      for (float[] det : detectResult) {
        String label = labels.get((int) det[9]);
        int itemId = Integer.parseInt(label);

        // Count the duplicate
        itemCountMap.put(itemId, itemCountMap.getOrDefault(itemId, 0) + 1);

        // Record the Treasure item and Landmark item with the max confidence respectively
        if (itemId / 10 == 1) { // Treasure Item
          if (det[8] > treasureMaxConfidence) {
            treasureMaxConfidence = det[8];
            treasureId = itemId;
          }
        } else if (itemId / 10 == 2) { // Landmark Item
          if (det[8] > landmarkMaxConfidence) {
            landmarkMaxConfidence = det[8];
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

  /**
   * Draw Bounding Boxes
   */
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

  /* ----------------------------- Tool Functions ----------------------------- */

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
   * Setup model
   */
  private void setupModel() throws IOException {
    InputStream inputStream = context.getAssets().open("best.tflite");
    ByteBuffer model = ByteBuffer.allocateDirect(inputStream.available());
    byte[] buffer = new byte[inputStream.available()];
    inputStream.read(buffer);
    model.put(buffer);
    model.rewind();
    Interpreter.Options options = new Interpreter.Options();
    options.setNumThreads(4); // Set the thread
    interpreter = new Interpreter(model, options);
    labels = FileUtil.loadLabels(context, "labels.txt");

    int[] inputShape = interpreter.getInputTensor(0).shape();
    int[] outputShape = interpreter.getOutputTensor(0).shape();

    tensorWidth = inputShape[1];
    tensorHeight = inputShape[2];
    numChannel = outputShape[1];
    numElements = outputShape[2];

    Log.i(TAG, "Input shape: " + Arrays.toString(inputShape));
    Log.i(TAG, "Output shape: " + Arrays.toString(outputShape));
    Log.i(TAG, "numChannel: " + numChannel + ", numElements: " + numElements);
  }

  /**
   * Preprocesses the input image with letterbox resizing and padding.
   * Returns [inputBuffer, padX, padY].
   */
  private ByteBuffer preprocess(Mat undistortImage) {
    // Convert Mat to Bitmap
    Bitmap bitmap = Bitmap.createBitmap(undistortImage.cols(), undistortImage.rows(), Bitmap.Config.ARGB_8888);
    Utils.matToBitmap(undistortImage, bitmap);
    Bitmap resizedBitmap = Bitmap.createScaledBitmap(bitmap, tensorWidth, tensorHeight, false);
    Log.i(TAG, "Original image size: " + undistortImage.cols() + "x" + undistortImage.rows());
    Log.i(TAG, "Resized to: " + tensorWidth + "x" + tensorHeight);
    
    TensorImage tensorImage = new TensorImage(DataType.FLOAT32);
    tensorImage.load(resizedBitmap);
    TensorImage processedImage = imageProcessor.process(tensorImage);
    ByteBuffer imageBuffer = processedImage.getBuffer();

    return imageBuffer;
  }

  /**
   * Run reference
   */
  private float[] runReference(ByteBuffer imageBuffer) {
    TensorBuffer output = TensorBuffer.createFixedSize(new int[]{1, numChannel, numElements}, DataType.FLOAT32);
    interpreter.run(imageBuffer, output.getBuffer());
    float[] outputArray = output.getFloatArray();

    return outputArray;
  }

  /**
   * Parse prediction
   */
  private List<float[]> parsePredictions(float[] outputArray) {
   List<float[]> detections = new ArrayList<>();
    for (int c = 0; c < numElements; c++) {
      float maxConf = -1.0F;
      int maxIdx = -1;
      for (int j = 4; j < numChannel; j++) {
        int arrayIdx = c + numElements * j;
        if (outputArray[arrayIdx] > maxConf) {
          maxConf = outputArray[arrayIdx];
          maxIdx = j - 4;
        }
      }

      if (maxConf > CONFIDENCE_THRESHOLD) {
        float cx = outputArray[c];
        float cy = outputArray[c + numElements];
        float w = outputArray[c + numElements * 2];
        float h = outputArray[c + numElements * 3];

        Log.i(TAG, "High confidence detection: cx=" + cx + ", cy=" + cy + ", w=" + w + ", h=" + h);

        float x1 = cx - w / 2.0F;
        float y1 = cy - h / 2.0F;
        float x2 = cx + w / 2.0F;
        float y2 = cy + h / 2.0F;

        Log.i(TAG, "  Bounding box: (" + x1 + "," + y1 + ") to (" + x2 + "," + y2 + ")");

        if (x1 >= 0.0F && x1 <= 1.0F && y1 >= 0.0F && y1 <= 1.0F && x2 >= 0.0F && x2 <= 1.0F && y2 >= 0.0F && y2 <= 1.0F) {
          detections.add(new float[]{x1, y1, x2, y2, cx, cy, w, h, maxConf, maxIdx});
          Log.i(TAG, "  Added valid detection with confidence " + maxConf + " for class " + maxIdx);
        } else {
          Log.i(TAG, "  Detection out of bounds, skipping");
        }
      }
    }
    Log.i(TAG, "Total valid detections before NMS: " + detections.size());
    return detections;
  }

  /**
   * Apply Non Maximum Suppression
   */
  private List<float[]> applyNonMaximumSuppression(List<float[]> detections) {
    List<float[]> results = new ArrayList<>();

    Log.i(TAG, "Detection count before NMS: " + detections.size());
    if (!detections.isEmpty()) {

      detections.sort(new Comparator<float[]>() {
        @Override
        public int compare(float[] o1, float[] o2) {
          return Float.compare(o2[8], o1[8]);
        }
      });

      while (!detections.isEmpty()) {
        float[] first = detections.remove(0);
        results.add(first);
        Iterator<float[]> iterator = detections.iterator();
        while (iterator.hasNext()) {
          float[] nextBox = iterator.next();
          if (calculateIoU(first, nextBox) >= IOU_THRESHOLD) {
            iterator.remove();
          }
        }
      }
    }

    Log.i(TAG, "Final detection count after NMS: " + results.size());
    for (int i = 0; i < results.size(); i++) {
      float[] det = results.get(i);
      Log.i(TAG, "Detection " + i + ": class=" + (int)det[9] + 
            ", confidence=" + det[8] + 
            ", box=(" + det[0] + "," + det[1] + "," + det[2] + "," + det[3] + ")");
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