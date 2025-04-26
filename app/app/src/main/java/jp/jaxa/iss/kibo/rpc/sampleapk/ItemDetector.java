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
  private static final float IOU_THRESHOLD = 0.4F;
  private static final float INPUT_MEAN = 0.0F;
  private static final float INPUT_STANDARD_DEVIATION = 255.0F;

  /**
   * Constructor for ItemDetector.
   * Sets up the model, label mappings, and image processor.
   *
   * @param context the application context
   * @param apiRef  reference to the KiboRpcApi
   */
  public ItemDetector(Context context, KiboRpcApi apiRef) {
    this.api = apiRef;
    this.context = context;

    initializeItemMappings();  // Initialize item ID mappings
    rand = new Random();       // Create a random generator

    try {
      setupModel();
    } catch (IOException e) {
      Log.e(TAG, "Failed to load model or labels", e);
    }

    // Build an image processor: normalize and cast input images
    this.imageProcessor = new ImageProcessor.Builder()
        .add(new NormalizeOp(INPUT_MEAN, INPUT_STANDARD_DEVIATION))
        .add(new CastOp(DataType.FLOAT32))
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
   * Filters the detection results to identify the treasure and landmark items with the highest confidence.
   *
   * @param detectResult A list of detection results, each containing item details and confidence scores.
   * @param area         The area identifier where the detection occurred.
   * @param tagPose      The pose associated with the detection area.
   * @return An array containing the selected treasure and landmark items.
   */
  public Item[] filterResult(List<float[]> detectResult, int area, Pose tagPose) {
    int treasureId = -1;
    int landmarkId = -1;
    float treasureMaxConfidence = -1.0f;
    float landmarkMaxConfidence = -1.0f;
    int treasureCount = 0;
    int landmarkCount = 0;
    int defaultCount = 0;

    Map<Integer, Integer> itemCountMap = new HashMap<>();
    Item[] results = new Item[2];

    // Handle empty detection results
    if (detectResult.isEmpty()) {
      Log.w(TAG, "No detection found. Leave it to fate.");
      treasureId = rand.nextInt(3) + 11; // Random treasure ID (11-13)
      landmarkId = rand.nextInt(8) + 21; // Random landmark ID (21-28)
      landmarkCount = rand.nextInt(3) + 1; // Random count (1-3)
      treasureCount = 1;
    } else {
      for (float[] det : detectResult) {
        String label = labels.get((int) det[9]); // Get item label from detection
        int itemId = Integer.parseInt(label);   // Convert label to item ID

        // Update item count
        itemCountMap.put(itemId, itemCountMap.getOrDefault(itemId, 0) + 1);

        // Check if the item is a treasure
        if (itemId / 10 == 1) { // Treasure IDs are in the range 10-19
          if (det[8] > treasureMaxConfidence) { // Update if higher confidence
            treasureMaxConfidence = det[8];
            treasureId = itemId;
          }
        }
        // Check if the item is a landmark
        else if (itemId / 10 == 2) { // Landmark IDs are in the range 20-29
          if (det[8] > landmarkMaxConfidence) { // Update if higher confidence
            landmarkMaxConfidence = det[8];
            landmarkId = itemId;
          }
        } else {
          Log.w(TAG, "Unknown item ID: " + itemId); // Log invalid IDs
        }
      }

      // Assign random IDs if no landmark was found
      if (landmarkId == -1) landmarkId = rand.nextInt(8) + 21; // Random landmark ID

      // Set item count
      defaultCount = rand.nextInt(3) + 1; // Random count (1-3)
      landmarkCount = itemCountMap.getOrDefault(landmarkId, defaultCount);
      treasureCount = itemCountMap.getOrDefault(landmarkId, 0);
    }

    // Create the resulting items
    if (treasureId == -1) {
      results[0] = new Item(area, landmarkId, idToNameMap.get(landmarkId), landmarkCount, tagPose);
    } else {
      results[0] = new Item(area, treasureId, idToNameMap.get(treasureId), treasureCount, tagPose);
      results[1] = new Item(area, landmarkId, idToNameMap.get(landmarkId), landmarkCount, tagPose);
    }

    return results; // Return the filtered treasure and landmark items
  }

  /**
   * Draws bounding boxes on the given image based on the detection results.
   *
   * @param undistortImage The input image to draw on.
   * @param detectResult   List of detection results containing bounding box coordinates.
   * @param area           The area identifier used for saving the image.
   */
  public void drawBoundingBoxes(Mat undistortImage, List<float[]> detectResult, int area) {
    if (detectResult == null || detectResult.isEmpty()) {
      Log.i(TAG, "No detections to draw.");
      return;
    }

    int imageWidth = undistortImage.cols();
    int imageHeight = undistortImage.rows();

    for (float[] det : detectResult) {
      // Extract bounding box coordinates
      int x1 = (int) (det[0] * imageWidth);
      int y1 = (int) (det[1] * imageHeight);
      int x2 = (int) (det[2] * imageWidth);
      int y2 = (int) (det[3] * imageHeight);

      Log.i(TAG, "Draw Bounding Boxe at (" + x1 + "," + y1 + "," + x2 + "," + y2 + ")");

      // Create a rectangle from the top-left and bottom-right points
      Rect rect = new Rect(new Point(x1, y1), new Point(x2, y2));

      // Define rectangle color (red) and thickness
      Scalar color = new Scalar(255, 0, 0); // Red color
      int thickness = 2;

      // Draw the rectangle on the image
      Imgproc.rectangle(undistortImage, rect, color, thickness);
    }

    api.saveMatImage(undistortImage, String.format("area%d_bbox.png", area));
  }

  /* ----------------------------- Tool Functions ----------------------------- */

  /**
   * Initializes the mapping between item IDs and item names.
   */
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
    idToNameMap.put(28, "treasure_box");
  }

  /**
   * Loads the TensorFlow Lite model and label file, and initializes interpreter settings.
   *
   * @throws IOException if the model or label files cannot be loaded.
   */
  private void setupModel() throws IOException {
    // Load the TFLite model from assets
    InputStream inputStream = context.getAssets().open("best.tflite");
    ByteBuffer model = ByteBuffer.allocateDirect(inputStream.available());
    byte[] buffer = new byte[inputStream.available()];
    inputStream.read(buffer);
    model.put(buffer);
    model.rewind();

    // Create interpreter options and set number of threads
    Interpreter.Options options = new Interpreter.Options();
    options.setNumThreads(4);

    // Initialize the TFLite interpreter with the model
    interpreter = new Interpreter(model, options);

    // Load labels from labels.txt
    labels = FileUtil.loadLabels(context, "labels.txt");

    // Get input and output tensor shapes
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
   * Preprocesses the input image by resizing and normalizing it.
   *
   * @param undistortImage The input image as a Mat object.
   * @return A ByteBuffer ready for model input.
   */
  private ByteBuffer preprocess(Mat undistortImage) {
    // Convert Mat to Bitmap format
    Bitmap bitmap = Bitmap.createBitmap(undistortImage.cols(), undistortImage.rows(), Bitmap.Config.ARGB_8888);
    Utils.matToBitmap(undistortImage, bitmap);

    // Resize Bitmap to match the model input size
    Bitmap resizedBitmap = Bitmap.createScaledBitmap(bitmap, tensorWidth, tensorHeight, false);

    Log.i(TAG, "Original image size: " + undistortImage.cols() + "x" + undistortImage.rows());
    Log.i(TAG, "Resized to: " + tensorWidth + "x" + tensorHeight);

    // Load the resized Bitmap into a TensorImage
    TensorImage tensorImage = new TensorImage(DataType.FLOAT32);
    tensorImage.load(resizedBitmap);

    // Apply image preprocessing (normalization, type casting, etc.)
    TensorImage processedImage = imageProcessor.process(tensorImage);

    // Get the ByteBuffer from the processed image
    ByteBuffer imageBuffer = processedImage.getBuffer();

    return imageBuffer;
  }

  /**
   * Runs the TensorFlow Lite model inference on the input image buffer.
   *
   * @param imageBuffer The preprocessed input image as a ByteBuffer.
   * @return The model output as a float array.
   */
  private float[] runReference(ByteBuffer imageBuffer) {
    // Create an output buffer with the expected shape and type
    TensorBuffer output = TensorBuffer.createFixedSize(new int[]{1, numChannel, numElements}, DataType.FLOAT32);

    // Run inference with the interpreter
    interpreter.run(imageBuffer, output.getBuffer());

    // Convert the output to a float array
    float[] outputArray = output.getFloatArray();

    return outputArray;
  }

  /**
   * Parses the raw model output into a list of valid detections.
   *
   * @param outputArray The raw output array from the model.
   * @return A list of detected bounding boxes and related information.
   */
  private List<float[]> parsePredictions(float[] outputArray) {
    List<float[]> detections = new ArrayList<>();

    // Loop through each element (candidate detection)
    for (int c = 0; c < numElements; c++) {
      float maxConf = -1.0F;
      int maxIdx = -1;

      // Find the class with the highest confidence for this candidate
      for (int j = 4; j < numChannel; j++) {
        int arrayIdx = c + numElements * j;
        if (outputArray[arrayIdx] > maxConf) {
          maxConf = outputArray[arrayIdx];
          maxIdx = j - 4; // Class index adjustment
        }
      }

      // If confidence is high enough, process the detection
      if (maxConf > CONFIDENCE_THRESHOLD) {
        // Get center coordinates and size
        float cx = outputArray[c];
        float cy = outputArray[c + numElements];
        float w = outputArray[c + numElements * 2];
        float h = outputArray[c + numElements * 3];

        // Calculate bounding box corners
        float x1 = cx - w / 2.0F;
        float y1 = cy - h / 2.0F;
        float x2 = cx + w / 2.0F;
        float y2 = cy + h / 2.0F;

        // Check if the bounding box is valid (inside [0, 1] range)
        if (x1 >= 0.0F && x1 <= 1.0F && y1 >= 0.0F && y1 <= 1.0F && x2 >= 0.0F && x2 <= 1.0F) {
          // Add valid detection [box coords, center, size, confidence, class]
          detections.add(new float[]{x1, y1, x2, y2, cx, cy, w, h, maxConf, maxIdx});
        } else {
          // Skip detections that are out of bounds
          Log.i(TAG, "  Detection out of bounds, skipping");
        }
      }
    }

    Log.i(TAG, "Total valid detections before NMS: " + detections.size());
    return detections;
  }


  /**
   * Applies Non-Maximum Suppression (NMS) to remove overlapping detections.
   *
   * @param detections List of raw detections [x1, y1, x2, y2, ..., confidence, class]
   * @return List of filtered detections after NMS
   */
  private List<float[]> applyNonMaximumSuppression(List<float[]> detections) {
    List<float[]> results = new ArrayList<>();

    Log.i(TAG, "Detection count before NMS: " + detections.size());

    if (!detections.isEmpty()) {
      // Sort detections by confidence score (descending)
      detections.sort(new Comparator<float[]>() {
        @Override
        public int compare(float[] o1, float[] o2) {
          return Float.compare(o2[8], o1[8]); // Confidence is at index 8
        }
      });

      // Perform NMS
      while (!detections.isEmpty()) {
        // Pick the detection with highest confidence
        float[] first = detections.remove(0);
        results.add(first);

        Iterator<float[]> iterator = detections.iterator();
        while (iterator.hasNext()) {
          float[] nextBox = iterator.next();
          // Remove detections with IoU greater than threshold
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
   * Calculates the Intersection over Union (IoU) between two bounding boxes.
   * IoU is the ratio of the intersection area to the union area of two bounding boxes.
   *
   * @param box1 The first bounding box in the form [x1, y1, x2, y2].
   * @param box2 The second bounding box in the form [x1, y1, x2, y2].
   * @return The IoU value as a float between 0 and 1.
   */
  private float calculateIoU(float[] box1, float[] box2) {
    // Calculate the coordinates of the intersection rectangle
    float x1 = Math.max(box1[0], box2[0]);
    float y1 = Math.max(box1[1], box2[1]);
    float x2 = Math.min(box1[2], box2[2]);
    float y2 = Math.min(box1[3], box2[3]);

    // Calculate the area of the intersection
    float interArea = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);

    // Calculate the area of both bounding boxes
    float box1Area = (box1[2] - box1[0]) * (box1[3] - box1[1]); // width * height of box1
    float box2Area = (box2[2] - box2[0]) * (box2[3] - box2[1]); // width * height of box2

    // Return the IoU value: intersection area / union area
    return interArea / (box1Area + box2Area - interArea);
  }
}