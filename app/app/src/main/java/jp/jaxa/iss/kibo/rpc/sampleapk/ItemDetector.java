package jp.jaxa.iss.kibo.rpc.sampleapk;

import jp.jaxa.iss.kibo.rpc.api.KiboRpcApi;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.util.Log;
import android.content.res.AssetFileDescriptor;

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
import java.io.FileInputStream;
import java.io.File;
import java.nio.MappedByteBuffer;
import java.nio.ByteBuffer;
import java.nio.channels.FileChannel;
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

  private List<String> labels;
  private Map<ModelType, Interpreter> modelMap;
  private ImageProcessor imageProcessor;
  private Random rand;

  public enum ModelType {
    s_15000_0516,
    n_20000_0519,
  }

  /**
   * Class representing a detected item with its bounding box, confidence, class ID, and class name.
   * The weight of the model is set to 1.0 by default.
   */
  public class Detection {
    private float[] box;
    private float confidence;
    private int classId;
    private String className;
    private float modelWeight;

    Detection(float[] box, float confidence, int classId, String className) {
      this.box = box;
      this.confidence = confidence;
      this.classId = classId;
      this.className = className;
      this.modelWeight = 1.0f;
    }

    Detection(float[] box, float confidence, int classId, String className, float modelWeight) {
      this.box = box;
      this.confidence = confidence;
      this.classId = classId;
      this.className = className;
      this.modelWeight = modelWeight;
    }

    public String toString() {
      return "Detection {class=" + className + ", confidence=" + confidence + '}';
    }
  }

  private class InterpreterWrapper {
    private final String TAG = this.getClass().getSimpleName();
    private Interpreter interpreter;
    private List<String> labels;
    private float modelWeight;
    private float confThreshold;

    InterpreterWrapper(ModelType modelType) {
      this.modelWeight = 1.0f;
      this.confThreshold = 0.4f;

      // Model
      interpreter = modelMap.get(modelType);
      if (interpreter == null) {
        Log.e(TAG, "Failed to load model" + modelType);
      }

      // Labels
      try {
        labels = FileUtil.loadLabels(context, "labels.txt");
      } catch (IOException e) {
        Log.e(TAG, "Failed to load label", e);
      }
    }

    InterpreterWrapper(ModelType modelType, float modelWeight, float confThreshold) {
      this.modelWeight = modelWeight;
      this.confThreshold = confThreshold;

      // Model
      interpreter = modelMap.get(modelType);
      if (interpreter == null) {
        Log.e(TAG, "Failed to load model" + modelType);
      }

      // Labels
      try {
        labels = FileUtil.loadLabels(context, "labels.txt");
      } catch (IOException e) {
        Log.e(TAG, "Failed to load label", e);
      }
    }
  }

  /**
   * Constructor for ItemDetector.
   * Sets up the model, label mappings, and image processor.
   * 
   * @param apiRef  reference to the KiboRpcApi
   * @param context the application context
   */
  public ItemDetector(KiboRpcApi apiRef, Context context) {
    this.api = apiRef;
    this.context = context;

    rand = new Random();
    modelMap = new HashMap<>();

    // Load model
    Interpreter nModel = loadInterpreter("n_20000_0519.tflite");
    if (nModel != null) {
      modelMap.put(ModelType.n_20000_0519, nModel);
    } else {
      Log.e(TAG, "Failed to load n_20000_0519 model");
    }

    // Load model
    Interpreter sModel = loadInterpreter("s_15000_0516.tflite");
    if (sModel != null) {
      modelMap.put(ModelType.s_15000_0516, sModel);
    } else {
      Log.e(TAG, "Failed to load s_15000_0516 model");
    }

    // Labels
    try {
      labels = FileUtil.loadLabels(context, "labels.txt");
    } catch (Exception e) {
      Log.e(TAG, "Failed to load label", e);
    }

    // Load image processor
    imageProcessor = new ImageProcessor.Builder()
      .add(new NormalizeOp(0.0F, 255.0F))
      .add(new CastOp(DataType.FLOAT32))
      .build();

    Log.i(TAG, "Initialized");
  }

  /**
   * Detect using multiple models with revised weighted box fusion method
   */
  public List<Detection> detect(Mat image) {
    List<Detection> detectionList = new ArrayList<>();

    // Loop through each model
    for (ModelType modelType : modelMap.keySet()) {
      // Init wrapper
      InterpreterWrapper interpreterWrapper = new InterpreterWrapper(modelType);
      
      // Preprocess the image
      ByteBuffer imageBuffer = preprocess(image, interpreterWrapper);

      // Run inference
      float[] outputArray = runReference(imageBuffer, interpreterWrapper);

      // Postprocess the output
      List<Detection> detections = postprocess(outputArray, interpreterWrapper);

      detectionList.addAll(detections);
    }
    
    // Apply revised Weighted Box Fusion
    float iouThreshold = 0.7f;
    float confidenceThreshold = 0.5f;
    List<Detection> results = wbf(detectionList, iouThreshold, confidenceThreshold);

    Log.i(TAG, "Detection results:");
    for (result : results) {
      Log.i(TAG, result.toString());
    }

    return results;
  }

  public List<Detection> detect(Mat image, ModelType modelType) {
    // Initialize wrapper
    InterpreterWrapper interpreterWrapper = new InterpreterWrapper(modelType);

    // Preprocess the image
    ByteBuffer imageBuffer = preprocess(image, interpreterWrapper);

    // Run inference
    float[] outputArray = runReference(imageBuffer, interpreterWrapper);

    // Postprocess the output
    List<Detection> detections = postprocess(outputArray, interpreterWrapper);

    // Apply Non-Maximum Suppression
    float iouThreshold = 0.7f;
    List<Detection> results = nms(detections, iouThreshold);

    Log.i(TAG, "Detection results:");
    for (Detection result : results) {
      Log.i(TAG, result.toString());
    }

    return results;
  }

  /**
   * Filters the detection results to identify the treasure and landmark items with the highest confidence.
   *
   * @param detectResult A list of detection results, each containing item details and confidence.
   * @param areaId       The area identifier where the detection occurred.
   * @param tagPose      The pose associated with the detection area.
   * @return An array containing the selected treasure and landmark items, in the format of [treasureItem, landmarkItem].
   */
  public List<Item> filterResult(List<Detection> detectionList, int areaId, Pose tagPose) {
    int treasureId = -1;
    int landmarkId = -1;
    String treasureName = null;
    String landmarkName = null;
    float treasureMaxConfidence = -1.0f;
    float landmarkMaxConfidence = -1.0f;
    int treasureCount = 0;
    int landmarkCount = 0;

    Map<Integer, Integer> itemCountMap = new HashMap<>();
    List<Item> itemList = new ArrayList<>();

    if (detectionList.isEmpty()) {
      Log.w(TAG, "No detection found, return empty list.");
      return itemList;
    }

    for (Detection det: detectionList) {
      String itemName = det.className;

      // Map classId to itemId
      int itemId = -1;
      if (det.classId >= 0 && det.classId < 3) {
        itemId = det.classId + 11;
      } else if (det.classId >= 3 && det.classId < 11) {
        itemId = det.classId + 18;
      }

      // Update item count
      itemCountMap.put(itemId, itemCountMap.getOrDefault(itemId, 0) + 1);

      // Record the treasure with the highest confidence
      if (itemId / 10 == 1) {
        if (det.confidence > treasureMaxConfidence) {
          treasureMaxConfidence = det.confidence;
          treasureId = itemId;
          treasureName = itemName;
        }
      }

      // Record the landmark with the highest confidence
      else if (itemId / 10 == 2) {
        if (det.confidence > landmarkMaxConfidence) {
          landmarkMaxConfidence = det.confidence;
          landmarkId = itemId;
          landmarkName = itemName;
        }
      } else {
        Log.w(TAG, "Unknown item ID: " + itemId); // Log invalid IDs
      }
    }

    // Create the treasure item
    if (treasureId == -1) {
      itemList.add(new Item());
    } else {
      treasureCount = itemCountMap.getOrDefault(treasureId, 1);
      itemList.add(new Item(areaId, treasureId, treasureName, treasureCount, tagPose));
    }

    // Create landmark item
    if (landmarkId == -1) {
      itemList.add(new Item());
    } else {
      landmarkCount = itemCountMap.getOrDefault(landmarkId, 1);
      itemList.add(new Item(areaId, landmarkId, landmarkName, landmarkCount, tagPose));
    }

    return itemList;
  }

  /**
   * Guess the result.
   *
   * @param areaId   The area identifier where the detection occurred.
   * @param tagPose  The pose associated with the detection area.
   * @return An array containing the selected treasure and landmark items, in the format of [treasureItem, landmarkItem].
   */
  public List<Item> guessResult(int areaId, Pose tagPose) {
    List<Item> results = new ArrayList<>();

    int treasureId = rand.nextInt(3);
    int landmarkId = rand.nextInt(8) + 3; 
    String treasureName = labels.get(treasureId);
    String landmarkName = labels.get(landmarkId);

    int treasureCount = 1;
    int landmarkCount = rand.nextInt(4) + 1; // Random count (1-3)

    results.add(new Item(areaId, treasureId, treasureName, treasureCount, tagPose));
    results.add(new Item(areaId, landmarkId, landmarkName, landmarkCount, tagPose));

    return results;
  }

  /**
   * Draws bounding boxes on the given image based on the detection results.
   *
   * @param image The input image to draw on.
   * @param detectResult   List of detection results containing bounding box coordinates.
   * @param area           The area identifier used for saving the image.
   */
  public void drawBoundingBoxes(Mat image, List<Detection> detectionList, int area) {
    if (detectionList == null || detectionList.isEmpty()) {
      Log.i(TAG, "No detections to draw.");
      return;
    }

    int imageWidth = image.cols();
    int imageHeight = image.rows();

    int inputWidth = 640;
    int inputHeight = 640;

    float scale = Math.min((float) inputWidth / imageWidth, (float) inputHeight / imageHeight);
    int newWidth = Math.round(imageWidth * scale);
    int newHeight = Math.round(imageHeight * scale);

    int padX = (inputWidth - newWidth) / 2;
    int padY = (inputHeight - newHeight) / 2;

    for (Detection det : detectionList) {
      float x1 = det.box[0] * inputWidth;
      float y1 = det.box[1] * inputHeight;
      float x2 = det.box[2] * inputWidth;
      float y2 = det.box[3] * inputHeight;

      x1 = (x1 - padX) / scale;
      y1 = (y1 - padY) / scale;
      x2 = (x2 - padX) / scale;
      y2 = (y2 - padY) / scale;

      int ix1 = Math.max(0, Math.min((int) x1, imageWidth - 1));
      int iy1 = Math.max(0, Math.min((int) y1, imageHeight - 1));
      int ix2 = Math.max(0, Math.min((int) x2, imageWidth - 1));
      int iy2 = Math.max(0, Math.min((int) y2, imageHeight - 1));

      // Create a rectangle from the top-left and bottom-right points
      Rect rect = new Rect(new Point(ix1, iy1), new Point(ix2, iy2));
      
      // Define rectangle color (red) and thickness
      Scalar color = new Scalar(0, 0, 0); // Black
      int thickness = 1;

      // Draw the rectangle on the image
      Imgproc.rectangle(image, rect, color, thickness);

      String label = labels.get((int) det.classId);
      Scalar textColor = new Scalar(0, 0, 0);
      int font = Imgproc.FONT_HERSHEY_SIMPLEX;
      double fontScale = 0.5;
      int thicknessText = 1;

      Point labelPosition = new Point(ix1, iy1 - 5); // Position slightly above the bounding box
      Imgproc.putText(image, label, labelPosition, font, fontScale, textColor, thicknessText);
    }

    api.saveMatImage(image, String.format("area%d_bbox.png", area));
  }

  /* ------------------------- Private Tool Functions ------------------------- */

  /**
   * Loads the TensorFlow Lite model from the assets folder.
   *
   * @param model_path The path to the model file in the assets folder.
   * @return An Interpreter object for running inference on the model.
   */
  private Interpreter loadInterpreter(String model_path) {
    Log.i(TAG, "Loading interpreter for " + model_path);
    FileInputStream inputStream = null;
    FileChannel fileChannel = null;

    try {
      // Load the model file from assets
      AssetFileDescriptor fileDescriptor = context.getAssets().openFd(model_path);
      inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
      fileChannel = inputStream.getChannel();
      
      long startOffset = fileDescriptor.getStartOffset();
      long declaredLength = fileDescriptor.getDeclaredLength();
      Log.i(TAG, "Model size: " + declaredLength / 1024.0 / 1024.0 + " MB");

      MappedByteBuffer model = fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
      Interpreter.Options options = new Interpreter.Options();
      options.setNumThreads(2);

      Interpreter interpreter = new Interpreter(model, options);

      return interpreter;
    } catch (Exception e) {
      Log.e(TAG, "Fail to load model at " + model_path + ": " + e.getMessage(), e);
      return null;
    } finally {
      if (fileChannel != null) {
        try {
          fileChannel.close();
        } catch (Exception e) {
          Log.e(TAG, "Error closing file channel", e);
        }
      }

      if (inputStream != null) {
        try {
          inputStream.close();
        } catch (Exception e) {
          Log.e(TAG, "Error closing input stream", e);
        }
      }
    }
  }

  /**
   * Preprocesses the input image by resizing and normalizing it.
   *
   * @param image The input image as a Mat object.
   * @return A ByteBuffer ready for model input.
   */
  private ByteBuffer preprocess(Mat image, InterpreterWrapper interpreterWrapper) {
    int[] inputShape = interpreterWrapper.interpreter.getInputTensor(0).shape();
    int tensorWidth = inputShape[1];
    int tensorHeight = inputShape[2];

    // Convert Mat to Bitmap format
    Bitmap bitmap = Bitmap.createBitmap(image.cols(), image.rows(), Bitmap.Config.ARGB_8888);
    Utils.matToBitmap(image, bitmap);

    // Resize Bitmap to match the model input size
    Bitmap resizedBitmap = letterbox(bitmap, tensorWidth, tensorHeight);

    // Load the resized Bitmap into a TensorImage
    TensorImage tensorImage = new TensorImage(DataType.FLOAT32);
    tensorImage.load(resizedBitmap);

    // Apply image preprocessing (normalization, type casting, etc.)
    TensorImage processedImage = imageProcessor.process(tensorImage);

    // Get the ByteBuffer from the processed image
    ByteBuffer imageBuffer = processedImage.getBuffer();

    return imageBuffer;
  }

  private Bitmap letterbox(Bitmap src, int targetWidth, int targetHeight) {
    int srcWidth = src.getWidth();
    int srcHeight = src.getHeight();

    float scale = Math.min((float) targetWidth / srcWidth, (float) targetHeight / srcHeight);
    int newWidth = Math.round(srcWidth * scale);
    int newHeight = Math.round(srcHeight * scale);

    // Resize the original image
    Bitmap resized = Bitmap.createScaledBitmap(src, newWidth, newHeight, true);

    // Create a new bitmap with gray background
    Bitmap letterboxed = Bitmap.createBitmap(targetWidth, targetHeight, Bitmap.Config.ARGB_8888);
    Canvas canvas = new Canvas(letterboxed);
    Paint paint = new Paint();
    paint.setColor(Color.rgb(114, 114, 114)); // gray background
    canvas.drawRect(0, 0, targetWidth, targetHeight, paint);

    // Paste the resized image in the center
    int dx = (targetWidth - newWidth) / 2;
    int dy = (targetHeight - newHeight) / 2;
    canvas.drawBitmap(resized, dx, dy, null);

    return letterboxed;
  }

  /**
   * Runs the TensorFlow Lite model inference on the input image buffer.
   *
   * @param imageBuffer The preprocessed input image as a ByteBuffer.
   * @return The model output as a float array.
   */
  private float[] runReference(ByteBuffer imageBuffer, InterpreterWrapper interpreterWrapper) {
    // Get output tensor shapes
    int[] outputShape = interpreterWrapper.interpreter.getOutputTensor(0).shape();
    int numChannel = outputShape[1];
    int numElements = outputShape[2];

    // Create an output buffer with the expected shape and type
    TensorBuffer output = TensorBuffer.createFixedSize(new int[]{1, numChannel, numElements}, DataType.FLOAT32);

    // Run inference with the interpreter
    interpreterWrapper.interpreter.run(imageBuffer, output.getBuffer());

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
  private List<Detection> postprocess(float[] outputArray, InterpreterWrapper interpreterWrapper) {
    // Get output tensor shapes
    int[] outputShape = interpreterWrapper.interpreter.getOutputTensor(0).shape();
    int numChannel = outputShape[1];
    int numElements = outputShape[2];

    // Loop through each detection
    List<Detection> detections = new ArrayList<>();
    for (int i = 0; i < numElements; i++) {
      // Find the class with the highest confidence for this candidate
      float maxConf = -1.0F;
      int maxIdx = -1;
      for (int j = 4; j < numChannel; j++) {
        if (outputArray[i + numElements * j] > maxConf) {
          maxConf = outputArray[i + numElements * j];
          maxIdx = j - 4;
        }
      }

      if (maxConf < interpreterWrapper.confThreshold) continue;

      // Get center coordinates and size
      float cx = outputArray[i];
      float cy = outputArray[i + numElements];
      float w = outputArray[i + numElements * 2];
      float h = outputArray[i + numElements * 3];

      // Calculate bounding box corners, staying within [0, 1] range
      float x1 = cx - w / 2.0F;
      float y1 = cy - h / 2.0F;
      float x2 = cx + w / 2.0F;
      float y2 = cy + h / 2.0F;

      // Create a Detection object
      Detection detection = new Detection(
        new float[]{x1, y1, x2, y2},
        maxConf,
        maxIdx,
        interpreterWrapper.labels.get(maxIdx),
        interpreterWrapper.modelWeight
      );
      detections.add(detection);
    }

    return detections;
  }

  private List<Detection> wbf(List<Detection> detections, float iouThreshold, float confThreshold) {
    if (detections.isEmpty()) {
      Log.i(TAG, "No detections to process.");
      return detections;
    }

    // Sort the score in descending order
    Collections.sort(detections, new Comparator<Detection>() {
      @Override
      public int compare(Detection d1, Detection d2) {
        return Float.compare(d2.confidence, d1.confidence); 
      }
    });

    List<Detection> fused = new ArrayList<>();
    boolean[] used = new boolean[detections.size()];

    for (int i = 0; i < detections.size(); i++) {
      if (used[i]) continue;

      List<Detection> group = new ArrayList<>();
      group.add(detections.get(i));
      used[i] = true;

      for (int j = i + 1; j < detections.size(); j++) {
        if (used[j]) continue;

        Detection di = detections.get(i);
        Detection dj = detections.get(j);
        float iou = calculateIoU(di.box, dj.box);
        boolean contained = isContained(dj.box, di.box, 0.9f);

        if (iou > iouThreshold || contained) {
          group.add(dj);
          used[j] = true;
        }
      }

      // Compute weighted box
      float confidenceSum = 0f;
      float weightSum = 0f;
      float x1 = 0f, y1 = 0f, x2 = 0f, y2 = 0f;

      for (Detection d : group) {
        confidenceSum += d.confidence * d.modelWeight;
        weightSum += d.modelWeight;
        x1 += d.box[0] * d.confidence * d.modelWeight;
        y1 += d.box[1] * d.confidence * d.modelWeight;
        x2 += d.box[2] * d.confidence * d.modelWeight;
        y2 += d.box[3] * d.confidence * d.modelWeight;
      }

      float confidenceAvg = confidenceSum / weightSum;
      if (confidenceAvg < confThreshold) {
        continue;
      }

      x1 /= confidenceSum;
      y1 /= confidenceSum;
      x2 /= confidenceSum;
      y2 /= confidenceSum;

      float minX = Math.min(x1, x2);
      float minY = Math.min(y1, y2);
      float maxX = Math.max(x1, x2);
      float maxY = Math.max(y1, y2);

      // Check if the bounding box is too small
      if ((maxX - minX) < 0.05 || (maxY - minY) < 0.05) {
        continue;
      }

      // Sort into descending order of confidence
      Collections.sort(group, new Comparator<Detection>() {
        @Override
        public int compare(Detection a, Detection b) {
          float weightedConfidenceA = a.confidence * a.modelWeight;
          float weightedConfidenceB = b.confidence * b.modelWeight;
          return Float.compare(weightedConfidenceB, weightedConfidenceA);
        }
      });

      Detection fusedDetection = new Detection(
        new float[]{minX, minY, maxX, maxY},
        confidenceAvg,
        group.get(0).classId,
        group.get(0).className
      );
      fused.add(fusedDetection);
    }

    return fused;
  }

  /**
   * Applies Non-Maximum Suppression (NMS) to remove overlapping detections.
   *
   * @param detections List of raw detections [x1, y1, x2, y2, ..., confidence, class]
   * @return List of filtered detections after NMS
   */
  private List<Detection> nms(List<Detection> detections, float iouThreshold) {
    List<Detection> results = new ArrayList<>();

    if (detections.isEmpty()) {
      Log.i(TAG, "No detections to process.");
      return results;
    }

    // Sort detections by confidence in descending order
    detections.sort(new Comparator<Detection>() {
      @Override
      public int compare(Detection det1, Detection det2) {
        return Float.compare(det2.confidence, det1.confidence);
      }
    });

    // Perform NMS
    while (!detections.isEmpty()) {
      // Pick the detection with highest confidence
      Detection first = detections.remove(0);
      results.add(first);

      // Remove overlapping detections
      Iterator<Detection> iterator = detections.iterator();
      while (iterator.hasNext()) {
        Detection next = iterator.next();
        if (calculateIoU(first.box, next.box) >= iouThreshold) {
          iterator.remove();
        }
      }
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

  /**
   * Checks if the inner bounding box is contained within the outer bounding box.
   * The function uses a threshold to determine if the inner box is sufficiently contained.
   * 
   * @param inner The inner bounding box [x, y, width, height].
   * @param outer The outer bounding box [x, y, width, height].
   * @param threshold The threshold for containment (0.0 to 1.0).
   * @return true if the inner box is contained within the outer box, false otherwise.
   */
  private boolean isContained(float[] inner, float[] outer, float threshold) {
    float xi = inner[0], yi = inner[1], wi = inner[2], hi = inner[3];
    float xo = outer[0], yo = outer[1], wo = outer[2], ho = outer[3];

    float xi2 = xi + wi, yi2 = yi + hi;
    float xo2 = xo + wo, yo2 = yo + ho;

    float interX1 = Math.max(xi, xo);
    float interY1 = Math.max(yi, yo);
    float interX2 = Math.min(xi2, xo2);
    float interY2 = Math.min(yi2, yo2);

    float interWidth = Math.max(0, interX2 - interX1);
    float interHeight = Math.max(0, interY2 - interY1);
    float interArea = interWidth * interHeight;
    float innerArea = wi * hi;

    return interArea / innerArea > threshold;
  }
}
