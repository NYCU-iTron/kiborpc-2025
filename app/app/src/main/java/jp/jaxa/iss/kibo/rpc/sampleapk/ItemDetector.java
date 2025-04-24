package jp.jaxa.iss.kibo.rpc.sampleapk;

import jp.jaxa.iss.kibo.rpc.api.KiboRpcApi;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.RectF;
import android.util.Log;

import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.support.common.FileUtil;
import org.tensorflow.lite.support.image.TensorImage;

import java.io.IOException;
import java.nio.MappedByteBuffer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.HashMap;
import java.util.Map;
import java.util.Collections;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;

import org.opencv.android.Utils;
import org.opencv.core.Mat;


/**
 * Class to detect the item from provided image using YOLO TFLite model
 */
public class ItemDetector {
  private final KiboRpcApi api;
  private final Context context;
  private final String TAG = this.getClass().getSimpleName();

  private Interpreter tflite;
  private List<String> labels;
  private static final float CONFIDENCE_THRESHOLD = 0.5f;

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
   * Setup the TensorFlow Lite model.
   */
  private void setupModel() throws IOException {
    MappedByteBuffer modelBuffer = FileUtil.loadMappedFile(context, "best.tflite");
    tflite = new Interpreter(modelBuffer);
    labels = FileUtil.loadLabels(context, "labels.txt");
  }

  /**
   * Detects items in the given undistorted image using a TensorFlow Lite model.
   *
   * @param undistortImage The input image as a Mat object, which has been undistorted.
   * @return A list of detected items, or an empty list if the TensorFlow Lite model is not initialized.
   */
  public List<Item> detect(Mat undistortImage) {
    if (tflite == null) return Collections.emptyList();

    // Convert Mat to Bitmap
    Bitmap bitmap = Bitmap.createBitmap(undistortImage.cols(), undistortImage.rows(), Bitmap.Config.ARGB_8888);
    Utils.matToBitmap(undistortImage, bitmap);

    // Get input tensor shape from model (e.g., [1, height, width, 3])
    int[] inputShape = tflite.getInputTensor(0).shape();
    int height = inputShape[1]; // 640
    int width = inputShape[2]; // 640

    // Resize bitmap
    Bitmap resized = Bitmap.createScaledBitmap(bitmap, width, height, true);
    TensorImage tensorImage = TensorImage.fromBitmap(resized);

    // Create input buffer
    ByteBuffer inputBuffer = ByteBuffer.allocateDirect(1 * width * height * 3 * 4); // FLOAT32
    inputBuffer.order(ByteOrder.nativeOrder());
    int[] intValues = new int[width * height];
    resized.getPixels(intValues, 0, width, 0, 0, width, height);

    // Normalize pixel values to [0, 1] and add to input buffer
    for (int pixel : intValues) {
      inputBuffer.putFloat(((pixel >> 16) & 0xFF) / 255.0f); // R
      inputBuffer.putFloat(((pixel >> 8) & 0xFF) / 255.0f);  // G
      inputBuffer.putFloat((pixel & 0xFF) / 255.0f);         // B
    }

    // Dynamic Output
    int[] outputShape = tflite.getOutputTensor(0).shape(); // [1 15 8400]
    float[][][] output = new float[outputShape[0]][outputShape[1]][outputShape[2]];
        
    // Feed to model
    tflite.run(inputBuffer, output);

    for (int i = 0; i < output[0].length; i++) {
      Log.i(TAG, "Output[" + i + "]: " + Arrays.toString(output[0][i]));
    }
    return postprocess(output[0], bitmap.getWidth(), bitmap.getHeight(), width, height);
  }

  /**
   * Processes the predictions from the TensorFlow Lite model to extract detected items.
   *
   * @param preds A 2D array of predictions from the model. Each prediction contains:
   *              [center_x, center_y, width, height, confidence, class_probabilities...].
   * @param origW The width of the original image.
   * @param origH The height of the original image.
   * @return A list of detected items
   */
  private List<Item> postprocess(float[][] preds, int origW, int origH, int inputW, int inputH) {
    List<Item> results = new ArrayList<>();
    Map<String, Integer> itemCountMap = new HashMap<>();
    int itemId = 0;

    int numFeatures = preds.length;        // 15
    int numPredictions = preds[0].length;  // 8400

    for (int i = 0; i < numPredictions; i++) {
      float cx = preds[0][i];
      float cy = preds[1][i];
      float w  = preds[2][i];
      float h  = preds[3][i];
      float conf = preds[4][i];

      if (conf > CONFIDENCE_THRESHOLD) {
        float maxProb = -1f;
        int classId = -1;

        for (int j = 5; j < numFeatures; j++) {
          if (preds[j][i] > maxProb) {
            maxProb = preds[j][i];
            classId = j - 5;
          }
        }

        if (maxProb > CONFIDENCE_THRESHOLD) {
          float left = (cx - w / 2) * origW / inputW;
          float top = (cy - h / 2) * origH / inputH;
          float right = (cx + w / 2) * origW / inputW;
          float bottom = (cy + h / 2) * origH / inputH;

          String label = labels.get(classId);
          itemCountMap.put(label, itemCountMap.getOrDefault(label, 0) + 1);

          Log.i(TAG, String.format("Detected item: %s, Confidence: %.2f, BBox: [%.1f, %.1f, %.1f, %.1f]",
            label, maxProb, left, top, right, bottom));

          // Item item = new Item(areaId, itemId++, label, itemCountMap.get(label), itemPose);
          // results.add(item);
        }
      }
    }

    return results;
  }
}
