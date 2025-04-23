package jp.jaxa.iss.kibo.rpc.sampleapk;

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
import java.util.List;

import jp.jaxa.iss.kibo.rpc.api.KiboRpcApi;

/**
 * Class to detect the item from provided image using YOLO TFLite model
 *
 * Usage:
 *   ItemDetector detector = new ItemDetector(context, api);
 *   List<ItemDetector.DetectionResult> results = detector.detect(bitmap);
 *   for (DetectionResult r : results) {
 *       r.label      // 類別名稱（如 "key"）
 *       r.confidence // 預測信心分數
 *       r.bbox       // 框框位置（RectF）
 *   }
 */
public class ItemDetector {
    private final KiboRpcApi api;
    private final Context context;
    private final String TAG = this.getClass().getSimpleName();
    private Interpreter tflite;
    private List<String> labels;
    private static final float CONFIDENCE_THRESHOLD = 0.5f;  // 最低信心門檻

    // 建構子：初始化 API、Context 並載入模型與標籤
    public ItemDetector(Context context, KiboRpcApi apiRef) {
        this.api = apiRef;
        this.context = context;
        Log.i(TAG, "Initialized");
        try {
            setupModel();
        } catch (IOException e) {
            Log.e(TAG, "Failed to load model or labels", e);
        }
    }

    // 載入 TFLite 模型與 labels.txt（放在 assets/）
    private void setupModel() throws IOException {
        MappedByteBuffer modelBuffer = FileUtil.loadMappedFile(context, "best.tflite");
        tflite = new Interpreter(modelBuffer);
        labels = FileUtil.loadLabels(context, "labels.txt");
    }

    // 傳入一張 Bitmap，回傳偵測結果（包含類別、信心分數與框）
    public List<DetectionResult> detect(Bitmap bitmap) {
        if (tflite == null) return null;

        // 圖片縮放為模型所需尺寸（預設 YOLO 輸入 640x640）
        Bitmap resized = Bitmap.createScaledBitmap(bitmap, 640, 640, true);
        TensorImage tensorImage = TensorImage.fromBitmap(resized);

        // 建立輸出 Tensor buffer
        float[][][] output = new float[1][8400][labels.size() + 5];
        tflite.run(tensorImage.getBuffer(), output);

        // 後處理：從模型輸出中篩選出有用的預測
        return postprocess(output[0], bitmap.getWidth(), bitmap.getHeight());
    }

    // 從模型輸出中解出結果並轉換為實際圖片座標
    private List<DetectionResult> postprocess(float[][] preds, int origW, int origH) {
        List<DetectionResult> results = new ArrayList<>();
        for (float[] pred : preds) {
            float conf = pred[4];
            if (conf > CONFIDENCE_THRESHOLD) {
                // 找出預測中機率最高的類別
                float maxProb = -1f;
                int classId = -1;
                for (int i = 5; i < pred.length; i++) {
                    if (pred[i] > maxProb) {
                        maxProb = pred[i];
                        classId = i - 5;
                    }
                }

                if (maxProb > CONFIDENCE_THRESHOLD) {
                    // 從 cx, cy, w, h 算出 bbox 座標，轉換回原圖比例
                    float cx = pred[0], cy = pred[1], w = pred[2], h = pred[3];
                    float left = (cx - w / 2) * origW / 640;
                    float top = (cy - h / 2) * origH / 640;
                    float right = (cx + w / 2) * origW / 640;
                    float bottom = (cy + h / 2) * origH / 640;
                    String label = labels.get(classId);
                    Log.i(TAG, "Detected: " + label +" | Confidence: " + maxProb +" | BBox: [" + left + ", " + top + ", " + right + ", " + bottom + "]");

                    results.add(new DetectionResult(label, maxProb, new RectF(left, top, right, bottom)));
                }
            }
        }
        return results;
    }

    // 偵測結果的結構：label 名稱、信心分數、框座標
    public static class DetectionResult {
        public final String label;
        public final float confidence;
        public final RectF bbox;

        public DetectionResult(String label, float confidence, RectF bbox) {
            this.label = label;
            this.confidence = confidence;
            this.bbox = bbox;
        }
    }
}
