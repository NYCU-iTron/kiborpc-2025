# Week 1

## Meeting Record

- 定義資料儲存格式 (待修改)

```java
public class Treasure {
  private int id;
  private String featureData; 
  private Landmark landmark;
}
```

```java
public class Landmark {
  private Int Id;
  private String name;
  private Point pose;
  private Quaternion orientation;
}
```

- 定義 Class (待修改)
  - Navigator: 移動控制, 路徑演算法
  - LandmarkScanner: 掃描地標, 辨識寶藏
  - TreasureManager: 紀錄寶藏, 比對真寶藏
  - Mission Controller: 決定何時作什麼
- Other
  - Nameing convention: camel case for variables and functions
  - 把yolo模型打包進apk: `.pt model -> .pb model -> .tflite model, use tflite model in android app`
  - 要考慮干擾 相機：光線、移動誤差（?）
  - 多執行序

## Progress

- Complete apk compile using docker, the apk will be generated in `app/app/build/outputs/apk/debug/` folder.

## Next
