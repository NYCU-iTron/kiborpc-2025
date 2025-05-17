from ultralytics import YOLO
import cv2
from pathlib import Path
import numpy as np
import time
from tqdm import tqdm

def iou(box1, box2):
    """計算兩個框的 IoU"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter_area
    return inter_area / union if union > 0 else 0

def weighted_boxes_fusion(
    boxes_list, scores_list, labels_list, weights=None,
    wbf_iou_threshould=0.3, wbf_score_threshould=0.1
):
    """
    boxes_list: List of [ [ [x1,y1,x2,y2], ... ], [..], .. ] for each model
    scores_list: List of [ [score1, score2, ...], [...], ... ] for each model
    labels_list: List of [ [label1, label2, ...], [...], ... ] for each model
    weights: List of float weights per model
    Returns: fused_boxes, fused_scores, fused_labels
    """
    if weights is None:
        weights = [1.0] * len(boxes_list)

    all_boxes = []
    all_scores = []
    all_labels = []
    all_model_indices = []  # 追蹤每個框來自哪個模型

    for i, (boxes, scores, labels, w) in enumerate(zip(boxes_list, scores_list, labels_list, weights)):
        for box, score, label in zip(boxes, scores, labels):
            if score >= wbf_score_threshould:
                all_boxes.append(box)
                all_scores.append(score * w)
                all_labels.append(label)
                all_model_indices.append(i)

    if not all_boxes:
        return [], [], []

    all_boxes = np.array(all_boxes)
    all_scores = np.array(all_scores)
    all_labels = np.array(all_labels)
    all_model_indices = np.array(all_model_indices)

    used = [False] * len(all_boxes)
    fused_boxes = []
    fused_scores = []
    fused_labels = []
    model_contributions = []  # 記錄每個最終框由哪些模型貢獻

    for i in range(len(all_boxes)):
        if used[i]:
            continue
        box_group = [all_boxes[i]]
        score_group = [all_scores[i]]
        model_group = [all_model_indices[i]]  # 記錄該組框的模型來源
        used[i] = True
        label = all_labels[i]

        for j in range(i + 1, len(all_boxes)):
            if used[j]:
                continue
            if all_labels[j] != label:
                continue
            if iou(all_boxes[i], all_boxes[j]) > wbf_iou_threshould:
                box_group.append(all_boxes[j])
                score_group.append(all_scores[j])
                model_group.append(all_model_indices[j])  # 添加模型索引
                used[j] = True

        box_group = np.array(box_group)
        score_group = np.array(score_group)
        total_score = np.sum(score_group)

        weighted_box = np.sum(box_group.T * score_group, axis=1) / total_score
        fused_boxes.append(weighted_box.tolist())
        fused_scores.append(float(np.max(score_group)))  # 使用最大分數
        fused_labels.append(int(label))
        model_contributions.append(list(set(model_group)))  # 記錄該框由哪些模型貢獻

    return fused_boxes, fused_scores, fused_labels, model_contributions

def normalize_boxes(boxes, w, h):
    """將框座標轉換為相對座標 (0-1)"""
    return [[x1/w, y1/h, x2/w, y2/h] for x1, y1, x2, y2 in boxes]

def denormalize_boxes(boxes, w, h):
    """將相對座標 (0-1) 轉換回絕對座標"""
    return [[int(x1*w), int(y1*h), int(x2*w), int(y2*h)] for x1, y1, x2, y2 in boxes]

def post_process_detections(boxes, scores, labels, min_area=100, max_area=None, min_aspect_ratio=0.1, max_aspect_ratio=10):
    """後處理過濾檢測結果"""
    filtered_boxes, filtered_scores, filtered_labels = [], [], []
    
    for box, score, label in zip(boxes, scores, labels):
        x1, y1, x2, y2 = box
        width = x2 - x1
        height = y2 - y1
        area = width * height
        aspect_ratio = width / height if height > 0 else 0
        
        # 過濾太小或太大的框
        if area < min_area:
            continue
        if max_area and area > max_area:
            continue
        
        # 過濾異常長寬比的框
        if aspect_ratio < min_aspect_ratio or aspect_ratio > max_aspect_ratio:
            continue
            
        filtered_boxes.append(box)
        filtered_scores.append(score)
        filtered_labels.append(label)
        
    return filtered_boxes, filtered_scores, filtered_labels

color_palette = np.random.uniform(128, 255, size=(11, 3))
def process_batch(models, image_paths, test_results_dir, model_weights, class_names, config):
    """批次處理多張圖片"""
    for image_path in image_paths:
        img = cv2.imread(str(image_path))
        if img is None:
            print(f"⚠️ 無法讀取：{image_path.name}")
            continue
        h, w = img.shape[:2]

        all_boxes = []
        all_scores = []
        all_labels = []

        # 對每個模型做推論
        for model_idx, model in enumerate(models):
            # 調整 NMS 閾值使其能保留更多重疊框
            results = model.predict(
                source=img, 
                conf=config['model_conf'], 
                iou=config['model_iou'], 
                verbose=False
            )
            
            for r in results:
                boxes = []
                scores = []
                labels = []
                for box in r.boxes:
                    x1, y1, x2, y2 = map(float, box.xyxy[0])
                    conf = float(box.conf)
                    cls_id = int(box.cls)
                    boxes.append([x1, y1, x2, y2])
                    scores.append(conf)
                    labels.append(cls_id)

            all_boxes.append(normalize_boxes(boxes, w, h))
            all_scores.append(scores)
            all_labels.append(labels)

        # 使用 Weighted Box Fusion 融合所有模型輸出
        boxes_fused, scores_fused, labels_fused, model_contribs = weighted_boxes_fusion(
            all_boxes, all_scores, all_labels,
            weights=model_weights,
            wbf_iou_threshould=config['wbf_iou_threshould'],
            wbf_score_threshould=config['wbf_score_threshould']
        )

        # 還原座標
        boxes_final = denormalize_boxes(boxes_fused, w, h)
        
        # 後處理過濾
        boxes_final, scores_fused, labels_fused = post_process_detections(
            boxes_final, scores_fused, labels_fused,
            min_area=config['min_area'],
            max_area=config['max_area']
        )

        # 建立結果副本
        result_img = img.copy()
        
        # 畫框
        for (x1, y1, x2, y2), score, cls_id in zip(boxes_final, scores_fused, labels_fused):
            cls_id = int(cls_id)
            color = color_palette[cls_id % len(color_palette)]
            label = f"{class_names[cls_id]} {score:.2f}"
            cv2.rectangle(result_img, (x1, y1), (x2, y2), color, 2)
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(result_img, (x1, y1 - th - 5), (x1 + tw, y1), color, -1)
            cv2.putText(result_img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        output_path = test_results_dir / image_path.name
        cv2.imwrite(str(output_path), result_img)

    return len(image_paths)

def main():
    # === 檔案與資料夾 ===
    base_dir = Path(__file__).resolve().parent
    test_set_dir = (base_dir / '../assets/test_set').resolve()
    test_results_dir = (base_dir / '../assets/test_result').resolve()
    model_paths = [
        (base_dir / '../assets/test_model/s_15000_0516.pt').resolve(),
        (base_dir / '../assets/test_model/n_2000_0509.pt').resolve(),
    ]

    config = {
        'model_conf': 0.1,              # 模型的初始信心閾值
        'model_iou': 0.05,              # 模型的NMS IoU閾值 (較低值有助於保留重疊框)
        'wbf_iou_threshould': 0.2,      # WBF的IoU閾值 (較低值有助於區分相近物體)
        'wbf_score_threshould': 0.4,    # WBF的最低分數閾值
        'min_area': 70,                 # 過濾太小的框 (像素)
        'max_area': None,               # 過濾太大的框 (像素，None表示不設限)
        'batch_size': 4                 # 批次處理大小
    }

    # 設定模型權重（和 model_paths 順序對應）
    model_weights = [5, 1] # 1 - 10
    model_weights = [w / sum(model_weights) for w in model_weights]

    # 建立結果資料夾
    test_results_dir.mkdir(parents=True, exist_ok=True)

    # 載入多個 YOLO 模型
    print("載入模型中...")
    models = [YOLO(str(path)) for path in model_paths]
    
    # 取得類別名稱
    class_names = models[0].names
    
    # 處理所有圖片
    start_time = time.time()
    image_paths = sorted(test_set_dir.glob("*.[jpJP]*[npNP]*[gG]"))
    print(f"找到 {len(image_paths)} 張圖片")
    
    # 分批處理圖片
    batch_size = config['batch_size']
    image_batches = [image_paths[i:i+batch_size] for i in range(0, len(image_paths), batch_size)]
    
    total_processed = 0
    with tqdm(total=len(image_paths), desc="處理圖片") as pbar:
        for batch in image_batches:
            processed = process_batch(models, batch, test_results_dir, model_weights, class_names, config)
            total_processed += processed
            pbar.update(processed)
    
    end_time = time.time()
    print(f"✅ 總共處理了 {total_processed} 張圖片")
    print(f"⏱️ 總耗時: {end_time - start_time:.2f} 秒")
    print(f"⏱️ 平均每張: {(end_time - start_time) / total_processed:.2f} 秒")

if __name__ == "__main__":
    main()
