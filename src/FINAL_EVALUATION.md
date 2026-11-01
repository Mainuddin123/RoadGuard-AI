# RoadGuard AI — Final Model Evaluation Report

## 1. Project Overview

RoadGuard AI is a road-damage detection system designed to identify three categories:

- Pothole
- Crack
- Manhole

The project uses YOLOv8n object detection and compares a model trained from scratch with a model initialized using pretrained YOLOv8n weights.

## 2. Dataset Summary

| Description | Count |
|---|---:|
| Unique images | 1,999 |
| Total annotated objects | 4,701 |
| Training images | 1,406 |
| Validation images | 292 |
| Test images | 301 |

### Class distribution

| Class | Objects |
|---|---:|
| Pothole | 1,253 |
| Crack | 2,498 |
| Manhole | 950 |
| **Total** | **4,701** |

The dataset was cleaned to remove duplicate images, and a group-aware split was used to avoid sequence-group leakage between training, validation and test sets.

## 3. Model Configuration

| Parameter | Model 1 | Model 2 |
|---|---|---|
| Architecture | YOLOv8n | YOLOv8n |
| Initialization | Random | Pretrained weights |
| Training epochs | 3 | 3 |
| Image size | 416 × 416 | 416 × 416 |
| Batch size | 1 | 1 |
| Device | CPU | CPU |
| Dataset fraction | 100% | 100% |

## 4. Training Performance

| Metric | Model 1: Scratch | Model 2: Pretrained |
|---|---:|---:|
| Training time | 34.3 minutes | 41.35 minutes |
| Model file size | 6.2 MB | 6.5 MB |

## 5. Held-out Test Set Results

Both models were evaluated on the same held-out test set of 301 images containing 709 annotated objects.

| Metric | Model 1: Scratch | Model 2: Pretrained |
|---|---:|---:|
| Precision | 0.0018 | 0.2646 |
| Recall | 0.1859 | 0.2911 |
| mAP@50 | 0.0023 | 0.1790 |
| mAP@50–95 | 0.0004 | 0.0782 |
| Inference time per image | 41.3 ms | 59.9 ms |

### Interpretation

The pretrained model performed substantially better than the scratch model on the held-out test set.

However, its detection performance remains limited. In particular, the crack class was difficult to detect.

Neither model should be considered production-ready based on these results.

## 6. IoU and Evaluation Metrics

Intersection over Union (IoU) measures the overlap between a predicted bounding box and its ground-truth box.

IoU = Area of Intersection / Area of Union

- Precision measures how many predicted detections are correct.
- Recall measures how many actual objects are detected.
- mAP@50 evaluates mean average precision at an IoU threshold of 0.50.
- mAP@50–95 averages performance over IoU thresholds from 0.50 to 0.95.

**Note:** A separate standalone average IoU value was not recorded in the current evaluation. The mAP metrics above are the measured localization and detection metrics.

## 7. Recorded Video Demonstration

Three recorded videos were tested using the pretrained model.

| Video | Frames | Detections | Processing time | FPS |
|---|---:|---:|---:|---:|
| demo1.mp4 | 142 | 4 | 13.56 s | 10.47 |
| demo2.mp4 | 39 | 0 | 4.03 s | 9.67 |
| demo3.mp4 | 21 | 0 | 3.43 s | 6.12 |

The model completed processing of all three videos. However, detections were inconsistent.

## 8. Live Webcam Demonstration

- Webcam access: Successful
- Fullscreen display: Implemented
- Bounding-box visualization: Implemented
- Live FPS display: Implemented
- Observed FPS: Approximately 11–13 FPS
- Confirmed road-hazard detections in the webcam demonstration: None

The webcam test used a mobile screen displaying road footage. This is not equivalent to testing against a direct road-camera feed.

## 9. Final Conclusion

The pretrained YOLOv8n model outperformed the randomly initialized model on the held-out test set.

The project demonstrates dataset preparation, object detection training, model evaluation, recorded-video inference and live webcam integration.

However, the current results reveal limitations in detection accuracy and generalization. Further work is required, including longer training, improved data diversity, annotation review and model optimization.

## 10. Important Architecture Note

Model 2 was initialized from pretrained YOLOv8n detector weights. This transfers pretrained weights across the detector, rather than initializing only a CNN backbone and training a newly initialized detection head.

This distinction should be disclosed when comparing the implementation with an assignment requirement specifically asking for a pretrained backbone and a separate detection head.

