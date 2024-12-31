# H100 benchmark 200 FPS

import cv2
import numpy as np
from utilities import Engine
import torch
import timeit
import colorsys
# COCO class names and colors setup
COCO_NAMES = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
              'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
              'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
              'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
              'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
              'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
              'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
              'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
              'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator',
              'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']

# Pre-compute colors for all classes
CLASS_COLORS = [tuple(int(x * 255) for x in colorsys.hsv_to_rgb((i * 0.1) % 1.0, 0.7, 0.9)) 
                for i in range(len(COCO_NAMES))]

def draw_detections(image, boxes, scores, labels, conf_threshold=0.45, nms_iou_threshold=0.45):
    # Apply confidence threshold
    mask = scores > conf_threshold
    boxes = boxes[mask]
    scores = scores[mask]
    labels = labels[mask]
    
    # Apply NMS per class
    final_boxes = []
    final_scores = []
    final_labels = []
    
    for class_id in np.unique(labels):
        class_mask = labels == class_id
        class_boxes = boxes[class_mask]
        class_scores = scores[class_mask]
        
        # Calculate IoU matrix
        x1 = class_boxes[:, 0]
        y1 = class_boxes[:, 1]
        x2 = class_boxes[:, 2]
        y2 = class_boxes[:, 3]
        areas = (x2 - x1) * (y2 - y1)
        
        order = class_scores.argsort()[::-1]
        keep = []
        
        while order.size > 0:
            i = order[0]
            keep.append(i)
            
            if order.size == 1:
                break
                
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            
            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            inter = w * h
            
            ovr = inter / (areas[i] + areas[order[1:]] - inter)
            ids = np.where(ovr <= nms_iou_threshold)[0]
            order = order[ids + 1]
        
        final_boxes.extend(class_boxes[keep])
        final_scores.extend(class_scores[keep])
        final_labels.extend([class_id] * len(keep))
    
    # Draw the filtered boxes
    for box, score, label_idx in zip(final_boxes, final_scores, final_labels):
        box = box.astype(int)
        rgb = CLASS_COLORS[label_idx]
        label_text = f"{COCO_NAMES[label_idx]}: {score:.2f}"
        cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), rgb, 2)
        cv2.putText(image, label_text, (box[0], box[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, rgb, 2)
    
    return image

@torch.no_grad()
def process_video(video_path, engine_path, output_path, conf_threshold=0.45, nms_threshold=0.45):
    video = cv2.VideoCapture(video_path)
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(video.get(cv2.CAP_PROP_FPS))

    video_writer = cv2.VideoWriter(
        output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    engine = Engine(engine_path)
    engine.load()
    engine.activate()
    engine.allocate_buffers()
    cudaStream = torch.cuda.current_stream().cuda_stream

    idx = 0
    start = timeit.default_timer()

    while True:
        success, frame = video.read()
        if not success:
            break
        idx += 1

        try:
            # Preprocess
            image = cv2.resize(frame, (640, 640))
            image_bchw = np.transpose(np.expand_dims(image, 0), (0, 3, 1, 2))
            image_torch = torch.from_numpy(image_bchw)

            # Inference
            result = engine.infer({"input": image_torch}, cudaStream)
            
            # Get boxes and scores
            output_keys = [k for k in result.keys() if k != "input"]
            boxes = result[output_keys[0]].cpu().numpy()[0]
            scores = result[output_keys[1]].cpu().numpy()[0]
            
            # Get class predictions
            class_scores = np.max(scores, axis=1)
            class_ids = np.argmax(scores, axis=1)

            # Draw detections
            result = draw_detections(
                image, 
                boxes, 
                class_scores, 
                class_ids,
                conf_threshold=conf_threshold,
                nms_iou_threshold=nms_threshold
            )
            
            upscaled = cv2.resize(result, (width, height))
            video_writer.write(upscaled)

        except Exception as e:
            video_writer.write(frame)
            continue

    end = timeit.default_timer()
    print('FPS: ', idx/(end-start), 'Frames: ', idx)

    video.release()
    video_writer.release()

if __name__ == "__main__":
    process_video(
        video_path="input.mp4",
        engine_path="./yolo_nas_m_coco.engine",
        output_path="output_video.mp4",
        conf_threshold=0.45,
        nms_threshold=0.45
    )
