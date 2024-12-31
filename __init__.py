import os
import folder_paths
import numpy as np
import torch
import cv2
import colorsys
from .utilities import Engine

ENGINE_DIR = os.path.join(folder_paths.models_dir, "tensorrt", "yolo-nas-detection")
os.makedirs(ENGINE_DIR, exist_ok=True)
folder_paths.add_model_folder_path("tensorrt", ENGINE_DIR)

# Pre-compute colors for all COCO classes
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

@torch.no_grad()
def draw_detections(image, boxes, scores, labels, conf_threshold=0.45, nms_iou_threshold=0.45):
    """Optimized detection box drawing with NMS"""
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

class YoloNasDetectionTensorrt:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",),
                "engine": (os.listdir(ENGINE_DIR),),
            },
            "optional": {
                "confidence_threshold": ("FLOAT", {
                    "default": 0.45, 
                    "min": 0.01, 
                    "max": 1.0, 
                    "step": 0.01
                }),
                "nms_threshold": ("FLOAT", {
                    "default": 0.45,
                    "min": 0.01,
                    "max": 1.0,
                    "step": 0.01
                })
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "detect"
    CATEGORY = "tensorrt"

    def __init__(self):
        self.engine = None
        self.stream = None

    @torch.no_grad()
    def detect(self, images, engine, confidence_threshold=0.45, nms_threshold=0.45):
        # Lazy load engine
        if self.engine is None or self.engine_path != engine:
            self.engine_path = engine
            self.engine = Engine(os.path.join(ENGINE_DIR, engine))
            self.engine.load()
            self.engine.activate()
            self.engine.allocate_buffers()
            self.stream = torch.cuda.current_stream().cuda_stream

        # Process images efficiently
        images_bchw = images.permute(0, 3, 1, 2)
        images_resized = torch.nn.functional.interpolate(
            images_bchw, size=(640, 640), mode='bilinear', align_corners=False)
        images_uint8 = (images_resized * 255.0).type(torch.uint8)
        
        # Process batch
        detection_frames = []
        for idx, img in enumerate(images_uint8.split(1)):
            # Run inference
            result = self.engine.infer({"input": img}, self.stream)
            
            # Get boxes and scores from the two output tensors
            output_keys = [k for k in result.keys() if k != "input"]
            boxes = result[output_keys[0]].cpu().numpy()[0]
            scores = result[output_keys[1]].cpu().numpy()[0]
            
            # Get class predictions
            class_scores = np.max(scores, axis=1)
            class_ids = np.argmax(scores, axis=1)
            
            # Convert original image to numpy and correct format
            orig_img = (images[idx] * 255).cpu().numpy().astype(np.uint8)
            orig_img_resized = cv2.resize(orig_img, (640, 640))
            
            # Draw detections with user-specified thresholds
            output_img = draw_detections(
                orig_img_resized, 
                boxes, 
                class_scores, 
                class_ids,
                conf_threshold=confidence_threshold,
                nms_iou_threshold=nms_threshold
            )
            detection_frames.append(output_img)

        # Batch process output frames
        detection_frames = torch.from_numpy(np.stack(detection_frames)).float() / 255.0
        return (torch.nn.functional.interpolate(
            detection_frames.permute(0, 3, 1, 2),
            size=(images.shape[1], images.shape[2]),
            mode='bilinear',
            align_corners=False
        ).permute(0, 2, 3, 1),)


NODE_CLASS_MAPPINGS = {
    "YoloNasDetectionTensorrt": YoloNasDetectionTensorrt
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "YoloNasDetectionTensorrt": "YoloNasDetectionTensorrt"
}