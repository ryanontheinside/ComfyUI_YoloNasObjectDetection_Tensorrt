# ComfyUI YOLO NAS Object Detection with TensorRT


# this is a work in progress and is experiemental at best

This custom node adds YOLO NAS Object Detection support to ComfyUI using TensorRT for fast inference.

## Features

- Fast object detection using YOLO NAS model
- TensorRT optimization for high-performance inference
- Support for 80 COCO classes
- Automatic color coding for different object classes
- Bounding box visualization with class labels and confidence scores

## Installation

1. Clone this repository into your `ComfyUI/custom_nodes` directory:
```bash
cd ComfyUI/custom_nodes
git clone https://github.com/ryanontheinside/ComfyUI_YoloNasObjectDetection_Tensorrt
cd ComfyUI_YoloNasObjectDetection_Tensorrt
pip install -r requirements.txt
```

2. Download the pre-converted ONNX model from [HuggingFace](https://huggingface.co/ryanontheinside/yolo_nas_coco_onnx/tree/main)

3. Convert to TensorRT:
```bash
python export_trt.py
```

Alternatively, if you want to convert from scratch:
```bash
# Install super-gradients for model export
pip install super-gradients

# Export ONNX model
python export_onnx.py

# Convert to TensorRT
python export_trt.py
```


## Usage

1. Find the "Yolo Nas Detection Tensorrt" node under the `tensorrt` category
2. Connect an image input to the node
3. Select your TensorRT engine file from the dropdown
4. The node will output the image with detected objects, bounding boxes, and labels

## Supported Classes

The model supports the 80 COCO classes including:
- Person
- Vehicle classes (car, truck, bicycle, etc.)
- Animals (dog, cat, bird, etc.)
- Common objects (bottle, chair, etc.)
- And many more...

## Requirements

- NVIDIA GPU with CUDA support
- TensorRT installed
- Python packages listed in requirements.txt

## Environment Tested

- Ubuntu 22.04 LTS
- CUDA 12.4
- TensorRT 10.1.0
- Python 3.10
- NVIDIA GPUs (tested on H100)

## Credits

- [Super-Gradients](https://github.com/Deci-AI/super-gradients) for the YOLO NAS implementation

- [yuvraj108c](https://github.com/yuvraj108c/ComfyUI-YoloNasPose-Tensorrt) for the pose detection node

## License

This project is licensed under the Apache License 2.0 - see the LICENSE file for details.
