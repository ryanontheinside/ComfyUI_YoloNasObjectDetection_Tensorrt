import os
from utilities import Engine
import folder_paths
import torch
import time
import os
from utilities import Engine


ENGINE_PATH = os.path.join(folder_paths.models_dir, "tensorrt", "yolo-nas-detection")
os.makedirs(ENGINE_PATH, exist_ok=True)  
folder_paths.add_model_folder_path("tensorrt", ENGINE_PATH)

# MODEL_REPO = "https://huggingface.co/ryanontheinside/yolo_nas_coco_onnx/tree/main"

#the downloaded onnx model
ONNX_PATH = "./yolo_nas_detection_s_coco.onnx"


def export_trt(trt_path: str, onnx_path: str, use_fp16: bool):
    # Create directory for trt_path if it doesn't exist
    os.makedirs(os.path.dirname(trt_path), exist_ok=True)
    
    engine = Engine(trt_path)
    
    torch.cuda.empty_cache()
    
    s = time.time()
    ret = engine.build(
        onnx_path,
        use_fp16,
        enable_preview=True,
    )
    e = time.time()
    print(f"Time taken to build: {(e-s)} seconds")
    
    return ret

#trt_path: path to your comfy models directory
#onnx_path: path to where you downloaded the onnx model

export_trt(trt_path=os.path.join(ENGINE_PATH, os.path.basename(ONNX_PATH).replace('.onnx', '.engine')),
          onnx_path=ONNX_PATH,
          use_fp16=True)

