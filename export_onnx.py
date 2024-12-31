from super_gradients.training import models
from super_gradients.common.object_names import Models

model = models.get(Models.YOLO_NAS_S, pretrained_weights="coco")

model.eval()
model.prep_model_for_conversion(input_size=[1, 3, 640, 640])
model.export("yolo_nas_s.onnx", postprocessing=None, preprocessing=None)

model = models.get(Models.YOLO_NAS_M, pretrained_weights="coco")

model.eval()
model.prep_model_for_conversion(input_size=[1, 3, 640, 640])
model.export("yolo_nas_m.onnx", postprocessing=None, preprocessing=None)

model = models.get(Models.YOLO_NAS_L, pretrained_weights="coco")

model.eval()
model.prep_model_for_conversion(input_size=[1, 3, 640, 640])
model.export("yolo_nas_l.onnx", postprocessing=None, preprocessing=None)