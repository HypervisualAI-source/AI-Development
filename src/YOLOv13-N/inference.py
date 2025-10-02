import os
from ultralytics import YOLO 

# model weight and configration
base_path = os.path.dirname(__file__)
model_weight = os.path.join(base_path, 'models', 'yolov13n.pt')
model_config = os.path.join(base_path, 'ultralytics/cfg/models/v13', 'yolov13n.yaml')
                                                                                                         
# loading YOLOv13-N                                                                                                                                                                                                                                                                                                                  
model = YOLO(model_config)                                                                                                                                                                     
model.load(model_weight)       
                                                                                   
# calculating MAP                                                                                                                                                                                                   
metrics = model.val(data='coco.yaml')                                                                                                                                                                  
print("metrics:", metrics.results_dict)

                                                                                       
# inferencing model                                                                                                                                     
results = model("test.jpg")                         
results[0].show()                           



