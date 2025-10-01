import numpy as np                                                  
import cv2     
import os    
import sys                
import torch                                                        
import copy                           
from PIL import Image
import time                                                                                   
import random       
import yaml                                
                                                                                              
                                                                                                                                                      
from ultralytics import YOLO                                                                                         
import torchvision.models as models   

                                                                                                                                                                                                                                                                                               
def demo(yolov13_model, video_path_yolo, imgsz, conf_thres, iou_thres,  task_yolo_cp,  coco_class_names, colors_yolo , fontScale_yolo_ori, vit_model,  video_path_vit_list, task_vit_cp, weights, fontScale_vit, thickness_vit, org_vit_category, org_vit_probability, color_vit, device, up_logo_cp, up_points, font, window_name):
                                                                                          
    cv2.namedWindow(window_name, cv2.WINDOW_GUI_NORMAL)
                                                                                                                                                                 
    while True:                                      
        if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
            break                                                             
                                                                                                                                                                                    
        cap = cv2.VideoCapture(video_path_yolo)                    
        prev_frame_time = 0
        new_frame_time = 0                                                                                                                                                                                                                          
        yolo_display_time = []
                                                                           
        frame_num = 0                                                                                                                            
        yolo_total = 0                                                              
        while True:                                                          
                ret, frame = cap.read()     
                
                if not ret:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue                
                                    
                start = time.time()       
                results = yolov13_model.predict(
                        source=frame,
                        imgsz=imgsz,                    
                        conf=conf_thres,                      
                        iou=iou_thres,
                        device=device, 
                        show=False,
                        save = False,                                                      
                        verbose=False                                   
                    )   
                                                                        
                
                end = time.time()                                                                                                                             
                inference_time =  str(int((end - start)  * 1000))                
                                                                                                                                                            
                object_boxes = results[0].boxes.xyxy
                object_colors = []                                                                        
                object_names = []                
                                                                                                                         
                for i, cls_id in enumerate(results[0].boxes.cls):                                         
                    name = coco_class_names[int(cls_id.item())]                     
                    object_names.append(name)                                                                                                                                                                                                     
                    id_object = int(cls_id.numpy())                     
                    object_colors .append(colors_yolo[id_object])                      
                                                                   
                object_confs_0 = (results[0].boxes.conf).numpy()
                object_confs = []                                                             
                for i in range(len(object_confs_0)):
                    a = round(object_confs_0[i], 2)                    
                    object_confs.append(a)                                       
                                                                                                                                                                                                                                                                                                                                                                           
                thickness = 2                                                                                                                                                                                                                        
                for object_box, object_color, object_name, object_conf in zip(object_boxes, object_colors, object_names, object_confs):
                    fontScale_yolo = fontScale_yolo_ori
                    text =  object_name + " "  +  str(object_conf)
                    (text_width, text_height), baseline = cv2.getTextSize(text, font, fontScale_yolo, thickness)  
                                                                                                                                                                       
                    while (object_box[3] - object_box[1]) < text_width:                                                                                                                                                                                                                       
                        fontScale_yolo = fontScale_yolo - 0.1                                                                                   
                        (text_width, text_height), baseline = cv2.getTextSize(text, font, fontScale_yolo, thickness)
                                                                                                                                                                        
                    top_left = (int(object_box[0].numpy()), int((object_box[1]).numpy() - text_height * 1.9) )                   
                    bottom_right = (int(object_box[0].numpy() + text_width), int((object_box[1]).numpy()))                                                                                                                                   
                    fill_color = object_color 
                                                                                                                                                                              
                    cv2.rectangle(frame, top_left, bottom_right, fill_color, thickness=-1)
                    cv2.rectangle(frame, (int(object_box[0].numpy()), int(object_box[1].numpy())), (int(object_box[2].numpy()), int(object_box[3].numpy())), object_color, 4)  
                    org_0 = (int(object_box[0].numpy()), int((object_box[1] - 8).numpy()))
                    
                    if object_color[0] == 255 and object_color[1] == 0 and object_color[2] == 0:
                        cv2.putText(frame, object_name + " "  +  str(object_conf), org_0, font, fontScale_yolo, [255, 255, 255], thickness, cv2.LINE_AA)
                    else:
                        cv2.putText(frame, object_name + " "  +  str(object_conf), org_0, font, fontScale_yolo, [0,0,0], thickness, cv2.LINE_AA)                        
                                                 
                                                          
                frame = cv2.resize(frame, up_points, interpolation= cv2.INTER_LINEAR)       
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)                                                                                                                                                                       
                frame = Image.fromarray(frame)                                                                                                                                                                                                                        
                frame_cp = frame.copy()                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               
                frame_cp.paste(up_logo_cp, (1650, 20))                                                                       
                frame_cp.paste(task_yolo_cp, (10, 20))                                                                                                                                                  
                frame = np.asarray(frame_cp)                                                                                                                                                                                                   
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) 
                                                                                  
                new_frame_time = time.time()       
                
                frame_time = new_frame_time - prev_frame_time
                fps = 1/ frame_time
                                                                                      
                prev_frame_time = new_frame_time
                fps = str(int(fps))                
                
                cv2.imshow(window_name, frame)
                cv2.waitKey(1)  

                print("\n")
                print("per frame at shape: "  + str((imgsz, imgsz, 3)) + "    "  + "inference: "  + inference_time + ("ms") + "    " + "fps: " + fps + "    " + "device: " + device)
                print("\n")                                                                                                            
                                                                       
                if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                    break                                                          

                if frame_num == 0:
                    pass
                else:
                    yolo_display_time.append((frame_time))
                                 
                yolo_total = sum(yolo_display_time)
                                                                                                                       
                frame_num += 1 
                                                                                               
                if yolo_total > 10:                                                          
                    for i in range(len(video_path_vit_list)) :   
                        cap = cv2.VideoCapture(video_path_vit_list[i])     
                        
                        vit_display_time = []                                                                   
                        frame_num = 0  
                                                                                                  
                        while True:                                                                                                                        
                                ret, frame = cap.read()                                                                                                                                                                                 
                                                                                                                                                          
                                frame_ori = frame.copy()                                                                                                                                                       
                                frame = cv2.resize(frame, (224, 224), interpolation= cv2.INTER_LINEAR)                                
                                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)                                                  
                                frame = torch.from_numpy(np.expand_dims(np.transpose(frame.astype(np.float32), (2, 0, 1)), axis = 0)) 
                                frame = frame / 255.0
                                
                                start = time.time()                                                    
                                output = vit_model(frame)  
                                end = time.time()
                                inference_time =  str(int((end - start)  * 1000))  
                                
                                probs = torch.nn.functional.softmax(output[0], dim=0)
                                top_prob, top_class_idx = torch.max(probs, dim=0)                                                                                                                                                                                                                                   
                                top_prob = f"{(top_prob.item() * 100):.2f}"                                                      
                                label = weights.meta["categories"][top_class_idx]
                                frame = cv2.resize(frame_ori, up_points, interpolation= cv2.INTER_LINEAR)
                                frame = cv2.putText(frame, "Category: " +  label, org_vit_category, font, fontScale_vit, color_vit, thickness_vit, cv2.LINE_AA)
                                frame = cv2.putText(frame, "Probability: " +  top_prob + "%",  org_vit_probability, font, fontScale_vit, color_vit, thickness_vit, cv2.LINE_AA)
                                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)                    
                                frame = Image.fromarray(frame)                                                                                                                                                                                                                                                                                                                
                                frame_cp = frame.copy()                                                                                                                                                                                                                                                                                                                                                                                                                              
                                                                                                                      
                                frame_cp.paste(up_logo_cp, (1650, 20))                                                                                                                                                                                                                       
                                frame_cp.paste(task_vit_cp, (10, 20))                                                                                                                                                                                                                                                                                  
                                frame = np.asarray(frame_cp)                                                                                           
                                                                                                                                                                                  
                                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                                display_frame = frame.copy() 
                                
                                new_frame_time = time.time()                                                                     
                                
                                frame_time = (new_frame_time-prev_frame_time)
                                fps = 1/frame_time
                                prev_frame_time = new_frame_time
                                fps = str(int(fps))                 
                                                                                                                                                                                
                                cv2.imshow(window_name, display_frame)
                                cv2.waitKey(1)                                                                                                                                                             
                              
                                print("\n")
                                print("per frame at shape: "  + str((224, 224, 3)) + "    "  + "inference: "  + inference_time + ("ms") + "    " + "fps: " + fps + "    " + "device: " + device)
                                print("\n")                                                
                                                                                                                
                                if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                                    break                                       
                                if frame_num == 0:                   
                                    pass
                                else:                                                                                                                                                                                              
                                    vit_display_time.append((frame_time))                                               
                                vit_display_total = sum(vit_display_time)                                                     
                                frame_num += 1    
                                if vit_display_total > 5:
                                    break                                            
                                                                                                                                                         
                        if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                            break                                                                            
                    break                                                                                                                                                                                                                                                                                                                                                     
                                              
                                                                          
                                                                 
                                                                                                                                                                                                                          

                                                                                                                           
if __name__ == "__main__":

    # read the configuration parameters from cofig.yal file
    with open('config.yml', 'r') as file:
        config = yaml.safe_load(file)

    # video source     
    video_path_yolo = config["video_source"]["video_path_yolo"]
    video_path_vit_0 = config["video_source"]["video_path_vit_0"]
    video_path_vit_1 = config["video_source"]["video_path_vit_1"]
    video_path_vit_2 = config["video_source"]["video_path_vit_2"]
    video_path_vit_3 = config["video_source"]["video_path_vit_3"]
    video_path_vit_4 = config["video_source"]["video_path_vit_4"]
    video_path_vit_5 = config["video_source"]["video_path_vit_5"]
    video_path_vit_6 = config["video_source"]["video_path_vit_6"]
    video_path_vit_7 = config["video_source"]["video_path_vit_7"]

    # make a list of videos for vit demo                                                                                                        
    video_path_vit_list  = []                                                                                                                                                                    
    video_path_vit_list.append(video_path_vit_0)
    video_path_vit_list.append(video_path_vit_1)
    video_path_vit_list.append(video_path_vit_2)
    video_path_vit_list.append(video_path_vit_3)
    video_path_vit_list.append(video_path_vit_4)
    video_path_vit_list.append(video_path_vit_5)
    video_path_vit_list.append(video_path_vit_6)
    video_path_vit_list.append(video_path_vit_7)
                                                  
    # the neccessary file of YOLov13-N
    yolov13_weights_path = config["yolov13_weight_cfg"]["yolov13_weights_path"]
    yolov13_cfg_path = config["yolov13_weight_cfg"]["yolov13_cfg_path"]
                        

    # check the necessary files
    if not os.path.exists(yolov13_weights_path):                           
        raise FileNotFoundError(f"'{yolov13_weights_path}' YOLOs' weights file not found. Please download it and place it in the correct path.")
    if not os.path.exists(video_path_yolo):
        raise FileNotFoundError(f"'{video_path_yolo}' video file not found. Please provide a video to test.")
    if not os.path.exists(video_path_vit_0):
        raise FileNotFoundError(f"'{video_path_vit_0}' video file not found. Please provide a video to test.")   
    if not os.path.exists(video_path_vit_1):
        raise FileNotFoundError(f"'{video_path_vit_1}' video file not found. Please provide a video to test.")   
    if not os.path.exists(video_path_vit_2):
        raise FileNotFoundError(f"'{video_path_vit_2}' video file not found. Please provide a video to test.") 
    if not os.path.exists(video_path_vit_3):
        raise FileNotFoundError(f"'{video_path_vit_3}' video file not found. Please provide a video to test.") 
    if not os.path.exists(video_path_vit_4):
        raise FileNotFoundError(f"'{video_path_vit_4}' video file not found. Please provide a video to test.")  
    if not os.path.exists(video_path_vit_5):
        raise FileNotFoundError(f"'{video_path_vit_5}' video file not found. Please provide a video to test.") 
    if not os.path.exists(video_path_vit_6):
        raise FileNotFoundError(f"'{video_path_vit_6}' video file not found. Please provide a video to test.")     
    if not os.path.exists(video_path_vit_7):
        raise FileNotFoundError(f"'{video_path_vit_7}' video file not found. Please provide a video to test.")          

    if not os.path.exists(yolov13_cfg_path):
        raise FileNotFoundError(f"'{yolov13_cfg_path}' yolov13 configuration file not found. Please provide a video to test.")
                                                                                                   
                                                                                      
    # optional device
    device = "cpu"                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       
                                                                                                                                                                                                                                                                                 
    # load YOLOv13-N                                                                                                                                            
    yolov13_model = YOLO(yolov13_cfg_path)
    yolov13_model.load(yolov13_weights_path)                                                                                                                                                               
                                                                                                         
    #  load ViT-B-16                                                                                                                                                                                                                                                                                  
    vit_model = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)
    vit_model.eval()    
                                                                                                                                            
    weights=models.ViT_B_16_Weights.IMAGENET1K_V1   
                                                                                                         
    # logo
    logo_path = config["logo"]["logo_path"]
    logo = cv2.imread(logo_path)
    up_width = int(logo.shape[1] * config["logo"]["up_width"])
    up_height = int(logo.shape[0] * config["logo"]["up_width"])
    up_points = (up_width, up_height)                       
    up_logo = cv2.resize(logo, up_points, interpolation= cv2.INTER_LINEAR)       
    up_logo = cv2.cvtColor(up_logo, cv2.COLOR_BGR2RGB)
    up_logo = Image.fromarray(up_logo)
    up_logo_cp = up_logo.copy()                                            

    # title
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = config["title"]["fontScale"]
    thickness = config["title"]["thickness"]
    org = tuple(config["title"]["org"])
    text_color = tuple(config["title"]["text_color"])
    text_background_color = config["title"]["text_background_color"]
    text_area_yolo = tuple(config["title"]["text_area_yolo"])
    text_area_vit = tuple(config["title"]["text_area_vit"])
    task_background_yolo = np.full(text_area_yolo, text_background_color, dtype=np.uint8)
    task_background_vit = np.full(text_area_vit, text_background_color, dtype=np.uint8)

    # task 0: YOLOv13-N  detection
    text_yolo = config["task_yolo"]["task_text"]
    task_yolo = cv2.putText(task_background_yolo, text_yolo,  org,  font, fontScale, text_color, thickness, cv2.LINE_AA)
    task_yolo= cv2.cvtColor(task_yolo, cv2.COLOR_BGR2RGB)                                             
    task_yolo = Image.fromarray(task_yolo)  
    task_yolo_cp = task_yolo.copy()                                      

    # task 1: ViT-B-16 classification                       
    text_vit = config["task_vit"]["task_text"]
    task_vit = cv2.putText(task_background_vit, text_vit,  org, font, fontScale, text_color, thickness, cv2.LINE_AA)
    task_vit = cv2.cvtColor(task_vit, cv2.COLOR_BGR2RGB)
    task_vit = Image.fromarray(task_vit)  
    task_vit_cp = task_vit.copy()

    #  the display features for ViT-B-16                                                                                                                                                                  
    fontScale_vit = config["display_ViT"]["fontScale_vit"]
    thickness_vit = config["display_ViT"]["thickness_vit"]
    org_vit_category = config["display_ViT"]["org_vit_category"]
    org_vit_probability = config["display_ViT"]["org_vit_probability"]
    color_vit = config["display_ViT"]["color_vit"]
                                                                                                                                            
    #  the features of YOLOv13-N detection                                                                                                                                                              
    imgsz = config["yolov13_features"]["imgsz"]
    conf_thres = config["yolov13_features"]["conf_thres"]
    iou_thres = config["yolov13_features"]["iou_thres"]
    fontScale_yolo_ori = config["yolov13_features"]["fontScale_yolo_ori"]
    colors_yolo = [[random.randint(0, 255) for _ in range(3)] for _ in range(80)]
    
    # the distinct colors
    colors_yolo[0] = config["distinct_colors"]["colors_0"]
    colors_yolo[1] = config["distinct_colors"]["colors_1"]
    colors_yolo[2] = config["distinct_colors"]["colors_2"]
    colors_yolo[3] = config["distinct_colors"]["colors_3"]
    colors_yolo[4] = config["distinct_colors"]["colors_4"]
    colors_yolo[5] = config["distinct_colors"]["colors_5"]
    colors_yolo[6] = config["distinct_colors"]["colors_6"]
    colors_yolo[7] = config["distinct_colors"]["colors_7"]

    # display resolution: 1920 x 1080
    up_width = config["display_resolution"]["up_width"]
    up_height = config["display_resolution"]["up_height"]
    up_points = (up_width, up_height)                                              
                                                                                                                                                             
    # the class name of COCO
    coco_path = config["coco"]["coco_path"]
    with open(coco_path, "r") as f:
        coco_class_names = [line.strip() for line in f .readlines()]

    # the name of window
    window_name = config["window"]["window_name"]

    # demo function                                                     
    demo(yolov13_model, video_path_yolo, imgsz, conf_thres, iou_thres,  task_yolo_cp,  coco_class_names, colors_yolo , fontScale_yolo_ori, vit_model,  video_path_vit_list, task_vit_cp, weights, fontScale_vit, thickness_vit, org_vit_category, org_vit_probability, color_vit, device, up_logo_cp, up_points, font, window_name)
                                                                        
