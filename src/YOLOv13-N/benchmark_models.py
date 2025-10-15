
import time                                                                 
import logging                            
from ultralytics.utils import LOGGER                                                                                                                                                                                                                                                                                                                                                
from ultralytics import YOLO                                                            
                                                                                                                           
LOGGER.setLevel(logging.WARNING)            
                                                                                
benchmark_model_list = []  
                                                                                                                                                            
                                                   
benchmark_model_list.append("yolov8n.pt") 
benchmark_model_list.append("yolov8s.pt")                          
benchmark_model_list.append("yolov8m.pt")  
benchmark_model_list.append("yolov8l.pt")                          
benchmark_model_list.append("yolov8x.pt") 


model_features = []                                                                          
for i in range(len(benchmark_model_list)):                                                    
    model_name = benchmark_model_list[i]                              
                                                                
                                                                                                   
    model = YOLO(model_name)   
    
    # model name
    model_features.append(model_name.split(".")[0])     
                            
    parameters_GFLOPs = model.info()                                                             
                                     
    # parameters   
    model_features.append(f"{(parameters_GFLOPs[1] / 1e6):.1f}")    
    # GFLOPs   
    model_features.append(f"{parameters_GFLOPs[3]: .1f}")                  
                                                                                                                                                                                                                       
                                                                                                                    
    time_list = []                                                         
    for n in range(101):  
        s_time = time.time()                                                                                    
        model("test.jpg", device='cpu')   
                
        e_time = time.time()                                                                    
        latency = (e_time - s_time)                                                                         
                                                      
        if n == 0:
            pass                          
        else:                              
            time_list.append(latency)                                                                               
                                                                                                                                                     
    average_latency = int((sum(time_list) / len(time_list)) * 1000)
    # average_latency                                                                                                                     
    model_features.append(f"{(average_latency):.2f}")                                                       
                                                                                                                                         
                                                                                                                                                                                        
    metrics = model.val(data='coco128.yaml',  verbose=False)                                                                  
    mAP50_95 = metrics.results_dict["metrics/mAP50-95(B)"]                                                    
    # mAP50_95                                                                                                                                                                                                
    model_features.append(f"{(mAP50_95 * 100):.1f}")                                                                                              
                                                                                                                                                                                  
                                                                                                                                                                  
features = ' '.join(model_features)                                                    
                                                                                                         
print("features:", features)                                              
                                                                                                                                            
                                    
                                                         
                                                                                                                          
                                                                        


                        
                    
                                              