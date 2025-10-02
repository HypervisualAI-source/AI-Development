### AI Development                                                                                        
Hypervisual AI is currently developing a project that incorporates YOLOv13-N, with plans to integrate ViT-B-16 in future iterations.
                                                                                                    
### Install Dependencies                                                              
1. git clone https://github.com/HypervisualAI-source/AI-Development.git                      
2. cd AI-Development
3. python3 -m venv venv
4. source venv/bin/activate
5. pip3 install -r requirements.txt

### Usage Guide
1. Training Model
   
    cd ./src/YOLOv13-N/
   
	python3 train.py      
	
2. Inferencing Model
   
    cd ./src/YOLOv13-N/
                        
	python3 inference.py

3. Demonstrating YOLOv13-N

    cd ./demos/YOLOv13-N/
   
	python3 demo.py 

    3.1 Demonstrating YOLOv13-N in shell script file
   
        cd ./demos/YOLOv13-N/
   
        chmod +x demo.sh
   
	    ./demo.sh

5. Demonstrating ViT-B-16
   
    cd ./demos/ViT-B-16/
   
	python3 demo.py                  
                                       
    4.1 Demonstrating ViT-B-16 in shell script file
   
        cd ./demos/ViT-B-16/
   
        chmod +x demo.sh
   
	    ./demo.sh

	
### Demos
#### YOLOv13-N
![Image](demos/YOLOv13-N/source/yolo_output.gif)

#### ViT-B-16
![Image](demos/ViT-B-16/source/vit_output.gif)

### Improvements
Comapared to the version (v0.0.rc1), the improvements of the version (v0.0.rc2) are:
1. Separate the demo into two sub-demos, YOLOv13-N and ViT-B-16 (demos/ViT-B-16 nad demos/YOLOv13-N)
2. Record the output from sub-demo in video (demos/YOLOv13-N/demo.py and demos/ViT-B-16/demo.py)
3. Supplement the pre-processing methods (src/YOLOv13-N/ultralytics/data/augment.py)
4. Upload the training dataset, COCO dataset (URL: http://images.cocodataset.org/zips/train2017.zip and http://images.cocodataset.org/zips/val2017.zip)
5. Calculate Mean Average Precision (MAP) on validation stage (src/YOLOv13-N/inference.py)

### TO DO
1. Make a comparison between YOLOv13-N and benchmark(YOLOv8, Ultralycs' latest official version)














