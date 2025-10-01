Following the instruction to train YOLOv13-N, inference YOLOv13-N and demonstrate the YOLOv13-N and ViT-B-16

1. Creating a virtual environment, "venv"
    python3 -m venv venv

2. Activate the virtual environment
    source venv/bin/activate

3. Install the essential dependencies
    pip3 install -r requirements.txt        

4. Train YOLOv13-N from scratch 
    cd ./src/YOLOv13-N/
	python3 train.py                        

5. Inference YOLOv13-N                            
    cd ./src/YOLOv13-N/                      
	python3 inference.py

6. Demonstrate YOLOv13-N and ViT-B-16
    cd ./demos
	python3 demo.py                                                
                                                                                                       
7. Demonstrate YOLOv13-N and ViT-B-16 directly
    cd ./demos
    chmod +x demo.sh 
	./demo.sh
  
                    
                                         
# AI-Development
# AI-Development
