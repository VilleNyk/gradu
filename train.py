from ultralytics import YOLO
import torch
from torch import multiprocessing


def main():
    print(torch.cuda.is_available()) 
    print(torch.cuda.device_count())  
    print(torch.cuda.get_device_name(0)) 

    model = YOLO("yolov8n-obb.pt")  

    model.train(data='lattices.yaml',  
                epochs=120,         
                imgsz=1280,         
                batch=8,           
                device=0,          
                hsv_h=0.01, hsv_s=0.6, hsv_v=0.3, flipud=0.3, fliplr=0.3
    )

if __name__ == '__main__':
    multiprocessing.set_start_method('spawn', force=True)  
    main()