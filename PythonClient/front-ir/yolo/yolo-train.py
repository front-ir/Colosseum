from ultralytics import YOLO
import os
from multiprocessing import Process, freeze_support, set_start_method

def train():

    script_path = os.path.dirname(p=os.path.realpath(__file__))

    # Load a YOLOv8 model, pre-trained on COCO or start fresh
    model = YOLO('yolov8n.pt')  # 'n' stands for nano model, can also be 's', 'm', 'l', 'x'

    # Train on your custom dataset
    data_path = os.path.join(script_path, "../screenshots", "yolo2.yaml")
    model.train(data=data_path, epochs=200, batch=1, imgsz=640, device='cuda', workers=2, optimizer='auto')

if __name__ == "__main__":
    freeze_support()
    set_start_method('spawn')
    p = Process(target=train)
    p.start()