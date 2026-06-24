from ultralytics import YOLO

def main():

    print("Loading extra-large base model...")
    model = YOLO('yolov8s.pt') 

    print("Starting training process...")
    results = model.train(
        data='datasets/data.yaml',    
        epochs=30,                                           
        imgsz=640,                                     
        batch=16,                                      
        name='ppe_detection_model_super'  
    )
    
    print("Training complete! Check the 'runs/detect/ppe_detection_model_super' folder for results.")

if __name__ == '__main__':
    main()