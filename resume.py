from ultralytics import YOLO

def main():
    # 1. Load the last saved checkpoint from the NEW super model folder
    print("Loading last checkpoint...")
    
    # Notice the folder name is now 'ppe_detection_model_super'
    model = YOLO('runs/detect/ppe_detection_model_super/weights/last.pt') 

    # 2. Tell YOLO to resume training from where it stopped
    print("Resuming training process...")
    results = model.train(resume=True)
    
    print("Training complete!")

if __name__ == '__main__':
    main()