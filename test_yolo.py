from ultralytics import YOLO

model = YOLO("yolov8n.pt")

results = model("test_images/test2.jpg", save=True)

print("Detection complete!")