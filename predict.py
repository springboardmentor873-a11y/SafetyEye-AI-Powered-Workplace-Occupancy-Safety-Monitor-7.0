from ultralytics import YOLO

model = YOLO(r"F:\internship\archive\runs\detect\train-7\weights\best.pt")

results = model.predict(
    source=r"F:\internship\archive\css-data\test\images",
    save=True,
    conf=0.25
)

print("Prediction completed!")