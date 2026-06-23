from ultralytics import YOLO

model = YOLO(r"F:\internship\archive\runs\detect\train-7\weights\best.pt")

metrics = model.val()

print(metrics)