from ultralytics import YOLO
YOLO("yolov8n.pt").export(format="onnx", imgsz=640, simplify=True, opset=12, dynamic=False)
print("ONNX export complete.")
