from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO("./best.pt")

# Export to ONNX
model.export(format="onnx")
