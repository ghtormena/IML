from ultralytics import YOLO

# Load a model
model = YOLO("yolo_models/yolo11m-cls.pt")

# Train the model with optimized parameters
results = model.train(data="datasets/dataset_yolo_split", epochs=300, imgsz=640, lr0=0.001)

print("✅ Treinamento concluído!")