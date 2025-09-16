from ultralytics import YOLO

# Load a model
model = YOLO("yolo11s-cls.pt")  # load a pretrained model (recommended for training)

# Train the model
results = model.train(data="dataset_yolo", epochs=300, imgsz=640)

print("✅ Treinamento concluído!")