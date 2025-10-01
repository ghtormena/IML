from ultralytics import YOLO

# Load a model
model = YOLO("yolo_models/yolo11m-cls.pt")

# Train the model with optimized parameters
results = model.train(
    data="/home/nexus/davi/IML/datasets/novo_dataset_2d/data_split2", 
    epochs=300, 
    imgsz=640,
    batch=-1,
)

print("✅ Treinamento concluído!")