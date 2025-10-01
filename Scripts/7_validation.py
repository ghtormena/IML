from ultralytics import YOLO

# Load a model
model = YOLO("runs/classify/train/weights/best.pt")  # load a pretrained model (recommended for training)

# Train the model
results = model.val(data="data.yaml", split="test")

print("✅ Validação concluída!")
print(f"Top-1 Accuracy: {results.top1:.4f}")
print(f"Top-5 Accuracy: {results.top5:.4f}")

