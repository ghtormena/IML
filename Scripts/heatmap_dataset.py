import os
import cv2
import torch
import numpy as np
from ultralytics import YOLO
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

# Caminhos base
base_dir = os.path.expanduser("~/Documents/IML2/Scripts/dataset_yolo/test")
heatmap_root = os.path.expanduser("~/Documents/IML2/Scripts/HEATMAPS")
model_path = os.path.expanduser("~/Documents/IML2/Scripts/runs/classify/train11/weights/best.pt")

# 🔢 Encontra o próximo índice disponível para a pasta de saída
existing_dirs = [d for d in os.listdir(heatmap_root) if d.startswith("heatmap_test")]
indices = []
for d in existing_dirs:
    parts = d.split("_")
    if len(parts) > 2 and parts[-1].isdigit():
        indices.append(int(parts[-1]))
next_index = max(indices, default=0) + 1

# Nova pasta de saída
output_base = os.path.join(heatmap_root, f"heatmap_test_{next_index}")
os.makedirs(output_base, exist_ok=True)
print(f"\n📁 Criando pasta de saída: {output_base}")

# Carrega modelo YOLO de classificação
model = YOLO(model_path)
model.model.eval()

# Define dispositivo
device = "cuda" if torch.cuda.is_available() else "cpu"
model.model.to(device)

# Define a camada alvo (penúltima convolucional)
target_layers = [model.model.model[-2]]
cam = GradCAM(model=model.model, target_layers=target_layers)

# Percorre cada subpasta do dataset
for class_folder in os.listdir(base_dir):
    class_input_dir = os.path.join(base_dir, class_folder)
    if not os.path.isdir(class_input_dir):
        continue  # ignora arquivos fora das pastas
    
    class_output_dir = os.path.join(output_base, class_folder)
    os.makedirs(class_output_dir, exist_ok=True)

    print(f"\n🔍 Processando pasta: {class_folder}")

    # Percorre cada imagem PNG
    for file_name in os.listdir(class_input_dir):
        if not file_name.lower().endswith(".png"):
            continue
        
        image_path = os.path.join(class_input_dir, file_name)
        output_path = os.path.join(class_output_dir, file_name)

        # Lê a imagem
        img = cv2.imread(image_path)
        if img is None:
            print(f"⚠️ Erro ao ler {image_path}")
            continue
        
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        rgb_img = np.float32(rgb_img) / 255.0

        # Pré-processa
        input_tensor = preprocess_image(rgb_img, mean=[0, 0, 0], std=[1, 1, 1])
        input_tensor = input_tensor.to(device)
        input_tensor.requires_grad_(True)

        # Forward
        pred = model.model.model(input_tensor)[0]
        pred.retain_grad()

        # Classe predita
        pred_class = pred.argmax().item()

        # Gera Grad-CAM
        targets = [ClassifierOutputTarget(pred_class)]
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0, :]

        # Sobrepõe o mapa na imagem
        visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

        # Salva o resultado
        cv2.imwrite(output_path, cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR))
        print(f"✅ {file_name} -> {output_path}")

print(f"\n🏁 Todos os heatmaps foram gerados com sucesso em: {output_base}")