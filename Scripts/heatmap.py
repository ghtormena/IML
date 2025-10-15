import cv2
import torch
import numpy as np
from ultralytics import YOLO
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

# Caminhos
model_path = "/home/giovanna/Documentos/IML/heat_map/best (1).pt"
image_path = "/home/giovanna/Documentos/IML/heat_map/pelve2.jpeg"

# Carrega o modelo YOLO de classificação
model = YOLO(model_path)
model.model.eval()

# Lê a imagem e converte para RGB
img = cv2.imread(image_path)
rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
rgb_img = np.float32(rgb_img) / 255.0

# Pré-processa imagem (sem unsqueeze)
input_tensor = preprocess_image(rgb_img, mean=[0, 0, 0], std=[1, 1, 1])
input_tensor = input_tensor.to("cuda" if torch.cuda.is_available() else "cpu")

# 💡 Garante que as operações rastreiem gradientes
input_tensor.requires_grad_(True)

# Move modelo para o mesmo device
device = input_tensor.device
model.model.to(device)

# 💡 Verifique as camadas disponíveis
# print(model.model.model)  # opcional

# Usa a penúltima camada convolucional
target_layers = [model.model.model[-2]]

# Cria GradCAM
cam = GradCAM(model=model.model, target_layers=target_layers)

# 💡 Faz forward mantendo gradientes
pred = model.model.model(input_tensor)[0]
pred.retain_grad()

# Classe mais provável
pred_class = pred.argmax().item()
print(f"Classe predita: {pred_class}")

# Gera mapa de ativação
targets = [ClassifierOutputTarget(pred_class)]
grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0, :]

# Gera visualização
visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

# Salva resultado
output_path = "/home/giovanna/Documentos/IML/heat_map/heatmap_result.jpg"
cv2.imwrite(output_path, cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR))
print(f"✅ Heatmap gerado com sucesso: {output_path}")
