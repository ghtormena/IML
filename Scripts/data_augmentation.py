import os
import cv2
import albumentations as A

INPUT_SIZE = 640

# Transformações
transform = A.Compose([
    A.RandomResizedCrop(size=(INPUT_SIZE, INPUT_SIZE), scale=(0.6, 1.0), p=0.4),
    A.RandomRotate90(p=0.5),
    A.Rotate(limit=30, p=0.7),
    A.RGBShift(r_shift_limit=25, g_shift_limit=25, b_shift_limit=25, p=0.4),
    A.ImageCompression(quality_range=(85, 100), p=0.4),
    A.Blur(blur_limit=3, p=0.3),
    A.GaussNoise(std_range=(0.1, 0.5), p=0.3),
    A.HorizontalFlip(p=0.5),
    # A.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=0.4),
])

# Diretório base do YOLO
base_dir = "dataset_yolo/train/"

# Extensões válidas
valid_ext = (".jpg", ".jpeg", ".png", ".bmp", ".tiff")

# Loop pelas classes (pastas dentro de train)
for cls_name in os.listdir(base_dir):
    cls_path = os.path.join(base_dir, cls_name)
    if not os.path.isdir(cls_path):
        continue  # pula arquivos que não sejam diretórios

    print(f"➡️ Processando classe: {cls_name}")

    for fname in os.listdir(cls_path):
        if fname.lower().endswith(valid_ext):
            img_path = os.path.join(cls_path, fname)
            img = cv2.imread(img_path)

            if img is None:
                print(f"⚠️ Erro ao ler {img_path}")
                continue

            # Converte BGR → RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Aplica transformações
            transformed = transform(image=img)
            aug_img = transformed["image"]

            # Volta para BGR para salvar
            aug_img = cv2.cvtColor(aug_img, cv2.COLOR_RGB2BGR)

            # Sobrescreve a imagem original
            cv2.imwrite(img_path, aug_img)

    print(f"✅ Concluído: {cls_name}")

print("\n🎉 Todas as imagens de treino foram aumentadas com sucesso!")
