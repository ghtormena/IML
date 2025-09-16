import os
import cv2
import albumentations as A

INPUT_SIZE = 640

NUM_AUGMENTATIONS_PER_IMAGE = 4

# Transformações
transform = A.Compose([
    A.OneOf([
        A.RandomResizedCrop(size=(INPUT_SIZE, INPUT_SIZE), scale=(0.6, 1.0), p=1.0),
        A.Rotate(limit=30, p=1.0),
    ], p=0.7),
    A.RGBShift(r_shift_limit=25, g_shift_limit=25, b_shift_limit=25, p=0.3),
    A.ImageCompression(quality_range=(85, 100), p=0.4),
    A.Blur(blur_limit=3, p=0.3),
    A.GaussNoise(std_range=(0.1, 0.2), p=0.3),
    A.HorizontalFlip(p=0.5),
    A.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=0.3),
])

# Diretório base do YOLO
base_dir = "dataset_yolo/train/"

# Extensões válidas
valid_ext = (".jpg", ".jpeg", ".png", ".bmp", ".tiff")

total_generated =0

# Loop pelas classes (pastas dentro de train)
for cls_name in os.listdir(base_dir):
    cls_path = os.path.join(base_dir, cls_name)
    if not os.path.isdir(cls_path):
        continue  # pula arquivos que não sejam diretórios

    print(f"➡️ Processando classe: {cls_name}")

    # IMPORTANTE: Lê a lista de arquivos originais ANTES de começar a adicionar novos
    original_files = [f for f in os.listdir(cls_path) if f.lower().endswith(valid_ext)]
    print(f"   Encontradas {len(original_files)} imagens originais.")

    for fname in original_files:
        img_path = os.path.join(cls_path, fname)
        img = cv2.imread(img_path)

        if img is None:
            print(f"⚠️ Erro ao ler {img_path}")
            continue

        # Converte BGR → RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Loop para gerar as novas imagens
        for i in range(1, NUM_AUGMENTATIONS_PER_IMAGE + 1):
            # Aplica transformações (será diferente a cada iteração)
            transformed = transform(image=img_rgb)
            aug_img = transformed["image"]

            # Volta para BGR para salvar
            aug_img_bgr = cv2.cvtColor(aug_img, cv2.COLOR_RGB2BGR)

            # --- Cria o novo nome do arquivo ---
            # Pega o nome base e a extensão (ex: "imagem1", ".jpg")
            base_name, extension = os.path.splitext(fname)
            # Cria o novo nome (ex: "imagem1_aug_1.jpg")
            new_fname = f"{base_name}_aug_{i}{extension}"
            new_img_path = os.path.join(cls_path, new_fname)

            # Salva a nova imagem aumentada
            cv2.imwrite(new_img_path, aug_img_bgr)
            total_generated += 1

    print(f"✅ Concluído: {cls_name}")

print("\n🎉 Todas as imagens de treino foram aumentadas com sucesso!")
print(f"   Total de novas imagens geradas: {total_generated}")
