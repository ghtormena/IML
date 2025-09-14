import os
import shutil
import random

# Pastas originais (classes)
input_dirs = [
    "Cranio Feminino",
    "Cranio Masculino",
    "Pelve Feminina",
    "Pelve Masculina",
]

# Pasta de saída no formato YOLO classification
out_base = "dataset_yolo"
splits = ["train", "val", "test"]

# Se já existir, remove a estrutura antiga
if os.path.exists(out_base):
    shutil.rmtree(out_base)

for split in splits:
    for cls in input_dirs:
        os.makedirs(os.path.join(out_base, split, cls), exist_ok=True)

# Proporções de divisão
train_ratio = 0.75
val_ratio = 0.10
test_ratio = 0.15

# Extensões válidas
valid_ext = (".jpg", ".jpeg", ".png", ".bmp", ".tiff")

for cls in input_dirs:
    files = [f for f in os.listdir(cls) if f.lower().endswith(valid_ext)]
    random.shuffle(files)

    n = len(files)
    n_train = int(train_ratio * n)
    n_val = int(val_ratio * n)
    # resto vai para teste
    n_test = n - n_train - n_val

    train_files = files[:n_train]
    val_files = files[n_train:n_train + n_val]
    test_files = files[n_train + n_val:]

    # Copiar para destino
    for fname in train_files:
        shutil.copy(os.path.join(cls, fname), os.path.join(out_base, "train", cls, fname))
    for fname in val_files:
        shutil.copy(os.path.join(cls, fname), os.path.join(out_base, "val", cls, fname))
    for fname in test_files:
        shutil.copy(os.path.join(cls, fname), os.path.join(out_base, "test", cls, fname))

    print(f"✅ Classe '{cls}': {n_train} treino, {n_val} validação, {n_test} teste")

print("\nEstrutura pronta em:", out_base)