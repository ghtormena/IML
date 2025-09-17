import os
import shutil
import random

# Pastas originais (classes)
input_dirs = [
    "Cranio Feminino",
    "Cranio Masculino",
    "Pelve Feminina",
    "Pelve Masculina"
]

# Pasta de saída no formato YOLO classification
out_base = "dataset_yolo"
splits = ["train", "val", "test"]

# Se já existir, remove a estrutura antiga
if os.path.exists(out_base):
    shutil.rmtree(out_base)

# --- MODIFICAÇÃO AQUI ---
# Cria a estrutura de pastas de destino usando apenas o nome da classe
for split in splits:
    for cls_path in input_dirs:
        class_name = os.path.basename(cls_path) # NOVA LINHA: Extrai "Feminino" do caminho completo
        os.makedirs(os.path.join(out_base, split, class_name), exist_ok=True) # LINHA MODIFICADA

# Proporções de divisão
train_ratio = 0.75
val_ratio = 0.10
test_ratio = 0.15

# Extensões válidas
valid_ext = (".jpg", ".jpeg", ".png", ".bmp", ".tiff")

for cls_path in input_dirs: # LINHA MODIFICADA (variável renomeada para clareza)
    class_name = os.path.basename(cls_path) # NOVA LINHA: Extrai o nome da classe novamente

    files = [f for f in os.listdir(cls_path) if f.lower().endswith(valid_ext)]
    random.shuffle(files)

    n = len(files)
    n_train = int(train_ratio * n)
    n_val = int(val_ratio * n)
    # resto vai para teste
    n_test = n - n_train - n_val

    train_files = files[:n_train]
    val_files = files[n_train:n_train + n_val]
    test_files = files[n_train + n_val:]

    # --- MODIFICAÇÃO AQUI ---
    # Copiar para destino usando apenas o nome da classe
    for fname in train_files:
        shutil.copy(os.path.join(cls_path, fname), os.path.join(out_base, "train", class_name, fname))
    for fname in val_files:
        shutil.copy(os.path.join(cls_path, fname), os.path.join(out_base, "val", class_name, fname))
    for fname in test_files:
        shutil.copy(os.path.join(cls_path, fname), os.path.join(out_base, "test", class_name, fname))

    print(f"✅ Classe '{class_name}': {n_train} treino, {n_val} validação, {n_test} teste") # LINHA MODIFICADA

print("\nEstrutura pronta em:", out_base)
