import os
import shutil
import random

# --- CONFIGURAÇÕES ---
# 1. Pastas de origem (contém as 4 classes).
# Certifique-se de que este caminho está correto e as pastas existem.
INPUT_DIR_BASE = "/home/nexus/davi/IML/datasets/novo_dataset_2d/anonimizado_padded_separado"

# 2. Defina o nome da nova pasta de saída para os conjuntos train/val/test.
# Esta pasta será criada e não vai sobrescrever a de origem.
OUTPUT_DIR_SPLIT = "/home/nexus/davi/IML/datasets/novo_dataset_2d/data_split2"

# Proporções de divisão
train_ratio = 0.75
val_ratio = 0.10
test_ratio = 0.15

# Extensões válidas
valid_ext = (".jpg", ".jpeg", ".png", ".bmp", ".tiff")

# --- LÓGICA PRINCIPAL ---

# Limpa a pasta de saída se ela já existir
if os.path.exists(OUTPUT_DIR_SPLIT):
    print(f"⚠️ Diretório de saída '{OUTPUT_DIR_SPLIT}' já existe. Removendo...")
    shutil.rmtree(OUTPUT_DIR_SPLIT)

print(f"🚀 Iniciando a divisão do dataset para '{INPUT_DIR_BASE}'...")

# Define os diretórios de classes dentro da pasta de origem
class_dirs = [
    "Feminino_cranio",
    "Feminino_pelve",
    "Masculino_cranio",
    "Masculino_pelve"
]

# Cria a estrutura de pastas de destino (train/val/test e classes)
for split in ["train", "val", "test"]:
    for cls_name in class_dirs:
        os.makedirs(os.path.join(OUTPUT_DIR_SPLIT, split, cls_name), exist_ok=True)

# Processa cada pasta de classe individualmente
for cls_name in class_dirs:
    cls_path = os.path.join(INPUT_DIR_BASE, cls_name)

    # Verifica se a pasta de origem da classe existe
    if not os.path.isdir(cls_path):
        print(f"  - ❌ Aviso: Diretório '{cls_path}' não encontrado. Pulando...")
        continue

    files = [f for f in os.listdir(cls_path) if f.lower().endswith(valid_ext)]
    random.shuffle(files)

    n = len(files)
    n_train = int(train_ratio * n)
    n_val = int(val_ratio * n)
    # O restante vai para o teste
    n_test = n - n_train - n_val

    train_files = files[:n_train]
    val_files = files[n_train:n_train + n_val]
    test_files = files[n_train + n_val:]

    # Copia os arquivos para os novos diretórios
    for fname in train_files:
        shutil.copy(os.path.join(cls_path, fname), os.path.join(OUTPUT_DIR_SPLIT, "train", cls_name, fname))
    for fname in val_files:
        shutil.copy(os.path.join(cls_path, fname), os.path.join(OUTPUT_DIR_SPLIT, "val", cls_name, fname))
    for fname in test_files:
        shutil.copy(os.path.join(cls_path, fname), os.path.join(OUTPUT_DIR_SPLIT, "test", cls_name, fname))

    print(f"✅ Classe '{cls_name}': {n_train} treino, {n_val} validação, {n_test} teste")

print(f"\n🎉 Processo concluído! Estrutura final salva em: '{OUTPUT_DIR_SPLIT}'")