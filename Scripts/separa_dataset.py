import os
import shutil
import random

# Pastas originais (classes)
input_dirs = [
    "Classes/Cranio Feminino",
    "Classes/Cranio Masculino",
    "Classes/Pelve Feminina",
    "Classes/Pelve Masculina"
]

# Pasta de saída no formato YOLO classification
out_base = "dataset_yolo"
splits = ["train", "val", "test"]

# Se já existir, remove a estrutura antiga
if os.path.exists(out_base):
    shutil.rmtree(out_base)

# Proporções de divisão
train_ratio = 0.75
val_ratio = 0.10
test_ratio = 0.15

# Extensões válidas
valid_ext = (".jpg", ".jpeg", ".png", ".bmp", ".tiff")

# --- Cria a estrutura de pastas de destino usando apenas o nome da classe ---
for split in splits:
    for cls_path in input_dirs:
        class_name = os.path.basename(cls_path)
        os.makedirs(os.path.join(out_base, split, class_name), exist_ok=True)

# --- Processa cada classe ---
for cls_path in input_dirs:
    class_name = os.path.basename(cls_path)

    # Lista apenas os subdiretórios de cada classe
    subdirs = [d for d in os.listdir(cls_path) if os.path.isdir(os.path.join(cls_path, d))]
    random.shuffle(subdirs)

    n = len(subdirs)
    n_train = int(train_ratio * n)
    n_val = int(val_ratio * n)
    n_test = n - n_train - n_val

    train_dirs = subdirs[:n_train]
    val_dirs = subdirs[n_train:n_train + n_val]
    test_dirs = subdirs[n_train + n_val:]

    # Criar log
    log_file = os.path.join(out_base, f"{class_name}_log.txt")
    with open(log_file, "w", encoding="utf-8") as log:
        log.write("TRAIN:\n")
        log.write("\n".join(train_dirs) + "\n\n")
        log.write("VAL:\n")
        log.write("\n".join(val_dirs) + "\n\n")
        log.write("TEST:\n")
        log.write("\n".join(test_dirs) + "\n")

    # Função auxiliar para copiar todos os arquivos de um subdir para a pasta de destino
    def copiar_subdirs_arquivos(subdir_list, split_name):
        dst_class_dir = os.path.join(out_base, split_name, class_name)
        for sd in subdir_list:
            src_dir = os.path.join(cls_path, sd)
            for root, _, files in os.walk(src_dir):
                for f in files:
                    if f.lower().endswith(valid_ext):
                        shutil.copy(os.path.join(root, f), os.path.join(dst_class_dir, f))

    # Copiar arquivos
    copiar_subdirs_arquivos(train_dirs, "train")
    copiar_subdirs_arquivos(val_dirs, "val")
    copiar_subdirs_arquivos(test_dirs, "test")

    print(f"✅ Classe '{class_name}': {n_train} dirs treino, {n_val} dirs validação, {n_test} dirs teste. Log: {log_file}")

print("\nEstrutura pronta em:", out_base)