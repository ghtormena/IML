#!/usr/bin/env python3
import cv2
import os
import shutil
from pathlib import Path

# --- CONFIGURAÇÕES ---
INPUT_DIRS = [
    "dataset_original/Feminino",
    "dataset_original/Masculino"
    # Adicione outras pastas aqui, se necessário
]

OUTPUT_DIR = "dataset_normalizado"
TARGET_SIZE = 640


# --- FUNÇÃO DE PROCESSAMENTO ---
def pad_and_resize(image_path: str, target_size: int):
    """
    Lê uma imagem, adiciona padding para torná-la quadrada e redimensiona.
    """
    try:
        img = cv2.imread(image_path)
        if img is None:
            print(f"  - Aviso: Não foi possível ler a imagem {image_path}")
            return None

        h, w, _ = img.shape
        max_dim = max(h, w)
        pad_h = max_dim - h
        pad_w = max_dim - w

        padded_img = cv2.copyMakeBorder(
            img,
            top=pad_h // 2,
            bottom=pad_h - (pad_h // 2),
            left=pad_w // 2,
            right=pad_w - (pad_w // 2),
            borderType=cv2.BORDER_CONSTANT,
            value=[0, 0, 0]
        )

        resized_img = cv2.resize(padded_img, (target_size, target_size), interpolation=cv2.INTER_AREA)
        return resized_img

    except Exception as e:
        print(f"  - Erro ao processar {image_path}: {e}")
        return None


# --- LÓGICA PRINCIPAL ---
def main():
    output_path = Path(OUTPUT_DIR)

    # Limpa o diretório de saída se ele já existir
    if output_path.exists():
        print(f"⚠️  Diretório de saída '{output_path}' já existe. Removendo...")
        shutil.rmtree(output_path)

    print(f"🚀 Iniciando o pré-processamento de imagens...")

    for dir_path_str in INPUT_DIRS:
        input_path = Path(dir_path_str)
        if not input_path.is_dir():
            print(f"  - Aviso: Diretório de entrada '{input_path}' não encontrado. Pulando.")
            continue

        print(f"\n➡️  Processando diretório: '{input_path.name}'")

        image_files = [p for p in input_path.rglob('*') if p.is_file()]
        count = 0

        for image_file in image_files:
            if image_file.suffix.lower() in ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']:
                processed_image = pad_and_resize(str(image_file), TARGET_SIZE)
                if processed_image is not None:
                    # Cria caminho relativo e mantém estrutura dentro do diretório de saída
                    relative_path = image_file.relative_to(input_path)
                    save_path = output_path / input_path.name / relative_path.parent
                    save_path.mkdir(parents=True, exist_ok=True)

                    full_save_path = save_path / image_file.name
                    cv2.imwrite(str(full_save_path), processed_image)
                    count += 1

        print(f"   ✅ Concluído: {count} imagens processadas e salvas mantendo estrutura original.")

    print(f"\n🎉 Processo finalizado com sucesso!")
    print(f"   Todas as imagens foram salvas em '{OUTPUT_DIR}' com tamanho {TARGET_SIZE}x{TARGET_SIZE}.")


if __name__ == "__main__":
    main()
