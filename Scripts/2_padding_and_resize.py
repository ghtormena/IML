import cv2
import os
import shutil
from pathlib import Path

# --- CONFIGURAÇÕES ---
# 1. Coloque aqui a lista com os caminhos para TODAS as suas pastas de imagens originais.
INPUT_DIRS = [
    "/home/nexus/davi/IML/datasets/novo_dataset_2d/anonimizado/Masculino",
    "/home/nexus/davi/IML/datasets/novo_dataset_2d/anonimizado/Feminino"
    # Adicione outras pastas de classe aqui, se necessário
]

# 2. Defina o nome da pasta de saída onde as imagens processadas serão salvas.
OUTPUT_DIR = "/home/nexus/davi/IML/datasets/novo_dataset_2d/anonimizado_padded"

# 3. Defina o tamanho final desejado para as imagens.
TARGET_SIZE = 640

# --- FUNÇÃO PRINCIPAL DE PROCESSAMENTO ---

def pad_and_resize(image_path: str, target_size: int) -> 'numpy.ndarray | None':
    """
    Lê uma imagem, adiciona padding para torná-la quadrada e, em seguida,
    redimensiona para o tamanho alvo, preservando o aspect ratio original.

    Args:
        image_path (str): Caminho para a imagem de entrada.
        target_size (int): Tamanho final da imagem (largura e altura).

    Returns:
        numpy.ndarray | None: A imagem processada ou None se houver erro.
    """
    try:
        # Lê a imagem usando OpenCV
        img = cv2.imread(image_path)
        if img is None:
            print(f"  - Aviso: Não foi possível ler a imagem {image_path}")
            return None

        h, w, _ = img.shape
        
        # Encontra a maior dimensão (altura ou largura)
        max_dim = max(h, w)
        
        # Calcula o padding necessário para tornar a imagem quadrada
        pad_h = max_dim - h
        pad_w = max_dim - w
        
        # Adiciona o padding para centralizar a imagem.
        # cv2.BORDER_CONSTANT com value=[0, 0, 0] cria bordas pretas.
        padded_img = cv2.copyMakeBorder(
            img,
            top=pad_h // 2,
            bottom=pad_h - (pad_h // 2),
            left=pad_w // 2,
            right=pad_w - (pad_w // 2),
            borderType=cv2.BORDER_CONSTANT,
            value=[0, 0, 0]  # Cor preta para o padding
        )

        # Agora que a imagem é quadrada, redimensiona para o tamanho final
        # com interpolação de alta qualidade (INTER_AREA).
        resized_img = cv2.resize(
            padded_img,
            (target_size, target_size),
            interpolation=cv2.INTER_AREA
        )
        
        return resized_img
    except Exception as e:
        print(f"  - Erro ao processar {image_path}: {e}")
        return None

# --- LÓGICA PRINCIPAL DO SCRIPT ---

def main():
    """
    Executa o processo de padding e resize para todos os diretórios de entrada.
    """
    output_path = Path(OUTPUT_DIR)
    
    # Limpa o diretório de saída se ele já existir, para evitar arquivos antigos
    if output_path.exists():
        print(f"⚠️  Diretório de saída '{output_path}' já existe. Removendo...")
        shutil.rmtree(output_path)
    
    print(f"🚀 Iniciando o pré-processamento de imagens...")

    for dir_path_str in INPUT_DIRS:
        input_path = Path(dir_path_str)
        if not input_path.is_dir():
            print(f"  - Aviso: Diretório de entrada '{input_path}' não encontrado. Pulando.")
            continue
            
        # Cria a estrutura de pastas correspondente no diretório de saída
        output_class_dir = output_path / input_path.name
        output_class_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\n➡️  Processando diretório: '{input_path.name}'")
        
        image_files = list(input_path.glob('*'))
        count = 0
        for image_file in image_files:
            # Verifica se é um arquivo de imagem comum
            if image_file.suffix.lower() in ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']:
                processed_image = pad_and_resize(str(image_file), TARGET_SIZE)
                
                if processed_image is not None:
                    # Define o caminho de salvamento e salva a imagem
                    save_path = output_class_dir / image_file.name
                    cv2.imwrite(str(save_path), processed_image)
                    count += 1
        
        print(f"   ✅ Concluído: {count} imagens processadas e salvas em '{output_class_dir}'")

    print(f"\n🎉 Processo finalizado com sucesso!")
    print(f"   Todas as imagens foram salvas em '{OUTPUT_DIR}' com tamanho {TARGET_SIZE}x{TARGET_SIZE}.")

if __name__ == "__main__":
    main()