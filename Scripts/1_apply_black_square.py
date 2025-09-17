import cv2
import os
import shutil
from pathlib import Path

# --- CONFIGURAÇÕES ---
# 1. Liste os diretórios com as imagens originais que você quer processar.
INPUT_DIRS = [
    "dataset_original/2d/2d/Amostras 2D/Feminino",
    "dataset_original/2d/2d/Amostras 2D/Masculino"
]

# 2. Defina o nome da pasta de saída para as imagens modificadas.
OUTPUT_DIR = "dataset_anonimizado"

# 3. Defina o tamanho do quadrado preto a ser aplicado.
SQUARE_SIZE = 50

# --- FUNÇÃO PRINCIPAL DE PROCESSAMENTO ---

def apply_black_square(image_path: str, size: int) -> 'numpy.ndarray | None':
    """
    Lê uma imagem e aplica um quadrado preto no canto superior direito.

    Args:
        image_path (str): Caminho para a imagem de entrada.
        size (int): O lado do quadrado preto em pixels.

    Returns:
        numpy.ndarray | None: A imagem modificada ou None se ocorrer um erro.
    """
    try:
        # Lê a imagem
        img = cv2.imread(image_path)
        if img is None:
            print(f"  - Aviso: Não foi possível ler a imagem {image_path}")
            return None

        h, w, _ = img.shape

        # Define as coordenadas do canto superior direito
        # Garante que o quadrado não saia dos limites de imagens pequenas
        y_start = 0
        y_end = min(size, h)
        x_start = max(0, w - size)
        x_end = w

        # Usa a indexação do NumPy para selecionar a região e torná-la preta
        # A cor (0, 0, 0) é preta no formato BGR do OpenCV
        img[y_start:y_end, x_start:x_end] = (0, 0, 0)

        return img
    except Exception as e:
        print(f"  - Erro ao processar {image_path}: {e}")
        return None

# --- LÓGICA PRINCIPAL DO SCRIPT ---

def main():
    """
    Executa o processo de aplicar o quadrado preto para todos os diretórios.
    """
    output_path = Path(OUTPUT_DIR)
    
    # Limpa o diretório de saída se ele já existir
    if output_path.exists():
        print(f"⚠️  Diretório de saída '{output_path}' já existe. Removendo...")
        shutil.rmtree(output_path)
    
    print(f"🚀 Iniciando processo para remover metadados visuais...")

    for dir_path_str in INPUT_DIRS:
        input_path = Path(dir_path_str)
        if not input_path.is_dir():
            print(f"  - Aviso: Diretório de entrada '{input_path}' não encontrado. Pulando.")
            continue
            
        # Cria a estrutura de pastas correspondente no diretório de saída
        output_class_dir = output_path / input_path.name
        output_class_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\n➡️  Processando diretório: '{input_path.name}'")
        
        image_files = list(input_path.rglob('*'))
        count = 0
        for image_file in image_files:
            if image_file.suffix.lower() in ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']:
                modified_image = apply_black_square(str(image_file), SQUARE_SIZE)
                
                if modified_image is not None:
                    # Define o caminho de salvamento e salva a imagem modificada
                    prefix = image_file.parent.name                # nome da subpasta
                    new_name = f"{prefix}_{image_file.name}"       # ex: pasta1_img1.jpg
                    save_path = output_class_dir / new_name
                    cv2.imwrite(str(save_path), modified_image)
                    count += 1
        
        print(f"   ✅ Concluído: {count} imagens processadas e salvas em '{output_class_dir}'")

    print(f"\n🎉 Processo finalizado com sucesso!")
    print(f"   As imagens com o canto superior direito coberto foram salvas em '{OUTPUT_DIR}'.")

if __name__ == "__main__":
    main()