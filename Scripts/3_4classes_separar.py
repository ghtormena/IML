import os
import shutil

# --- CONFIGURAÇÕES ---
# 1. Defina o caminho para a pasta principal que contém os diretórios 'Masculino' e 'Feminino'.
#    Exemplo: 'caminho/para/meu/dataset'
INPUT_DIR_BASE = "/home/nexus/davi/IML/datasets/novo_dataset_2d/anonimizado_padded"

# 2. Defina o nome da pasta de saída onde os novos diretórios serão criados.
#    Exemplo: 'caminho/para/meu/dataset_organizado'
OUTPUT_DIR_BASE = "/home/nexus/davi/IML/datasets/novo_dataset_2d/anonimizado_padded_separado"

# --- LÓGICA PRINCIPAL DO SCRIPT ---

def organize_images():
    """
    Varre os diretórios de entrada, organiza as imagens por tipo (crânio ou pelve)
    e move-as para os diretórios de saída correspondentes.
    """
    print("🚀 Iniciando a organização das imagens...")

    # Cria o diretório de saída se ele não existir
    if not os.path.exists(OUTPUT_DIR_BASE):
        os.makedirs(OUTPUT_DIR_BASE)

    # Define os novos diretórios de saída
    output_dirs = {
        'Masculino_cranio': os.path.join(OUTPUT_DIR_BASE, 'Masculino_cranio'),
        'Masculino_pelve': os.path.join(OUTPUT_DIR_BASE, 'Masculino_pelve'),
        'Feminino_cranio': os.path.join(OUTPUT_DIR_BASE, 'Feminino_cranio'),
        'Feminino_pelve': os.path.join(OUTPUT_DIR_BASE, 'Feminino_pelve')
    }

    # Cria todos os diretórios de saída
    for path in output_dirs.values():
        if not os.path.exists(path):
            os.makedirs(path)

    # Processa os diretórios 'Masculino' e 'Feminino'
    genders = ['Masculino', 'Feminino']
    
    for gender in genders:
        input_path = os.path.join(INPUT_DIR_BASE, gender)
        
        if not os.path.isdir(input_path):
            print(f"⚠️ Aviso: Diretório '{input_path}' não encontrado. Pulando...")
            continue
            
        print(f"\n➡️ Processando diretório: '{gender}'")
        
        # Lista todos os arquivos no diretório atual
        files_to_process = os.listdir(input_path)
        
        if not files_to_process:
            print(f"  - Aviso: Diretório '{input_path}' está vazio. Nenhum arquivo para processar.")
            continue

        count = 0
        for filename in files_to_process:
            # Verifica se é um arquivo e se a extensão é de imagem
            if os.path.isfile(os.path.join(input_path, filename)):
                
                # Obtém a parte final do nome do arquivo antes da extensão (ex: '1M_a')
                name_without_ext = os.path.splitext(filename)[0]
                
                # Obtém a letra que define a parte do corpo (ex: 'a', 'b', 'c')
                part_identifier = name_without_ext[-1] # Pega o último caractere do nome

                # Define o diretório de destino
                destination_dir = None
                
                # Lógica para determinar o destino da imagem
                if gender == 'Masculino':
                    if part_identifier in ['a', 'b', 'A', 'B']:
                        destination_dir = output_dirs['Masculino_cranio']
                    elif part_identifier in ['c', 'C']:
                        destination_dir = output_dirs['Masculino_pelve']
                elif gender == 'Feminino':
                    if part_identifier in ['a', 'b', 'A', 'B']:
                        destination_dir = output_dirs['Feminino_cranio']
                    elif part_identifier in ['c', 'C']:
                        destination_dir = output_dirs['Feminino_pelve']
                
                # Move o arquivo se o destino foi definido
                if destination_dir:
                    source_path = os.path.join(input_path, filename)
                    destination_path = os.path.join(destination_dir, filename)
                    shutil.move(source_path, destination_path)
                    count += 1
        
        print(f"   ✅ Concluído: {count} imagens movidas de '{gender}'.")

    print("\n🎉 Organização finalizada com sucesso!")
    print(f"As imagens foram organizadas em '{OUTPUT_DIR_BASE}'.")

if __name__ == "__main__":
    organize_images()