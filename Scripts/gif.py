import os
from PIL import Image

# --- Configuração ---

# Caminho para a pasta que contém as 4 pastas
base_dir = os.path.expanduser("~/Documents/IML2/Scripts/HEATMAPS/heatmap_test_9")

# Lista dos nomes das pastas que você quer processar
folder_names = ["Cranio Masculino", "Cranio Feminino", "Pelve Masculina", "Pelve Feminina"]

# Duração de cada frame no GIF (em milissegundos)
frame_duration = 100

# Cor de fundo (padding) para preencher o espaço extra (R, G, B)
# (0, 0, 0) = Preto
# (255, 255, 255) = Branco
background_color = (0, 0, 0)

# --- Fim da Configuração ---

print("Iniciando processo de criação de GIFs (com padding)...")

# Loop principal: executa o processo para cada pasta
for folder_name in folder_names:
    
    input_dir = os.path.join(base_dir, folder_name)
    # Define o nome do GIF de saída
    output_filename = folder_name.replace(" ", "_").lower() + ".gif"
    # Salva o GIF no diretório base (um nível acima das pastas de imagem)
    output_gif = os.path.join(base_dir, output_filename)

    print(f"\n--- Processando pasta: {folder_name} ---")

    if not os.path.isdir(input_dir):
        print(f"⚠️ Aviso: Pasta não encontrada. Pulando: {input_dir}")
        continue

    # 1. Lista todas as imagens
    try:
        image_files = [f for f in sorted(os.listdir(input_dir)) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
    except Exception as e:
        print(f"❌ Erro ao listar arquivos em {input_dir}: {e}")
        continue

    if not image_files:
        print("⚠️ Nenhuma imagem encontrada na pasta.")
        continue

    # 2. Carrega as imagens e encontra o tamanho máximo (max_w, max_h)
    max_w = 0
    max_h = 0
    original_images = []
    
    print(f"   Carregando {len(image_files)} imagens para encontrar dimensões...")
    try:
        for img_name in image_files:
            img_path = os.path.join(input_dir, img_name)
            img = Image.open(img_path)
            
            if img.width > max_w:
                max_w = img.width
            if img.height > max_h:
                max_h = img.height
                
            original_images.append(img)
            
    except Exception as e:
        print(f"❌ Erro ao carregar imagem: {e}")
        # Limpa o que foi carregado antes de pular
        for img in original_images:
            img.close()
        continue

    print(f"   Canvas do GIF definido para: {max_w} x {max_h} pixels.")

    # 3. Cria os novos frames com padding (fundo)
    frames = []
    print("   Criando frames com padding...")
    for img in original_images:
        # Cria o novo "fundo" com o tamanho máximo e a cor definida
        # Usamos "RGB" para garantir compatibilidade com GIF
        new_frame = Image.new("RGB", (max_w, max_h), background_color)
        
        # Calcula a posição para centralizar a imagem
        paste_x = (max_w - img.width) // 2
        paste_y = (max_h - img.height) // 2
        
        # Cola a imagem original no centro do novo fundo
        new_frame.paste(img, (paste_x, paste_y))
        frames.append(new_frame)

    # 4. Cria e salva o GIF
    try:
        print(f"   Salvando GIF em: {output_gif}")
        frames[0].save(
            output_gif,
            format="GIF",
            append_images=frames[1:],
            save_all=True,
            duration=frame_duration,
            loop=0  # 0 = loop infinito
        )
        print(f"✅ GIF criado com sucesso!")
    
    except Exception as e:
        print(f"❌ Erro ao salvar o GIF: {e}")
    
    finally:
        # 5. Limpa a memória (importante!)
        for img in original_images:
            img.close()
        for frame in frames:
            frame.close()

print("\n--- Processo concluído para todas as pastas. ---")