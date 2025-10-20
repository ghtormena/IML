import os
from PIL import Image

# Caminho da pasta com as imagens
input_dir = os.path.expanduser("~/Documents/IML2/Scripts/HEATMAPS/heatmap_test_9/Cranio Feminino")
# Caminho de saída do GIF
output_gif = os.path.join(os.path.dirname(input_dir), "cranio_feminino.gif")

# Lista todas as imagens (ordem alfabética)
images = [f for f in sorted(os.listdir(input_dir)) if f.lower().endswith((".png", ".jpg", ".jpeg"))]

if not images:
    print("⚠️ Nenhuma imagem encontrada na pasta.")
    exit()

# Carrega as imagens
frames = [Image.open(os.path.join(input_dir, img)) for img in images]

# Cria e salva o GIF
frames[0].save(
    output_gif,
    format="GIF",
    append_images=frames[1:],
    save_all=True,
    duration=100,  # duração por frame (ms)
    loop=0         # 0 = loop infinito
)

print(f"✅ GIF criado com sucesso: {output_gif}")