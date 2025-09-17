import os
import shutil
import re
from PIL import Image, ImageDraw, ImageOps

def limpar_pasta(pasta):
    if os.path.exists(pasta):
        shutil.rmtree(pasta)

def processar_e_salvar(src_path, dest_folder, espelhar=False, box_size=40):
    """
    Abre src_path, converte para RGB, desenha um quadrado preto 40x40
    no canto superior direito, salva em dest_folder com nome prefix_originalfilename,
    e então salva também a versão espelhada (da imagem com quadrado) com sufixo _espelhado.
    """
    try:
        img = Image.open(src_path).convert("RGB")
    except Exception as e:
        print(f"⚠️  Erro ao abrir {src_path}: {e}")
        return

    largura, altura = img.size
    
    # montar nomes
    arquivo = os.path.basename(src_path)
    nome_base, ext = os.path.splitext(arquivo)
    nome_dest = f"{arquivo}"              # ex: pasta_a.jpg -> 123M_a.jpg
    caminho_dest = os.path.join(dest_folder, nome_dest)

    # salva a versão com quadrado
    try:
        img.copy().save(caminho_dest)
    except Exception as e:
        print(f"⚠️  Erro ao salvar {caminho_dest}: {e}")
        return

    if espelhar:
        # cria e salva a versão espelhada (da imagem já com quadrado)
        try:
            img_espelhada = ImageOps.mirror(img.copy())
            nome_dest_esp = f"{nome_base}_espelhado{ext}"
            caminho_dest_esp = os.path.join(dest_folder, nome_dest_esp)
            img_espelhada.save(caminho_dest_esp)
        except Exception as e:
            print(f"⚠️  Erro ao salvar espelhada {caminho_dest_esp}: {e}")
        return

def organizar_amostras_com_quadrado(base_path):
    # Destinos (apenas estas 4 pastas serão criadas)
    cranio_masc = os.path.join(base_path, "Cranio Masculino")
    cranio_fem  = os.path.join(base_path, "Cranio Feminino")
    pelve_masc  = os.path.join(base_path, "Pelve Masculina")
    pelve_fem   = os.path.join(base_path, "Pelve Feminina")

    destinos = [cranio_masc, cranio_fem, pelve_masc, pelve_fem]

    # Remove/limpa e recria somente as quatro pastas de destino
    for d in destinos:
        limpar_pasta(d)
        os.makedirs(d, exist_ok=True)

    # Fonte: Amostras 2D/Masculino e Amostras 2D/Feminino
    amostras_root = os.path.join(base_path, "dataset_anonimizado_padded")
    if not os.path.isdir(amostras_root):
        print(f"⚠️  Pasta '{amostras_root}' não encontrada. Abortando.")
        return

    padrao = re.compile(r".*[MF]$")  # subpastas terminando em M ou F
    valid_names = {"a.jpg", "b.jpg", "c.jpg"}  # nomes esperados (comparação em lower())

    counts = {cranio_masc:0, cranio_fem:0, pelve_masc:0, pelve_fem:0}

    for genero in ["Masculino", "Feminino"]:
        genero_path = os.path.join(amostras_root, genero)
        if not os.path.isdir(genero_path):
            continue

        for arquivo in os.listdir(genero_path):
            low = arquivo.lower()
            src_file = os.path.join(genero_path, arquivo)
            if not os.path.isfile(src_file):
                continue

            if low.endswith("_a.jpg") or low.endswith("_b.jpg"):
                # cranio destino dependendo do genero
                dest = cranio_masc if genero == "Masculino" else cranio_fem
                processar_e_salvar(src_file, dest, espelhar=True)
                counts[dest] += 1

            elif low.endswith("_c.jpg"):
                dest = pelve_masc if genero == "Masculino" else pelve_fem
                processar_e_salvar(src_file, dest)
                counts[dest] += 1

    # relatório
    print("Organização + processamento concluídos.")
    for d in destinos:
        if d in [cranio_masc, cranio_fem]:
            print(f"  {os.path.basename(d)}: {counts[d]} imagens (cada uma gerou +1 espelhada).")
        else:
            print(f"  {os.path.basename(d)}: {counts[d]} imagens.")

if __name__ == "__main__":
    # executa no diretório atual (ajuste se quiser outro)
    base_path = os.path.expanduser(os.getcwd())
    organizar_amostras_com_quadrado(base_path)
