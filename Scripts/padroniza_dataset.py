#!/usr/bin/env python3
import os
import shutil

def consolidar_pastas(base_dir, output_dir="Classes"):
    """
    Consolida os arquivos de A e B em W, X, Y e Z conforme a estrutura descrita.
    """

    # Caminhos base
    pasta_A = os.path.join(base_dir, "dataset_normalizado/Feminino")
    pasta_B = os.path.join(base_dir, "dataset_normalizado/Masculino")

    # Cria as pastas de destino
    destinos = {
        "Cranio Feminino": os.path.join(output_dir, "Cranio Feminino"),  # C de A
        "Pelve Feminina": os.path.join(output_dir, "Pelve Feminina"),  # D de A
        "Cranio Masculino": os.path.join(output_dir, "Cranio Masculino"),  # C de B
        "Pelve Masculina": os.path.join(output_dir, "Pelve Masculina"),  # D de B
    }

    for d in destinos.values():
        os.makedirs(d, exist_ok=True)

    # Função auxiliar para copiar todos os arquivos de subpastas
    def copiar_arquivos(origem_base, nome_subpasta, destino_base):
        if not os.path.isdir(origem_base):
            return
        for subdir in sorted(os.listdir(origem_base)):
            caminho_dir = os.path.join(origem_base, subdir, nome_subpasta)
            if os.path.isdir(caminho_dir):
                for root, _, files in os.walk(caminho_dir):
                    for arquivo in files:
                        caminho_arquivo = os.path.join(root, arquivo)
                        # Calcula o caminho relativo a partir de "origem_base/subdir"
                        relative_path = os.path.relpath(caminho_arquivo, os.path.join(origem_base, subdir, nome_subpasta))
                        destino_final = os.path.join(destino_base, subdir, os.path.basename(caminho_arquivo))
                        os.makedirs(os.path.dirname(destino_final), exist_ok=True)
                        shutil.copy2(caminho_arquivo, destino_final)

    # C de A -> W
    copiar_arquivos(pasta_A, "Crânio", destinos["Cranio Feminino"])
    # D de A -> X
    copiar_arquivos(pasta_A, "Pelve", destinos["Pelve Feminina"])
    # C de B -> Y
    copiar_arquivos(pasta_B, "Crânio", destinos["Cranio Masculino"])
    # D de B -> Z
    copiar_arquivos(pasta_B, "Pelve", destinos["Pelve Masculina"])

    print(f"✅ Consolidação concluída! Pastas criadas em: {os.path.abspath(output_dir)}")


if __name__ == "__main__":
    # Use o diretório atual como base por padrão
    base = os.getcwd()
    consolidar_pastas(base)
