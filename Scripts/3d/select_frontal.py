#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
select_frontal.py - Seleção manual de visualizações frontais (Imagens Pivotais)

Este script lê um arquivo de mapeamento manual (frontal_map.txt) onde cada linha 
contém o caminho da imagem PNG que representa a vista frontal (0°) de uma série.
Ele copia essa imagem e N imagens vizinhas (antes e depois) para um novo
diretório, preservando a estrutura de pastas.
"""

import os
import sys
import shutil
from pathlib import Path

# --- CONFIGURAÇÕES DO USUÁRIO ---
# Caminho da pasta de entrada (limpa, após a conversão DICOM e filtro de 150 imagens)
INPUT_PNG_ROOT = '/home/nexus/davi/3d/PNG'

# Caminho da pasta de saída para as visualizações frontais
OUTPUT_FRONTAL_ROOT = '/home/nexus/davi/3d/PNG_FRONTAL_MANUAL'

# Nome do arquivo de mapeamento que você irá preencher
MAP_FILE_NAME = 'frontal_map.txt'

# Número de imagens a incluir ANTES e DEPOIS da imagem frontal (pivô)
# (5 antes + 1 pivô + 5 depois = máximo de 11 imagens)
FRONTAL_VIEWS_WINDOW = 5
# --- FIM DAS CONFIGURAÇÕES DO USUÁRIO ---


def select_frontal_views_manual(source_root: Path, target_root: Path, window: int, map_file: Path):
    """
    Lê o arquivo de mapeamento, processa cada caminho pivô e copia a janela de imagens.
    """
    source_root = source_root.resolve()
    target_root = target_root.resolve()
    target_root.mkdir(parents=True, exist_ok=True)
    
    if not map_file.exists():
        print(f"[ERRO] Arquivo de mapeamento não encontrado: {map_file}")
        print("Crie e preencha este arquivo com os caminhos relativos das suas imagens frontais.")
        return 0

    with open(map_file, 'r') as f:
        # Lê caminhos, ignora linhas vazias e comentários (#)
        frontal_paths = [line.strip() for line in f if line.strip() and not line.startswith('#')]

    if not frontal_paths:
        print("[AVISO] O arquivo de mapeamento está vazio. Nenhuma imagem será copiada.")
        return 0
        
    copied_count = 0
    processed_series = 0
    
    print(f"[INFO] Lendo {len(frontal_paths)} caminhos pivotais de '{map_file.name}'...")
    print(f"[INFO] Janela de cópia definida para {window} imagens antes e depois do pivô.")

    for relative_path_str in frontal_paths:
        processed_series += 1
        
        # 1. Localiza a imagem frontal de referência
        # O caminho completo é: /home/nexus/davi/3d/PNG / Feminino/26F/.../pivo.png
        frontal_path = source_root / Path(relative_path_str)
        
        if not frontal_path.is_file():
            print(f"[AVISO] Arquivo pivô não encontrado: {relative_path_str}. Pulando.")
            continue
            
        source_dir = frontal_path.parent
        
        # 2. Lista, ordena todos os PNGs no diretório da série e encontra o índice
        png_files = sorted(list(source_dir.glob("*.png")))
        
        try:
            # Garante que o objeto Path que representa o pivô esteja na lista
            center_index = png_files.index(frontal_path)
        except ValueError:
            print(f"[ERRO] Imagem pivô {frontal_path.name} não encontrada na série. Pulando.")
            continue

        num_images = len(png_files)
        
        # 3. Define a janela de cópia (start, end)
        start_index = max(0, center_index - window)
        end_index = min(num_images, center_index + window + 1)
        files_to_copy = png_files[start_index:end_index]
        
        # 4. Define e cria o novo diretório de destino
        # Ex: /PNG_FRONTAL_MANUAL/Feminino/26F/...
        relative_series_path = source_dir.relative_to(source_root)
        target_dir = target_root / relative_series_path
        target_dir.mkdir(parents=True, exist_ok=True)
        
        # 5. Copia os arquivos
        for p in files_to_copy:
            shutil.copy2(p, target_dir / p.name)
            copied_count += 1
            
        print(f"  [COPIADO] Série {relative_series_path}: {len(files_to_copy)} imagens copiadas.")
        
    print(f"\n[INFO] Séries processadas (baseado na lista manual): {processed_series}")
    print(f"[INFO] Total de imagens frontais copiadas para {target_root}: {copied_count}")
    return copied_count

def main():
    try:
        source = Path(INPUT_PNG_ROOT)
        target = Path(OUTPUT_FRONTAL_ROOT)
        map_file = Path(MAP_FILE_NAME)
        
        if not source.is_dir():
            print(f"[ERRO] O diretório de entrada não existe: {source}", file=sys.stderr)
            sys.exit(1)
            
        select_frontal_views_manual(source, target, FRONTAL_VIEWS_WINDOW, map_file)
        
        print("\n[INFO] Seleção frontal concluída.")
        
    except Exception as e:
        print(f"[ERRO FATAL] Ocorreu um erro: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()