import os
import shutil
import re

def carregar_alvos_txt(caminho_txt):
    """
    Lê o arquivo texto e retorna caminhos relativos normalizados:
      - Remove espaços e vírgulas finais.
      - Remove tudo antes de 'Masculino/' ou 'Feminino/' para padronizar.
    Exemplo de linha válida: '/X/Y/Z/M/dir1/C/imagem.png,'
    """
    alvos = []
    padrao_normalizar = re.compile(r'.*?/(Masculino|Feminino)/')  # captura o primeiro /Masculino/ ou /Feminino/

    with open(caminho_txt, 'r', encoding='utf-8') as f:
        for linha in f:
            linha = linha.strip().rstrip(',')
            if not linha or linha.startswith('#'):
                continue

            # Normaliza o caminho para começar em Masculino/ ou Feminino/
            m = padrao_normalizar.search(linha)
            if m:
                linha = linha[m.start(1):]  # mantém desde Masculino/ ou Feminino/
            else:
                print(f"⚠️ Caminho sem 'Masculino/' ou 'Feminino/': {linha}")
                continue

            alvos.append(linha)
    return alvos


def natural_key(text):
    """Ordena diretórios de forma natural (1, 2, 10 ao invés de 1, 10, 2)."""
    return [int(t) if t.isdigit() else t for t in re.split(r'(\d+)', text)]


def copiar_com_vizinhos(base_dir, output_dir, alvos, X):
    os.makedirs(output_dir, exist_ok=True)

    for alvo_rel in alvos:
        # Caminho completo para o arquivo alvo
        caminho_alvo = os.path.join(base_dir, alvo_rel)
        if not os.path.exists(caminho_alvo):
            print(f"⚠️ Arquivo alvo não encontrado: {caminho_alvo}")
            continue

        # Diretório onde está o arquivo
        dir_alvo = os.path.dirname(caminho_alvo)
        nome_alvo = os.path.basename(caminho_alvo)

        # Diretório de saída correspondente
        dir_relativo = os.path.relpath(dir_alvo, base_dir)
        dir_saida = os.path.join(output_dir, dir_relativo)
        os.makedirs(dir_saida, exist_ok=True)

        # Lista de arquivos no diretório, ordenada naturalmente
        arquivos = sorted(os.listdir(dir_alvo), key=natural_key)
        N = len(arquivos)
        if N == 0:
            print(f"⚠️ Diretório vazio: {dir_alvo}")
            continue

        if nome_alvo not in arquivos:
            print(f"⚠️ Arquivo '{nome_alvo}' não encontrado em {dir_alvo}")
            continue

        # Índice do arquivo alvo
        i = arquivos.index(nome_alvo)
        # Circularidade
        indices = [(i + j) % N for j in range(-X, X + 1)]

        for idx in indices:
            nome_arquivo = arquivos[idx]
            src = os.path.join(dir_alvo, nome_arquivo)
            dst = os.path.join(dir_saida, nome_arquivo)

            if not os.path.exists(dst):
                shutil.copy2(src, dst)

        print(f"✅ Copiado {2*X+1} arquivos em {dir_alvo} (alvo: {nome_alvo})")

    print("\n🎯 Cópia completa para todos os diretórios.")


def main():
    base_dir = "./All slices" # Caminho para o All slices
    output_dir = base_dir + "_modificado"
    arquivo_alvos = "./lista_pivos.txt" # Caminho para a lista de pivôs
    X = 10  # número de vizinhos antes/depois

    alvos = carregar_alvos_txt(arquivo_alvos)
    copiar_com_vizinhos(base_dir, output_dir, alvos, X)


# Execução direta
if __name__ == "__main__":
    main()
