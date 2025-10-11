#(miranha): código para converter imagens dicom em nifti

import SimpleITK as sitk
import os
import sys

def converter_serie_dicom(serie_dir, diretorio_saida):
    """
    Tenta converter uma única série DICOM. Se for bem-sucedido, salva como NIfTI.
    Se a pasta não contiver uma série válida, ignora silenciosamente.

    Args:
        serie_dir (str): O caminho para a pasta que pode conter os arquivos DICOM.
        diretorio_saida (str): A pasta principal onde todos os NIfTIs serão salvos.
    """
    reader = sitk.ImageSeriesReader()
    try:
        # Tenta obter os nomes dos arquivos da série DICOM na pasta
        dicom_names = reader.GetGDCMSeriesFileNames(serie_dir)
        if not dicom_names:
            # Se a lista estiver vazia, não é uma série DICOM válida.
            return

        reader.SetFileNames(dicom_names)
        image = reader.Execute()

        # ---- Lógica para criar um nome de arquivo único e informativo ----
        # Pega o nome da pasta da série (ex: "42100000")
        serie_id = os.path.basename(os.path.normpath(serie_dir))
        
        # Pega o nome da pasta do paciente (ex: "1F")
        caminho_paciente = os.path.dirname(serie_dir)
        paciente_id = os.path.basename(os.path.normpath(caminho_paciente))

        # Cria um nome de arquivo combinado para garantir que seja único
        nome_arquivo_saida = f"{paciente_id}_{serie_id}.nii.gz"
        caminho_arquivo_saida = os.path.join(diretorio_saida, nome_arquivo_saida)

        print(f"  [SUCESSO] Convertendo '{serie_dir}' para '{caminho_arquivo_saida}'")
        sitk.WriteImage(image, caminho_arquivo_saida)

    except Exception as e:
        # Se ocorrer qualquer erro (ex: pasta com arquivos misturados), informa e continua
        print(f"  [AVISO] Não foi possível processar '{serie_dir}'. Não parece ser uma série DICOM válida. Erro: {e}")

def main(diretorio_raiz, diretorio_saida):
    """
    Função principal que varre o diretório raiz em busca de séries DICOM para converter.

    Args:
        diretorio_raiz (str): A pasta principal para iniciar a busca (ex: 'data/Feminino_separado/transversal').
        diretorio_saida (str): A pasta onde todos os arquivos NIfTI convertidos serão salvos.
    """
    # Cria o diretório de saída, se ele não existir
    os.makedirs(diretorio_saida, exist_ok=True)
    print(f"Iniciando a varredura em: '{diretorio_raiz}'")
    print(f"Os arquivos NIfTI serão salvos em: '{diretorio_saida}'")
    print("-" * 30)

    # os.walk() é perfeito para isso: ele percorre a árvore de diretórios
    # 'root' é a pasta atual, 'dirs' são as subpastas, 'files' são os arquivos
    for root, dirs, files in os.walk(diretorio_raiz):
        # Uma boa heurística: uma pasta de série DICOM geralmente contém arquivos, mas não outras subpastas.
        if files and not dirs:
            # Encontramos uma pasta que parece ser uma série DICOM. Tentamos convertê-la.
            converter_serie_dicom(root, diretorio_saida)

    print("-" * 30)
    print("Processo de conversão em lote concluído!")


if __name__ == "__main__":
    # O script espera 2 argumentos: <pasta_de_entrada> <pasta_de_saida>
    if len(sys.argv) != 3:
        print("Uso: python scripts/converter_lote.py <caminho_para_pasta_de_entrada> <caminho_para_pasta_de_saida>")
        print("\nExemplo:")
        print("python scripts/converter_lote.py data/Feminino_separado/transversal nifti_output/feminino_transversal")
        sys.exit(1)

    input_dir = sys.argv[1]
    output_dir = sys.argv[2]

    if not os.path.isdir(input_dir):
        print(f"ERRO: O diretório de entrada '{input_dir}' não foi encontrado.")
        sys.exit(1)

    main(input_dir, output_dir)
