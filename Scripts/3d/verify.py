import os
import pydicom

def check_dicom_metadata(root_dir, num_files_to_check=1):
    """
    Percorre os diretórios e exibe os metadados de alguns arquivos DICOM.
    
    Args:
        root_dir (str): O diretório raiz a ser verificado.
        num_files_to_check (int): O número de arquivos a serem inspecionados.
    """
    print("Iniciando a verificação de metadados...")
    print(f"Verificando os metadados dos primeiros {num_files_to_check} arquivos encontrados.")
    print("-" * 50)
    
    files_checked = 0
    
    # Percorre todos os diretórios e subdiretórios
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if files_checked >= num_files_to_check:
                break
                
            # Tenta ler todos os arquivos
            filepath = os.path.join(dirpath, filename)
            try:
                # Lê o arquivo DICOM
                # force=True ajuda a ler arquivos que não têm o cabeçalho 'DICM'
                ds = pydicom.dcmread(filepath, force=True) 
                
                print(f"Arquivo: {filepath}")
                
                # --- Metadados Essenciais (Os que você já tinha) ---
                print(f"  PatientID (ID do Paciente): {ds.get('PatientID', 'Não Encontrado')}")
                print(f"  PatientName (Nome do Paciente): {ds.get('PatientName', 'Não Encontrado')}")
                print(f"  PatientSex (Sexo do Paciente): {ds.get('PatientSex', 'Não Encontrado')}")
                print(f"  StudyDate (Data do Estudo): {ds.get('StudyDate', 'Não Encontrado')}")
                print(f"  StudyDescription (Descrição do Estudo): {ds.get('StudyDescription', 'Não Encontrado')}")
                print(f"  BodyPartExamined (Parte do Corpo Examinada): {ds.get('BodyPartExamined', 'Não Encontrado')}")
                
                # --- Coleta de TODOS os Metadados (A nova funcionalidade) ---
                if files_checked == 0:
                    print("\n--- TODOS OS METADADOS (APENAS PARA O PRIMEIRO ARQUIVO) ---")
                    # Imprime a representação completa do dataset, que inclui todas as tags e seus valores
                    print(ds) 
                    print("----------------------------------------------------------\n")
                
                # --- Fim da Coleta de Metadados ---
                
                print("-" * 50)
                
                files_checked += 1
            except pydicom.errors.InvalidDicomError:
                # Silenciosamente ignora arquivos que não são DICOM
                pass
            except Exception as e:
                # Captura outros erros, como permissão ou corrupção
                print(f"Erro ao processar o arquivo {filepath}: {e}")
                print("-" * 50)

        if files_checked >= num_files_to_check:
            break
            
    if files_checked == 0:
        print("Nenhum arquivo DICOM válido encontrado no diretório especificado.")
    else:
        print("Verificação de metadados concluída.")

if __name__ == "__main__":
    # Define o diretório raiz a ser verificado
    # **AVISO**: Mantenha o caminho original apenas se ele for válido no seu ambiente.
    diretorio_raiz = '/home/nexus/davi/3d/NOVO_3D/Masculino/18M/01310000' 
    
    # Exemplo de uso: verifique 2 arquivos. 
    # O primeiro terá todos os metadados impressos, e o segundo, o formato resumido.
    # check_dicom_metadata(diretorio_raiz, num_files_to_check=2)
    
    # Usando o padrão de 1 arquivo para manter a compatibilidade
    check_dicom_metadata(diretorio_raiz)