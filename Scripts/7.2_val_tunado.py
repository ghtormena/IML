from ultralytics import YOLO
import numpy as np

# --- 1. CONFIGURAÇÕES ---
MODEL_PATH = '/home/nexus/davi/IML/runs/classify/train11/weights/best.pt'
DATASET_PATH = 'data.yaml'  # Certifique-se de que o caminho para o dataset está correto

# --- 2. CARREGAR O MODELO ---
model = YOLO(MODEL_PATH)

# --- 3. EXECUTAR A AVALIAÇÃO NO CONJUNTO DE TESTE ---
print("\n🧪 Iniciando a avaliação no conjunto de teste...")
metrics = model.val(
    data=DATASET_PATH,
    split='test',
    project='runs/evaluation',
    name='novo_nano_full_eval',
)

# --- 4. ANALISAR OS RESULTADOS ---
print("\n📊 Métricas de Desempenho no Conjunto de Teste:")

# Acurácia geral
print(f"  - Acurácia Top-1: {metrics.top1:.4f}")
print(f"  - Acurácia Top-5: {metrics.top5:.4f}")

# Obter a matriz de confusão
conf_matrix = metrics.confusion_matrix.matrix

# Nomes das classes (CORRIGIDO: buscar do objeto 'model')
class_names = model.names

# Calcular Precision, Recall e F1-score por classe
print("\n🔍 Métricas por Classe:")
epsilon = 1e-10

conf_matrix_real = conf_matrix[:-1, :-1]

TP = np.diag(conf_matrix_real)
FP = np.sum(conf_matrix_real, axis=0) - TP
FN = np.sum(conf_matrix_real, axis=1) - TP

precision_per_class = TP / (TP + FP + epsilon)
recall_per_class = TP / (TP + FN + epsilon)
f1_per_class = 2 * (precision_per_class * recall_per_class) / (precision_per_class + recall_per_class + epsilon)

# Note que model.names é um dicionário {index: 'name'}, então pegamos os valores
for i, class_name in enumerate(class_names.values()):
    print(f"  - Classe: {class_name}")
    print(f"    - Precision: {precision_per_class[i]:.4f}")
    print(f"    - Recall:    {recall_per_class[i]:.4f}")
    print(f"    - F1-score:  {f1_per_class[i]:.4f}")
    print("-" * 20)

# Calcular a média Macro
macro_precision = np.mean(precision_per_class)
macro_recall = np.mean(recall_per_class)
macro_f1 = np.mean(f1_per_class)

print("\n📈 Média das Métricas (Macro Avg):")
print(f"  - Precision (Média): {macro_precision:.4f}")
print(f"  - Recall (Média):    {macro_recall:.4f}")
print(f"  - F1-score (Média):  {macro_f1:.4f}")

# Caminho para a matriz de confusão salva
print(f"\n🖼️ A Matriz de Confusão foi salva em: {metrics.save_dir}/confusion_matrix.png")