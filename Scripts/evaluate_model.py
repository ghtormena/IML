from ultralytics import YOLO
import torch

# --- 1. CONFIGURAÇÕES ---

# Caminho para os pesos do seu melhor modelo
MODEL_PATH = '/home/nexus/Documents/IML/Scripts/runs/classify/train4/weights/best.pt' # IMPORTANTE: Verifique se este caminho está correto!

# Caminho para o diretório principal do seu dataset
DATASET_PATH = 'dataset_yolo'

# --- 2. CARREGAR O MODELO ---

# Carrega o modelo treinado a partir do arquivo .pt

model = YOLO(MODEL_PATH)

# --- 3. EXECUTAR A AVALIAÇÃO NO CONJUNTO DE TESTE ---

print("\n🧪 Iniciando a avaliação no conjunto de teste...")

# A função .val() executa a avaliação.
# Usamos o argumento split='test' para dizer ao YOLO para usar a pasta 'test'.
metrics = model.val(
    data=DATASET_PATH,
    split='test',
    project='runs/evaluation', # Salva os resultados em uma nova pasta
    name='test_results_med'      # Nome do subdiretório para esta avaliação
)

print("\n✅ Avaliação concluída!")

# --- 4. ANALISAR OS RESULTADOS ---

print("\n📊 Métricas de Desempenho no Conjunto de Teste:")

# A 'metrics' é um objeto que contém todos os resultados.
# Para classificação, as principais métricas são top1 e top5 accuracy.
print(f"  - Acurácia Top-1 (precisão da melhor previsão): {metrics.top1:.4f}")
print(f"  - Acurácia Top-5 (precisão das 5 melhores previsões): {metrics.top5:.4f}")

# O YOLO também salva uma matriz de confusão, que é excelente para análise de erros.
print(f"\n📈 A Matriz de Confusão foi salva em: {metrics.save_dir}/confusion_matrix.png")
print(f"   Use a matriz para ver quais classes o modelo está confundindo.")