from ultralytics import YOLO

# Load a model
model = YOLO("yolo_models/yolo11m-cls.pt")

# Train the model com os parâmetros otimizados
results = model.train(
    data="/home/nexus/davi/IML/datasets/novo_dataset_2d/data_split", 
    epochs=300, 
    imgsz=640,
    batch=-1,
    
    # Parâmetros para desativar as augmentations
    hsv_h=0.0,
    hsv_s=0.0,
    hsv_v=0.0,
    degrees=0.0,
    translate=0.0,
    scale=0.0,
    shear=0.0,
    perspective=0.0,
    flipud=0.0,
    fliplr=0.0,
    bgr=0.0,
    mosaic=0.0,
    mixup=0.0,
    cutmix=0.0,
    copy_paste=0.0,
    erasing=0.0,
    auto_augment=None,
    
    # NOVAS ADIÇÕES PARA DESATIVAÇÃO MÁXIMA
    # 'rect=True' desativa o redimensionamento de 'aspect ratio' aleatório (jitter)
    rect=True, 
    
    # 'single_cls=False' para garantir que não haja manipulação de classes (embora para classificação isso seja menos relevante)
    single_cls=False 
)

print("✅ Treinamento concluído!")