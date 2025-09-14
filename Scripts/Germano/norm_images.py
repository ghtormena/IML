import sys
from pathlib import Path
from PIL import Image, ImageOps

def resize_with_padding(img: Image.Image, size=(512, 512), color=(0,0,0)):
    img = img.convert("RGB")
    img.thumbnail(size, Image.Resampling.LANCZOS)  # mantém proporção
    dw, dh = size[0] - img.width, size[1] - img.height
    padding = (dw//2, dh//2, dw - dw//2, dh - dh//2)
    return ImageOps.expand(img, padding, fill=color)

def process_dir(src_dir, dst_dir, size=(512,512), color=(0,0,0)):
    src_dir, dst_dir = Path(src_dir), Path(dst_dir)

    # cria o diretório de saída sempre, mesmo se não existir
    dst_dir.mkdir(parents=True, exist_ok=True)

    exts = {".png",".jpg",".jpeg",".bmp",".tif",".tiff"}

    for p in src_dir.glob("*"):  # apenas arquivos da pasta raiz
        if p.suffix.lower() in exts:
            outp = dst_dir / p.name  # salva no novo diretório
            im = Image.open(p)
            out = resize_with_padding(im, size=size, color=color)
            if outp.suffix.lower() in [".jpg", ".jpeg"]:
                out.save(outp, quality=95)
            else:
                out.save(outp)

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Uso: python normalize_images_flat.py <pasta_entrada> <pasta_saida> ")
        sys.exit(1)

    src, dst = sys.argv[1], sys.argv[2]
    side = int(sys.argv[3]) if len(sys.argv) > 3 else 512

    process_dir(src, dst, size=(side, side), color=(0,0,0))
    print(f"✅ Imagens normalizadas foram salvas em: {dst}")
