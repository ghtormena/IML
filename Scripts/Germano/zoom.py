import sys
from pathlib import Path
from PIL import Image

def resize_with_zoom(img: Image.Image, size=(512, 512), extra_zoom=1.0):
    """
    extra_zoom > 1.0 -> aplica mais zoom (corta mais bordas)
    """
    img = img.convert("RGB")
    w, h = img.size
    tw, th = size

    # escala para cobrir o alvo
    scale = max(tw / w, th / h) * extra_zoom
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)

    # crop central
    left = (new_w - tw) // 2
    top = (new_h - th) // 2
    right = left + tw
    bottom = top + th
    return img.crop((left, top, right, bottom))

def process_dir(src_dir, dst_dir, size=(512,512), extra_zoom=1.0):
    src_dir, dst_dir = Path(src_dir), Path(dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)
    exts = {".png",".jpg",".jpeg",".bmp",".tif",".tiff"}

    for p in src_dir.glob("*"):
        if p.suffix.lower() in exts:
            outp = dst_dir / p.name
            im = Image.open(p)
            out = resize_with_zoom(im, size=size, extra_zoom=extra_zoom)
            if outp.suffix.lower() in [".jpg", ".jpeg"]:
                out.save(outp, quality=95)
            else:
                out.save(outp)

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Uso: python normalize_zoom_crop.py <pasta_entrada> <pasta_saida> ")
        sys.exit(1)

    src, dst = sys.argv[1], sys.argv[2]
    side = int(sys.argv[3]) if len(sys.argv) > 3 else 512
    extra = float(sys.argv[4]) if len(sys.argv) > 4 else 1.2

    process_dir(src, dst, size=(side, side), extra_zoom=extra)
    print(f"✅ Imagens com zoom e crop central salvas em: {dst}")
