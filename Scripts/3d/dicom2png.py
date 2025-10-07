#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
dicom2png2.py — DICOM -> PNG com filtros anti-lixo + separação por estudo/série

Novidades:
- --split-by {study,series,series+bodypart}  (default: series)
  Cria subpastas de saída como:
    study:   <StudyDate>__<StudyDesc>__<StudyUID6>
    series:  <StudyDate>__S<SeriesNum>__<SeriesDesc>__<SeriesUID6>
    series+bodypart: idem series, adicionando __<BodyPart>
- Nomes "slug" seguros (sem espaços/especiais).
- Mantém todas as opções anteriores (--only-mods, --frames, --clahe, etc).

Requisitos:
  pip install pydicom pillow numpy
  (opcional) pip install opencv-python
"""

import argparse, os, sys, concurrent.futures as cf, re
from pathlib import Path
import numpy as np
from PIL import Image
import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut

# --- CONFIGURAÇÕES DO USUÁRIO ---
# Defina seus caminhos e opções aqui
# Substitua pelos seus caminhos reais!
CODED_INPUT_PATH = '/home/nexus/davi/3d/3D'
CODED_OUTPUT_PATH = '/home/nexus/davi/3d/PNG'

CODED_OPTIONS = {
    "split_by": "series",       # Como organizar: study, series, series+bodypart
    "drop_localizer": True,     # Filtrar imagens localizadoras/scout
    "clahe": True,              # Aplicar melhoria de contraste (requer opencv-python)
    "frames": "all",         # Qual quadro salvar (first, middle, last, all)
    "max_size": 1024,           # Redimensionar o lado maior (0 para desativar)
    "workers": 4,               # Número de threads
    "min_entropy": 1.5,         # Filtrar imagens muito escuras/uniformes
    "drop_derived": False,
    # ADICIONE ESTAS QUATRO LINHAS (Parâmetros de Qualidade/Filtro):
    "min_nonblack_pct": 0.05, # Mínimo de pixels não-pretos (evita imagens totalmente pretas)
    "min_p2p": 10,            # Mínimo contraste (Peak-to-Peak)
    "min_var": 50,            # Mínima variância do pixel (evita imagens uniformes)
    
    # ADICIONE ESTA LINHA (Configuração de Cor):
    "rgb": True,             # Se deve forçar imagens de saída para RGB (False = monocromático, se a fonte for)
    "only_mods": "",
}
# --- FIM DAS CONFIGURAÇÕES DO USUÁRIO ---

# ---------- util ----------
def slugify(s: str, maxlen=64):
    if not s:
        return "NA"
    s = str(s)
    s = re.sub(r"[^\w\-\.]+", "_", s, flags=re.UNICODE)  # troca espaços/especiais por _
    s = re.sub(r"_+", "_", s).strip("_")
    return s[:maxlen] if maxlen else s

def short_uid(u):
    u = str(u or "")
    return u[-6:] if len(u) >= 6 else u

# ---------- filtros/normalização ----------
try:
    import cv2
    HAS_CV2 = True
except Exception:
    HAS_CV2 = False

LOCALIZER_TOKENS = {"LOCALIZER","SCOUT"}
NOT_IMAGE_SOP_PREFIXES = {
    "1.2.840.10008.5.1.4.1.1.66",
    "1.2.840.10008.5.1.4.1.1.88",
    "1.2.840.10008.5.1.4.1.1.481",
    "1.2.840.10008.5.1.4.1.1.104",
}

def looks_like_non_image(ds):
    if "PixelData" not in ds: return True
    sop = str(getattr(ds,"SOPClassUID","") or "")
    return any(sop.startswith(pref) for pref in NOT_IMAGE_SOP_PREFIXES)

def is_localizer_or_derived(ds):
    imgtype = [t.upper() for t in (ds.get("ImageType",[]) or [])]
    return any(t in imgtype for t in LOCALIZER_TOKENS), ("DERIVED" in imgtype)

def modality_allowed(ds, only_mods):
    if not only_mods: return True
    return str(getattr(ds,"Modality","") or "").upper() in only_mods

def load_pixels(ds):
    arr = ds.pixel_array
    slope = float(getattr(ds,"RescaleSlope",1.0) or 1.0)
    inter = float(getattr(ds,"RescaleIntercept",0.0) or 0.0)
    if slope!=1.0 or inter!=0.0:
        arr = arr.astype(np.float32)*slope + inter
    try:
        arr = apply_voi_lut(arr, ds)
    except Exception:
        pass
    if str(getattr(ds,"PhotometricInterpretation","")).upper()=="MONOCHROME1":
        arr = arr.max()-arr
    return arr

def to_uint8(img):
    def scale(x):
        lo,hi = np.percentile(x,2), np.percentile(x,98)
        if hi<=lo:
            lo,hi = float(np.min(x)), float(np.max(x))
        if hi<=lo:
            hi = lo + 1.0
        y = np.clip((x-lo)/(hi-lo),0,1)
        return (y*255.0 + 0.5).astype(np.uint8)
    if img.ndim==2:
        return scale(img)
    if img.ndim==3:
        return np.stack([scale(img[...,c]) for c in range(img.shape[2])],axis=2)
    raise ValueError("Dimensão não suportada para to_uint8")

def ensure_rgb(u8, force_rgb):
    if u8.ndim==2 and force_rgb:
        return np.stack([u8,u8,u8],axis=2)
    return u8

def apply_clahe(u8, use):
    if not use or not HAS_CV2:
        return u8
    if u8.ndim==2:
        clahe=cv2.createCLAHE(clipLimit=2.0,tileGridSize=(8,8))
        return clahe.apply(u8)
    lab=cv2.cvtColor(u8,cv2.COLOR_RGB2LAB)
    l,a,b=cv2.split(lab)
    clahe=cv2.createCLAHE(clipLimit=2.0,tileGridSize=(8,8))
    l2=clahe.apply(l)
    return cv2.cvtColor(cv2.merge([l2,a,b]), cv2.COLOR_LAB2RGB)

def entropy(u8):
    g=u8 if u8.ndim==2 else (0.299*u8[...,0]+0.587*u8[...,1]+0.114*u8[...,2]).astype(np.uint8)
    hist=np.bincount(g.ravel(),minlength=256).astype(float)
    s=hist.sum()
    if s<=0: return 0.0
    p=hist/s; p=p[p>0]
    return float(-(p*np.log2(p)).sum())

def nonblack(u8,th=5):
    g=u8 if u8.ndim==2 else (0.299*u8[...,0]+0.587*u8[...,1]+0.114*u8[...,2]).astype(np.uint8)
    return float(np.mean(g>th))

def contrast(u8):
    g=u8 if u8.ndim==2 else (0.299*u8[...,0]+0.587*u8[...,1]+0.114*u8[...,2]).astype(np.uint8)
    return int(g.max()-g.min())

def passes(u8, a,b,c,d):
    return (nonblack(u8)>=a and entropy(u8)>=b and contrast(u8)>=c and float(np.var(u8.astype(np.float32)))>=d)

def resize_if_needed(u8,maxsz):
    if not maxsz or max(u8.shape[:2])<=maxsz: return u8
    scale = maxsz/float(max(u8.shape[:2]))
    new_w,new_h = max(1,int(round(u8.shape[1]*scale))), max(1,int(round(u8.shape[0]*scale)))
    if HAS_CV2:
        return cv2.resize(u8,(new_w,new_h),interpolation=cv2.INTER_AREA if scale<1 else cv2.INTER_LINEAR)
    im = Image.fromarray(u8)
    im = im.resize((new_w,new_h), resample=Image.Resampling.LANCZOS)
    return np.asarray(im)

def save_png(u8, path:Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(u8 if u8.ndim==2 else u8[...,:3]).save(path)

def choose_frames(arr, which):
    # arr: (H,W), (F,H,W) ou (F,H,W,C)
    if arr.ndim==2 or (arr.ndim==3 and arr.shape[-1] in (3,4)):
        return [arr]
    if arr.ndim==3:
        F=arr.shape[0]
        if which=="first": return [arr[0]]
        if which=="last":  return [arr[-1]]
        if which=="middle":return [arr[F//2]]
        return [arr[i] for i in range(F)]
    if arr.ndim==4:
        F=arr.shape[0]
        if which=="first": return [arr[0]]
        if which=="last":  return [arr[-1]]
        if which=="middle":return [arr[F//2]]
        return [arr[i] for i in range(F)]
    return [np.squeeze(arr)]

# ---------- separação por estudo/série ----------
def out_subdir_for(ds, split_by: str):
    study_uid   = getattr(ds, "StudyInstanceUID", None)
    series_uid  = getattr(ds, "SeriesInstanceUID", None)
    study_date  = getattr(ds, "StudyDate", "") or ""
    study_desc  = getattr(ds, "StudyDescription", "") or ""
    series_num  = getattr(ds, "SeriesNumber", None)
    series_desc = getattr(ds, "SeriesDescription", "") or ""
    body_part   = getattr(ds, "BodyPartExamined", "") or ""

    # Normaliza strings
    date_s   = slugify(study_date, maxlen=16)
    sdesc_s  = slugify(study_desc, maxlen=48)
    srsdesc  = slugify(series_desc, maxlen=48)
    body_s   = slugify(body_part, maxlen=24)
    snum_s   = f"S{int(series_num):03d}" if str(series_num).isdigit() else "SNA"
    suid6    = short_uid(study_uid)
    seuid6   = short_uid(series_uid)

    if split_by == "study":
        # <StudyDate>__<StudyDesc>__<StudyUID6>
        base = f"{date_s}__{sdesc_s}__{suid6}"
    elif split_by == "series":
        # <StudyDate>__S<SeriesNum>__<SeriesDesc>__<SeriesUID6>
        base = f"{date_s}__{snum_s}__{srsdesc}__{seuid6}"
    elif split_by == "series+bodypart":
        base = f"{date_s}__{snum_s}__{srsdesc}__{body_s}__{seuid6}"
    else:
        base = f"{date_s}__{snum_s}__{srsdesc}__{seuid6}"
    return slugify(base, maxlen=120) or "UNSORTED"

# ---------- pipeline ----------

from pathlib import Path  # Ensure Path is imported if you're pasting this function alone

from pathlib import Path  # Ensure Path is imported at the top of your script

def worker(p, rel, args, only_mods, drop_derived, subfolder_name):
    try:
        # Tenta ler o arquivo DICOM
        ds=pydicom.dcmread(str(p), force=True)
    except Exception:
        return 0

    # Checagens iniciais de validade
    if looks_like_non_image(ds): return 0
    if not modality_allowed(ds, only_mods): return 0
    
    # Verifica se a imagem é localizador (loc) ou derivada (der)
    loc, der = is_localizer_or_derived(ds)
    
    # ----------------------------------------------------
    # FILTRO MODIFICADO: Ignora APENAS localizadores (se drop_localizer=True)
    #                   Processa imagens Primárias (não-DERIVED) e Derivadas.
    # ----------------------------------------------------
    # 1. Ignora se for localizador E o usuário pediu para ignorar localizadores
    if args.drop_localizer and loc: 
        return 0
        
    # **A LINHA DE FILTRO 'if not der' FOI REMOVIDA AQUI**
    # Agora, se a imagem for primária (not der) e não for localizador, ela continua.
    
    # O filtro original 'if drop_derived and der' também é inativo
    # pois você definiu drop_derived: False nas CODED_OPTIONS.
    # ----------------------------------------------------

    try:
        # Carrega e aplica VOI LUT/Windowing/Rescale
        arr = load_pixels(ds)
    except Exception:
        return 0

    # ... (o resto da função worker permanece inalterado) ...
    sub_metadata = out_subdir_for(ds, args.split_by)

    try:
        rel_parts = p.relative_to(Path(args.input).resolve()).parts
        
        if len(rel_parts) >= 2:
            path_to_keep = Path(rel_parts[0]) / rel_parts[1]
        elif len(rel_parts) == 1:
            path_to_keep = Path(rel_parts[0])
        else:
            path_to_keep = Path(".")
            
    except Exception:
        path_to_keep = Path(".")
        
    saved = 0
    for i, fr in enumerate(choose_frames(arr, args.frames)):
        u8 = to_uint8(fr)
        u8 = ensure_rgb(u8, args.rgb)
        u8 = apply_clahe(u8, args.clahe)
        u8 = resize_if_needed(u8, args.max_size)

        if not passes(u8, args.min_nonblack_pct, args.min_entropy, args.min_p2p, args.min_var):
            continue

        out_dir = Path(args.output).resolve() / path_to_keep / sub_metadata
        
        name = p.stem + (f"_f{i:04d}" if i>0 else "")
        save_png(u8, out_dir / f"{name}.png")
        saved += 1
    return saved

def parse_args():
    # Isso simula a leitura do argparse, mas usa os valores CODED_OPTIONS
    class Args:
        pass
    
    args = Args()
    
    # Define as propriedades obrigatórias (Input/Output)
    args.input = CODED_INPUT_PATH
    args.output = CODED_OUTPUT_PATH
    
    # Define as demais propriedades usando CODED_OPTIONS
    for key, value in CODED_OPTIONS.items():
        setattr(args, key, value)
        
    # Adiciona as opções que não estão no CODED_OPTIONS mas são necessárias (ou inversas)
    args.keep_tree = False
    # Garante que 'keep_derived' seja o oposto de 'drop_derived' ou True
    args.keep_derived = not args.drop_derived
    
    # Se você quiser que o script imprima os valores que está usando:
    # print("[INFO] Usando configurações codificadas:")
    # for attr in sorted(dir(args)):
    #     if not attr.startswith('__') and not callable(getattr(args, attr)):
    #         print(f"  {attr}: {getattr(args, attr)}")
            
    return args

def main():
    args=parse_args()
    in_root=Path(args.input).resolve()
    out_root=Path(args.output).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    if not in_root.exists():
        print(f"[ERRO] Input não existe: {in_root}", file=sys.stderr); sys.exit(1)

    only_mods = set(m.strip().upper() for m in args.only_mods.split(",") if m.strip()) if args.only_mods else None
    drop_derived = args.drop_derived and not args.keep_derived

    # Itera sobre subfolders do input
    subfolders = [f for f in in_root.iterdir() if f.is_dir()]
    if not subfolders:
        print(f"[ERRO] Nenhum subfolder encontrado em {in_root}", file=sys.stderr)
        sys.exit(1)

    total = 0
    for subfolder in subfolders:
        files = [p for p in subfolder.rglob("*") if p.is_file()]
        print(f"[INFO] {len(files)} candidatos encontrados em {subfolder}")
        def pack(p):
            try:
                rel = p.parent.relative_to(subfolder) if args.keep_tree else Path(".")
            except Exception:
                rel = Path(".")
            return worker(p, rel, args, only_mods, drop_derived, subfolder.name)
        with cf.ThreadPoolExecutor(max_workers=max(1,int(args.workers))) as ex:
            for n in ex.map(pack, files):
                total += int(n or 0)

    print(f"[INFO] Imagens salvas: {total}")
    print("[INFO] Concluído.")

if __name__=="__main__":
    main()