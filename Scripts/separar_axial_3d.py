#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, sys, argparse, shutil, unicodedata
from collections import defaultdict
import numpy as np
import re
import pydicom
from pydicom.errors import InvalidDicomError

AXIAL_THR = 0.85  # quão “próximo do eixo Z” para considerar axial

def norm_txt(s: str) -> str:
    if not s:
        return ""
    s = unicodedata.normalize("NFKD", s).encode("ASCII", "ignore").decode("ASCII")
    return " ".join(s.lower().split())

def read_meta_fast(path):
    try:
        ds = pydicom.dcmread(path, stop_before_pixels=True, force=True)
        _ = ds.get((0x0008, 0x0016))  # SOPClassUID (sanity check)
        return ds, None
    except Exception as e:
        return None, str(e)

def series_key(ds):
    return getattr(ds, "SeriesInstanceUID", None) or f"NO_SERIES__{id(ds)}"

def collect_text(ds):
    parts = [
        norm_txt(getattr(ds,"SeriesDescription","")),
        norm_txt(getattr(ds,"ProtocolName","")),
        norm_txt(getattr(ds,"StudyDescription","")),
        norm_txt(" ".join(getattr(ds,"ImageType",[]) if isinstance(getattr(ds,"ImageType",[]),(list,tuple)) else [str(getattr(ds,"ImageType",""))]))
    ]
    return " ".join(p for p in parts if p)

def plane_axial(ds) -> bool | None:
    iop = getattr(ds, "ImageOrientationPatient", None)
    if not iop or len(iop) < 6:
        return None
    try:
        r = np.array([float(iop[0]),float(iop[1]),float(iop[2])], dtype=float)
        c = np.array([float(iop[3]),float(iop[4]),float(iop[5])], dtype=float)
        n = np.cross(r, c); nn = np.linalg.norm(n)
        if nn < 1e-6: return None
        n = n / nn
        return abs(n[2]) >= AXIAL_THR  # normal ~ Z  ⇒ axial/transversal
    except Exception:
        return None

# 3D “de verdade”: VRT/MIP/VR/RADIAL/CSAMANIPULATED, “vtr/vol/range/3d”
TOK_3D_STRONG = {"VRT", "VR", "MIP", "SSD", "SURFACE", "RADIAL", "CSAMANIPULATED", "VOLUME", "3D"}

def is_true_3d(ds) -> bool:
    """Detecta se a série é volumétrica (3D) com base em tokens fortes e estrutura multi-frame."""
    # 1) Multi-frame real
    try:
        if int(getattr(ds, "NumberOfFrames", 0) or 0) > 1:
            return True
    except Exception:
        pass

    # 2) Coleta textos relevantes (em maiúsculas)
    texts = []
    for attr in ["SeriesDescription", "ProtocolName", "ImageType"]:
        val = getattr(ds, attr, "")
        if isinstance(val, (list, tuple)):
            val = " ".join(val)
        texts.append(str(val).upper())
    bigtxt = " ".join(texts)

    # 3) Verifica tokens fortes inteiros
    tokens = set(re.findall(r"[A-Z0-9_]+", bigtxt))
    if TOK_3D_STRONG & tokens:
        return True

    # 4) Evita falsos positivos ("VOL PELVE", "VOL TORAX", etc.)
    if re.search(r"\bVOL(UME)?\b", bigtxt) and not re.search(r"VRT|RADIAL|CSAMANIPULATED|VR|MIP|SSD", bigtxt):
        return False

    return False

def ensure_dir(p): os.makedirs(p, exist_ok=True)

def place_file(src, dst, args):
    if args.dry_run:
        return
    if args.move:
        shutil.move(src, dst)
    elif args.hardlink:
        os.link(src, dst)
    elif args.symlink:
        os.symlink(src, dst)
    else:
        shutil.copy2(src, dst)

def classify_series(rep_ds, num_files=None) -> str:
    # Se for 3D por sinais fortes, classifica 3D
    if is_true_3d(rep_ds):
        return "3d"

    # Pista axial → favorece transversal
    axial_hint = plane_axial(rep_ds)
    it = getattr(rep_ds, "ImageType", [])
    if not isinstance(it, (list, tuple)):
        it = [str(it)]
    it_tokens = {str(x).strip().upper() for x in it if str(x).strip()}

    if axial_hint is True or "AXIAL" in it_tokens:
        return "transversal"

    # fallback conservador (sem pistas contrárias): transversal
    return "transversal"

def main():
    ap = argparse.ArgumentParser(description="Separa DICOMs em 2 classes: transversal e 3D, preservando a árvore original.")
    ap.add_argument("-i","--input", required=True, help="Pasta raiz com DICOMs (scan recursivo).")
    ap.add_argument("-o","--output", required=True, help="Pasta de saída.")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--move", action="store_true", help="Mover arquivos (mesmo FS).")
    g.add_argument("--hardlink", action="store_true", help="Hardlinks (mesma partição).")
    g.add_argument("--symlink", action="store_true", help="Symlinks (atalhos).")
    ap.add_argument("--dry-run", action="store_true", help="Não escreve; apenas imprime o que faria.")
    ap.add_argument("--progress-step", type=int, default=1000, help="Quantos arquivos por ponto de progresso (default=1000).")
    args = ap.parse_args()

    in_root = os.path.abspath(args.input)
    out_root = os.path.abspath(args.output)
    ensure_dir(out_root)
    ensure_dir(os.path.join(out_root, "transversal"))
    ensure_dir(os.path.join(out_root, "3d"))

    # Agrupar por série
    series = defaultdict(lambda: {"files": [], "rep": None})
    for root, _, files in os.walk(in_root):
        for fn in files:
            path = os.path.join(root, fn)
            ds, err = read_meta_fast(path)
            if ds is None:
                continue
            sid = series_key(ds)
            s = series[sid]
            s["files"].append(path)
            if s["rep"] is None:
                s["rep"] = ds

    counts = {"transversal": 0, "3d": 0}
    processed = 0

    print("[PROCESSO] executando...", end="", flush=True)

    for sid, entry in series.items():
        rep = entry["rep"]
        cat = "transversal" if rep is None else classify_series(rep)

        for src in entry["files"]:
            rel_dir = os.path.relpath(os.path.dirname(src), start=in_root)
            dest_dir = os.path.join(out_root, cat, rel_dir)
            try:
                ensure_dir(dest_dir)
            except OSError as e:
                # continua silenciosamente, mas registra erro mínimo
                print(f"\n[ERRO] criar {dest_dir}: {e}", file=sys.stderr)
                continue

            dst = os.path.join(dest_dir, os.path.basename(src))
            try:
                place_file(src, dst, args)
                counts[cat] += 1
                processed += 1
                if processed % max(1, args.progress_step) == 0:
                    print(".", end="", flush=True)
            except OSError as e:
                print(f"\n[WARN] Falha ao processar {src}: {e}", file=sys.stderr)

    total = counts["transversal"] + counts["3d"]
    print(f"\n[OK] finalizado. imagens: total={total} | transversal={counts['transversal']} | 3d={counts['3d']}")

if __name__ == "__main__":
    main()
