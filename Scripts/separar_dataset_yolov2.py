#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Prep YOLO Classification dataset by PATIENT (stratified), merging 'cranio' and 'pelve',
and creating a 'holdout' set of patients fully excluded from train/val/test.

- Input root structure (example):
  root/
    feminino/
      PACIENTE_001/
        cranio/ ...images...
        pelve/  ...images...
      PACIENTE_002/ ...
    masculino/
      PACIENTE_003/ ...
      ...

- Output structure (Ultralytics YOLO classify expects):
  out/
    train/
      feminino/  (all images of patients assigned to train for this class)
      masculino/
    val/
      feminino/
      masculino/
    test/
      feminino/
      masculino/
    holdout/      (kept completely outside training/val/test)
      feminino/
      masculino/

Usage:
  python prep_yolo_by_patient.py --root /path/in --out /path/out \
      --val 0.15 --test 0.15 --holdout 0.05 --seed 42 [--copy]

Notes:
- By default creates symlinks to save disk space. Use --copy to physically copy files.
- File names are prefixed with the patient id to avoid collisions.
- A manifest.csv is saved in the output with per-patient counts and split assignment.
"""

import argparse
import os
import sys
import shutil
import random
from pathlib import Path
from collections import defaultdict
import csv
import hashlib

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

def is_image(p: Path) -> bool:
    return p.is_file() and p.suffix.lower() in IMG_EXTS

def collect_patients(root: Path):
    """
    Lê a estrutura root/<class>/<patient>/(cranio|pelve|...)/images
    Retorna:
      patients: dict[(class_name, patient_id)] -> list[Path] (arquivos de imagem)
      class_names: lista de classes
    """
    if not root.is_dir():
        raise RuntimeError(f"Root dir not found: {root}")

    class_names = [d.name for d in root.iterdir() if d.is_dir()]
    class_names.sort()

    patients = {}
    empty_patients = []

    for cls in class_names:
        class_dir = root / cls
        for patient_dir in sorted([d for d in class_dir.iterdir() if d.is_dir()]):
            pid = patient_dir.name
            images = [p for p in patient_dir.rglob("*") if is_image(p)]
            key = (cls, pid)
            if images:
                patients[key] = images
            else:
                empty_patients.append(key)

    if empty_patients:
        print(f"[AVISO] Pacientes sem imagens (ignorados): {len(empty_patients)}", file=sys.stderr)
        for cls, pid in empty_patients[:10]:
            print(f"  - {cls}/{pid}", file=sys.stderr)
        if len(empty_patients) > 10:
            print("  ...", file=sys.stderr)

    if not patients:
        raise RuntimeError("Nenhuma imagem encontrada. Verifique a estrutura e extensões permitidas.")

    return patients, class_names

def stratified_split_by_patient(patients, class_names, val_ratio, test_ratio, holdout_ratio, seed):
    """
    patients: dict[(cls, patient)] -> list[images]
    Retorna: dict patient_split[(cls, patient)] = 'train' | 'val' | 'test' | 'holdout'
    """
    rng = random.Random(seed)
    per_class_patients = defaultdict(list)
    for (cls, pid) in patients.keys():
        per_class_patients[cls].append(pid)

    patient_split = {}

    for cls in class_names:
        ids = sorted(per_class_patients[cls])
        rng.shuffle(ids)

        n = len(ids)
        n_hold = int(round(n * holdout_ratio))
        n_test = int(round(n * test_ratio))
        n_val  = int(round(n * val_ratio))

        n_hold = min(n_hold, n)
        n_test = min(n_test, max(0, n - n_hold))
        n_val  = min(n_val,  max(0, n - n_hold - n_test))
        n_train = n - (n_hold + n_val + n_test)
        assert n_train >= 0

        hold_ids = ids[:n_hold]
        val_ids  = ids[n_hold:n_hold + n_val]
        test_ids = ids[n_hold + n_val:n_hold + n_val + n_test]
        train_ids= ids[n_hold + n_val + n_test:]

        for pid in hold_ids:
            patient_split[(cls, pid)] = "holdout"
        for pid in val_ids:
            patient_split[(cls, pid)] = "val"
        for pid in test_ids:
            patient_split[(cls, pid)] = "test"
        for pid in train_ids:
            patient_split[(cls, pid)] = "train"

    return patient_split

def copy_file(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)

def build_output_structure(out_root: Path, class_names):
    for split in ("train", "val", "test", "holdout"):
        for cls in class_names:
            (out_root / split / cls).mkdir(parents=True, exist_ok=True)

def make_unique_name(src: Path, pid: str, idx: int) -> str:
    """
    Evita colisões de nomes usando um hash do caminho fonte.
    Ex.: PACIENTE_X tem cranio/image001.png e pelve/image001.png -> nomes distintos.
    """
    h = hashlib.sha1(str(src.as_posix()).encode('utf-8')).hexdigest()[:10]
    return f"{pid}__{h}__{idx:04d}{src.suffix.lower()}"

def main():
    ap = argparse.ArgumentParser(description="Prepare YOLO classification dataset by patient (with holdout), copying files (no symlinks).")
    ap.add_argument("--root", type=Path, required=True, help="Diretório raiz do dataset original.")
    ap.add_argument("--out",  type=Path, required=True, help="Diretório de saída para o dataset YOLO.")
    ap.add_argument("--val", type=float, default=0.15, help="Proporção de pacientes para validação (por classe).")
    ap.add_argument("--test", type=float, default=0.15, help="Proporção de pacientes para teste (por classe).")
    ap.add_argument("--holdout", type=float, default=0.05, help="Proporção de pacientes para holdout (por classe).")
    ap.add_argument("--seed", type=int, default=42, help="Seed para aleatoriedade.")
    ap.add_argument("--flatten", action="store_true", help="Achatar: salvar imagens diretamente na pasta da classe (mantendo unicidade por hash).")
    args = ap.parse_args()

    if args.val < 0 or args.test < 0 or args.holdout < 0:
        raise SystemExit("As proporções val/test/holdout devem ser >= 0.")
    if args.val + args.test + args.holdout >= 1.0:
        raise SystemExit("A soma val+test+holdout deve ser < 1.0 para sobrar pacientes em 'train'.")

    patients, class_names = collect_patients(args.root)
    patient_split = stratified_split_by_patient(
        patients, class_names, args.val, args.test, args.holdout, args.seed
    )

    build_output_structure(args.out, class_names)

    manifest_rows = []
    per_split_counts = {s: defaultdict(int) for s in ("train","val","test","holdout")}
    total_images = 0

    for (cls, pid), imgs in patients.items():
        split = patient_split[(cls, pid)]
        base = args.out / split / cls

        for idx, img in enumerate(imgs):
            # nome único e estável por caminho + índice
            dst_name = make_unique_name(img, pid, idx)
            dst = base / dst_name
            copy_file(img, dst)

        n_imgs = len(imgs)
        total_images += n_imgs
        per_split_counts[split][cls] += n_imgs
        manifest_rows.append({
            "class": cls,
            "patient": pid,
            "split": split,
            "num_images": n_imgs
        })

    manifest_path = args.out / "manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["class","patient","split","num_images"])
        w.writeheader()
        for row in sorted(manifest_rows, key=lambda r:(r["split"], r["class"], r["patient"])):
            w.writerow(row)

    print("\n=== RESUMO ===")
    print(f"Saída: {args.out}")
    print(f"Total de imagens copiadas: {total_images}")
    for split in ("train","val","test","holdout"):
        tot_split = sum(per_split_counts[split].values())
        print(f" {split.upper():7s}: {tot_split:6d} imagens  ", end="")
        for cls in class_names:
            print(f"{cls}={per_split_counts[split][cls]:d}  ", end="")
        print()
    print(f"Manifest salvo em: {manifest_path}")
    print("\nExemplo de treino (Ultralytics YOLO - classificação):")
    print(f"  yolo classify train data='{args.out}' model=resnet18.pt epochs=50 imgsz=224")

if __name__ == "__main__":
    main()