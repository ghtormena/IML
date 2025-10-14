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

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}  # ajuste se precisar

def is_image(p: Path) -> bool:
    return p.is_file() and p.suffix.lower() in IMG_EXTS

def collect_patients(root: Path):
    """
    Walks the expected structure:
      root/<class>/<patient>/(cranio|pelve|...)/images
    Returns:
      patients: dict[(class_name, patient_id)] -> list[Path] (image files)
      class_names: sorted list of class folder names
    """
    if not root.is_dir():
        raise RuntimeError(f"Root dir not found: {root}")

    class_names = [d.name for d in root.iterdir() if d.is_dir()]
    class_names.sort()

    patients = {}  # key: (cls, patient) -> list of image Paths
    empty_patients = []

    for cls in class_names:
        class_dir = root / cls
        for patient_dir in sorted([d for d in class_dir.iterdir() if d.is_dir()]):
            patient_id = patient_dir.name
            # merge cranio + pelve + qualquer outra subpasta
            images = [p for p in patient_dir.rglob("*") if is_image(p)]
            key = (cls, patient_id)
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
        raise RuntimeError("Nenhuma imagem encontrada. Verifique a estrutura do dataset e as extensões permitidas.")

    return patients, class_names

def stratified_split_by_patient(patients, class_names, val_ratio, test_ratio, holdout_ratio, seed):
    """
    patients: dict[(cls, patient)] -> list[images]
    Returns dict patient_split[(cls, patient)] = 'train' | 'val' | 'test' | 'holdout'
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
        # garante que não ultrapasse
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

def safe_symlink_or_copy(src: Path, dst: Path, do_copy: bool):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if do_copy:
        shutil.copy2(src, dst)
    else:
        # remove se já existir (para rebuilds idempotentes)
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        os.symlink(src, dst)

def build_output_structure(out_root: Path, class_names):
    for split in ("train", "val", "test", "holdout"):
        for cls in class_names:
            (out_root / split / cls).mkdir(parents=True, exist_ok=True)

def main():
    ap = argparse.ArgumentParser(description="Prepare YOLO classification dataset by patient (with holdout).")
    ap.add_argument("--root", type=Path, required=True, help="Diretório raiz do dataset original.")
    ap.add_argument("--out",  type=Path, required=True, help="Diretório de saída para o dataset YOLO.")
    ap.add_argument("--val", type=float, default=0.15, help="Proporção de pacientes para validação (por classe).")
    ap.add_argument("--test", type=float, default=0.15, help="Proporção de pacientes para teste (por classe).")
    ap.add_argument("--holdout", type=float, default=0.05, help="Proporção de pacientes para holdout (por classe).")
    ap.add_argument("--seed", type=int, default=42, help="Seed para aleatoriedade.")
    ap.add_argument("--copy", action="store_true", help="Copiar arquivos ao invés de criar symlinks.")
    ap.add_argument("--flatten", action="store_true", help="Achatar: salvar imagens direto na pasta da classe (default = manter prefixo por paciente no nome).")
    args = ap.parse_args()

    if args.val < 0 or args.test < 0 or args.holdout < 0:
        raise SystemExit("As proporções val/test/holdout devem ser >= 0.")
    if args.val + args.test + args.holdout >= 1.0:
        raise SystemExit("A soma val+test+holdout deve ser < 1.0 para sobrar pacientes em 'train'.")

    patients, class_names = collect_patients(args.root)
    patient_split = stratified_split_by_patient(
        patients, class_names, args.val, args.test, args.holdout, args.seed
    )

    # cria estrutura de saída
    build_output_structure(args.out, class_names)

    # manifest por paciente + contagem de imagens movidas
    manifest_rows = []
    per_split_counts = {s: defaultdict(int) for s in ("train","val","test","holdout")}
    total_images = 0

    for (cls, pid), imgs in patients.items():
        split = patient_split[(cls, pid)]
        # destino base (classe dentro do split)
        base = args.out / split / cls

        if args.flatten:
            # imagens diretamente em base/<cls> com prefixo do paciente no nome
            for idx, img in enumerate(imgs):
                stem = img.stem
                dst = base / f"{pid}__{idx:04d}{img.suffix.lower()}"
                safe_symlink_or_copy(img, dst, args.copy)
        else:
            # mantém apenas o prefixo no nome para unicidade, sem criar subpasta do paciente
            for idx, img in enumerate(imgs):
                dst = base / f"{pid}__{img.stem}{img.suffix.lower()}"
                safe_symlink_or_copy(img, dst, args.copy)

        n_imgs = len(imgs)
        total_images += n_imgs
        per_split_counts[split][cls] += n_imgs
        manifest_rows.append({
            "class": cls,
            "patient": pid,
            "split": split,
            "num_images": n_imgs
        })

    # salva manifest
    manifest_path = args.out / "manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["class","patient","split","num_images"])
        w.writeheader()
        for row in sorted(manifest_rows, key=lambda r:(r["split"], r["class"], r["patient"])):
            w.writerow(row)

    # resumo
    print("\n=== RESUMO ===")
    print(f"Saída: {args.out}")
    print(f"Total de imagens vinculadas: {total_images} ({'copiadas' if args.copy else 'symlinks'})")
    for split in ("train","val","test","holdout"):
        tot_split = sum(per_split_counts[split].values())
        print(f" {split.upper():7s}: {tot_split:6d} imagens  ", end="")
        for cls in class_names:
            print(f"{cls}={per_split_counts[split][cls]:d}  ", end="")
        print()
    print(f"Manifest salvo em: {manifest_path}")

if __name__ == "__main__":
    main()
