"""
reset_dry_run.py — Setzt alle generierten Dry-Run-Dateien zurück.

Verschiebt alle von den Notebooks erzeugten Dateien in einen Backup-Ordner
(backup/YYYY-MM-DD_HHMMSS/) oder löscht sie direkt (mit --delete).

Betroffene Verzeichnisse/Dateien:
  NB 00: data/partitions/*.json
  NB 01: data/queries/dry_run/  (WAV-Dateien + manifest_dry.csv)
  NB 02: results/dry_run/shazam_raw.csv, shazam_efficiency.json
  NB 03: results/dry_run/quad_raw.csv, quad_efficiency.json
  NB 04: checkpoints/dry_run/, pfann/lists/dry_*.txt, pfann/lists/musan_*.txt,
          pfann/configs/dry_run.json
  NB 05: results/dry_run/neuralFP_*.*, results/dry_run/pfann_db/
  NB 06: results/dry_run/snr_breakpoints.csv, efficiency_table.tex,
          results/plots/*.pdf

Nicht angefasst:
  - data/fma_medium/, data/musan/, data/gtzan/  (Rohdaten)
  - src/pfann/configs/ (außer dry_run.json)
  - src/pfann/lists/readme.md
  - notebooks/*.ipynb  (Notebook-Dateien selbst)
  - src/ Code-Dateien
"""

import argparse
import shutil
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).parent

# Alle Pfade, die von den Notebooks erzeugt werden.
# Jeder Eintrag ist (notebook, pfad_relativ_zu_ROOT).
GENERATED = [
    # NB 00 — Partitionierung
    ("NB00", "data/partitions/dry_ref.json"),
    ("NB00", "data/partitions/dry_train.json"),
    ("NB00", "data/partitions/dry_ood.json"),
    ("NB00", "data/partitions/ood_pool.json"),
    ("NB00", "data/partitions/musan_split.json"),
    ("NB00", "data/partitions/ref.json"),
    ("NB00", "data/partitions/train.json"),

    # NB 01 — Query-Generierung (gesamtes Verzeichnis + Manifest)
    ("NB01", "data/queries/dry_run"),          # WAV-Verzeichnis
    ("NB01", "data/queries/manifest_dry.csv"),

    # NB 02 — Shazam
    ("NB02", "results/dry_run/shazam_raw.csv"),
    ("NB02", "results/dry_run/shazam_efficiency.json"),

    # NB 03 — Quad
    ("NB03", "results/dry_run/quad_raw.csv"),
    ("NB03", "results/dry_run/quad_efficiency.json"),

    # NB 04 — NeuralFP Training
    ("NB04", "checkpoints/dry_run"),
    ("NB04", "src/pfann/lists/dry_train.txt"),
    ("NB04", "src/pfann/lists/dry_ref.txt"),
    ("NB04", "src/pfann/lists/dry_validate.txt"),
    ("NB04", "src/pfann/lists/musan_train.txt"),
    ("NB04", "src/pfann/configs/dry_run.json"),

    # NB 05 — NeuralFP Pipeline
    ("NB05", "src/pfann/lists/dry_queries.txt"),
    ("NB05", "results/dry_run/neuralFP_matches.tsv"),
    ("NB05", "results/dry_run/neuralFP_matches.tsv.bin"),
    ("NB05", "results/dry_run/neuralFP_matches_detail.csv"),
    ("NB05", "results/dry_run/neuralFP_raw.csv"),
    ("NB05", "results/dry_run/pfann_db"),

    # NB 06 — Evaluation
    ("NB06", "results/dry_run/snr_breakpoints.csv"),
    ("NB06", "results/dry_run/efficiency_table.tex"),
    ("NB06", "results/plots"),                 # gesamtes Plots-Verzeichnis
]


def collect_existing(entries: list[tuple[str, str]]) -> list[tuple[str, Path]]:
    """Gibt alle Einträge zurück, die tatsächlich auf der Platte existieren."""
    existing = []
    for nb, rel in entries:
        p = ROOT / rel
        if p.exists():
            existing.append((nb, p))
    return existing


def move_to_backup(existing: list[tuple[str, Path]], backup_root: Path) -> None:
    backup_root.mkdir(parents=True, exist_ok=True)
    for nb, src in existing:
        # Relativer Pfad innerhalb von ROOT → gleiche Struktur im Backup
        rel = src.relative_to(ROOT)
        dst = backup_root / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dst))
        print(f"  [{nb}] {rel}  →  backup/{backup_root.name}/{rel}")


def delete_files(existing: list[tuple[str, Path]]) -> None:
    for nb, p in existing:
        rel = p.relative_to(ROOT)
        if p.is_dir():
            shutil.rmtree(p)
        else:
            p.unlink()
        print(f"  [{nb}] gelöscht: {rel}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--delete",
        action="store_true",
        help="Dateien direkt löschen statt in Backup verschieben.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Nur anzeigen, was getan würde — nichts verändern.",
    )
    args = parser.parse_args()

    existing = collect_existing(GENERATED)

    if not existing:
        print("Nichts zu tun — keine generierten Dateien gefunden.")
        return

    print(f"Gefundene generierte Dateien/Verzeichnisse ({len(existing)}):")
    for nb, p in existing:
        print(f"  [{nb}] {p.relative_to(ROOT)}")

    if args.dry_run:
        action = "löschen" if args.delete else "in Backup verschieben"
        print(f"\n[dry-run] Würde {len(existing)} Einträge {action}.")
        return

    if args.delete:
        confirm = input(f"\n{len(existing)} Einträge unwiderruflich löschen? [ja/N] ")
        if confirm.strip().lower() != "ja":
            print("Abgebrochen.")
            sys.exit(0)
        print()
        delete_files(existing)
        print(f"\nFertig — {len(existing)} Einträge gelöscht.")
    else:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        backup_root = ROOT / "backup" / timestamp
        print(f"\nVerschiebe nach: backup/{timestamp}/")
        move_to_backup(existing, backup_root)
        print(f"\nFertig — {len(existing)} Einträge gesichert unter backup/{timestamp}/")


if __name__ == "__main__":
    main()
