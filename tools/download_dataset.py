#!/usr/bin/env python3
"""
Download & prepare popular datasets for the RTMODT pipeline.

Supported datasets
──────────────────
  coco128     —  128-image COCO subset (quick smoke-test)
  coco        —  Full COCO 2017 (118K train / 5K val images)
  mot17       —  MOT17 benchmark (tracking evaluation)

Usage
─────
    python tools/download_dataset.py --dataset coco128
    python tools/download_dataset.py --dataset coco   --dest datasets/coco
    python tools/download_dataset.py --dataset mot17  --dest datasets/mot17

After download the script:
  1. Filters COCO annotations to project-relevant classes (person, vehicle).
  2. Converts annotations to YOLO-format .txt labels.
  3. Generates a ready-to-use dataset.yaml.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import zipfile
from pathlib import Path
from typing import Optional
from urllib.request import urlretrieve

import click
import yaml
from loguru import logger

# ── Project root on sys.path ──
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# =========================================================================
# COCO class mapping  →  RTMODT classes
# =========================================================================
# COCO 80-class IDs we care about → our simplified class map
COCO_TO_RTMODT = {
    0: 0,     # person    → person
    1: 1,     # bicycle   → vehicle
    2: 1,     # car       → vehicle
    3: 1,     # motorcycle → vehicle
    5: 1,     # bus       → vehicle
    7: 1,     # truck     → vehicle
}

RTMODT_NAMES = {0: "person", 1: "vehicle"}

# =========================================================================
# Download helpers
# =========================================================================

_URLS = {
    "coco128": {
        "images": "https://ultralytics.com/assets/coco128.zip",
    },
    "coco": {
        "train_images": "http://images.cocodataset.org/zips/train2017.zip",
        "val_images":   "http://images.cocodataset.org/zips/val2017.zip",
        "annotations":  "http://images.cocodataset.org/annotations/annotations_trainval2017.zip",
    },
    "mot17": {
        "data": "https://motchallenge.net/data/MOT17.zip",
    },
}


def _download(url: str, dest: Path) -> Path:
    """Download a file with progress, skip if already cached."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        logger.info("Already downloaded: {}", dest.name)
        return dest

    logger.info("Downloading {} → {}", url, dest)

    def _progress(block, block_size, total):
        pct = block * block_size / total * 100 if total > 0 else 0
        print(f"\r  {pct:5.1f}%  ({block * block_size / 1e6:.1f} / {total / 1e6:.1f} MB)", end="", flush=True)

    urlretrieve(url, str(dest), reporthook=_progress)
    print()
    return dest


def _extract(archive: Path, dest: Path) -> None:
    """Extract a zip archive."""
    logger.info("Extracting {} → {}", archive.name, dest)
    with zipfile.ZipFile(archive, "r") as zf:
        zf.extractall(dest)


# =========================================================================
# COCO → YOLO conversion
# =========================================================================

def _coco_to_yolo_labels(
    annotation_json: Path,
    images_dir: Path,
    labels_dir: Path,
    class_map: dict[int, int],
) -> int:
    """
    Convert COCO JSON annotations to YOLO-format .txt files.

    Returns the number of label files written.
    """
    labels_dir.mkdir(parents=True, exist_ok=True)

    with open(annotation_json) as f:
        coco = json.load(f)

    # Build COCO category_id → contiguous 0-based index
    cat_id_to_coco_idx = {}
    for i, cat in enumerate(coco["categories"]):
        cat_id_to_coco_idx[cat["id"]] = i

    # Build image_id → {file_name, width, height}
    img_info = {}
    for img in coco["images"]:
        img_info[img["id"]] = {
            "file_name": img["file_name"],
            "w": img["width"],
            "h": img["height"],
        }

    # Group annotations by image_id
    from collections import defaultdict
    anns_by_img: dict[int, list] = defaultdict(list)
    for ann in coco["annotations"]:
        if ann.get("iscrowd", 0):
            continue
        anns_by_img[ann["image_id"]].append(ann)

    written = 0
    for img_id, info in img_info.items():
        lines = []
        for ann in anns_by_img.get(img_id, []):
            coco_idx = cat_id_to_coco_idx.get(ann["category_id"])
            if coco_idx is None or coco_idx not in class_map:
                continue
            rtmodt_cls = class_map[coco_idx]

            # COCO bbox = [x_min, y_min, width, height] (absolute)
            x, y, w, h = ann["bbox"]
            img_w, img_h = info["w"], info["h"]

            # Convert to YOLO: cx, cy, w, h (normalized)
            cx = (x + w / 2) / img_w
            cy = (y + h / 2) / img_h
            nw = w / img_w
            nh = h / img_h

            # Clamp to [0, 1]
            cx = max(0.0, min(1.0, cx))
            cy = max(0.0, min(1.0, cy))
            nw = max(0.0, min(1.0, nw))
            nh = max(0.0, min(1.0, nh))

            if nw < 0.001 or nh < 0.001:
                continue  # skip degenerate boxes

            lines.append(f"{rtmodt_cls} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

        if lines:
            stem = Path(info["file_name"]).stem
            label_file = labels_dir / f"{stem}.txt"
            label_file.write_text("\n".join(lines) + "\n")
            written += 1

    logger.info("Wrote {} YOLO label files to {}", written, labels_dir)
    return written


# =========================================================================
# Dataset-specific download & prepare functions
# =========================================================================

def _prepare_coco128(dest: Path) -> Path:
    """Download COCO128 and restructure for the project."""
    cache = dest / "_cache"
    archive = _download(_URLS["coco128"]["images"], cache / "coco128.zip")
    _extract(archive, cache)

    # coco128 ships with images + labels already in YOLO format
    src = cache / "coco128"
    out = dest / "coco128"

    # Copy images
    for split in ["images", "labels"]:
        src_split = src / split / "train2017"
        if not src_split.exists():
            src_split = src / split / "train"  # some versions
        if not src_split.exists():
            # Flat structure
            src_split = src / split
        dst_split = out / split / "train"
        if dst_split.exists():
            shutil.rmtree(dst_split)
        shutil.copytree(src_split, dst_split)

    # Create val = last 28 images (for smoke-testing)
    train_imgs = sorted((out / "images" / "train").glob("*.*"))
    val_imgs_dir = out / "images" / "val"
    val_lbls_dir = out / "labels" / "val"
    val_imgs_dir.mkdir(parents=True, exist_ok=True)
    val_lbls_dir.mkdir(parents=True, exist_ok=True)

    for img_path in train_imgs[-28:]:
        shutil.move(str(img_path), str(val_imgs_dir / img_path.name))
        lbl = out / "labels" / "train" / (img_path.stem + ".txt")
        if lbl.exists():
            shutil.move(str(lbl), str(val_lbls_dir / lbl.name))

    # Write dataset yaml
    ds_yaml = out / "dataset.yaml"
    ds_cfg = {
        "path": str(out.resolve()),
        "train": "images/train",
        "val": "images/val",
        "nc": 80,
        "names": {i: n for i, n in enumerate([
            "person","bicycle","car","motorcycle","airplane","bus","train","truck",
            "boat","traffic light","fire hydrant","stop sign","parking meter","bench",
            "bird","cat","dog","horse","sheep","cow","elephant","bear","zebra","giraffe",
            "backpack","umbrella","handbag","tie","suitcase","frisbee","skis","snowboard",
            "sports ball","kite","baseball bat","baseball glove","skateboard","surfboard",
            "tennis racket","bottle","wine glass","cup","fork","knife","spoon","bowl",
            "banana","apple","sandwich","orange","broccoli","carrot","hot dog","pizza",
            "donut","cake","chair","couch","potted plant","bed","dining table","toilet",
            "tv","laptop","mouse","remote","keyboard","cell phone","microwave","oven",
            "toaster","sink","refrigerator","book","clock","vase","scissors","teddy bear",
            "hair drier","toothbrush",
        ])},
    }
    ds_yaml.write_text(yaml.dump(ds_cfg, default_flow_style=False, sort_keys=False))

    # Also write project-filtered dataset yaml (person + vehicle only)
    filtered_yaml = out / "dataset_rtmodt.yaml"
    filtered_cfg = {
        "path": str(out.resolve()),
        "train": "images/train",
        "val": "images/val",
        "nc": 2,
        "names": dict(RTMODT_NAMES),
        "_note": "Labels use full COCO 80-class IDs. "
                 "Set detection.classes in default.yaml to [0,1,2,3,5,7] to filter at inference.",
    }
    filtered_yaml.write_text(yaml.dump(filtered_cfg, default_flow_style=False, sort_keys=False))

    logger.info("✅  COCO128 ready at {}", out)
    logger.info("    Full 80-class YAML:      {}", ds_yaml)
    logger.info("    Filtered RTMODT YAML:     {}", filtered_yaml)
    return out


def _prepare_coco(dest: Path) -> Path:
    """Download full COCO 2017 and convert to YOLO format."""
    cache = dest / "_cache"
    out = dest / "coco2017"

    # Download
    ann_zip = _download(_URLS["coco"]["annotations"], cache / "annotations_trainval2017.zip")
    train_zip = _download(_URLS["coco"]["train_images"], cache / "train2017.zip")
    val_zip = _download(_URLS["coco"]["val_images"], cache / "val2017.zip")

    # Extract
    _extract(ann_zip, cache)
    _extract(train_zip, out / "images")
    _extract(val_zip, out / "images")

    # Rename to expected structure
    for split in ["train2017", "val2017"]:
        src = out / "images" / split
        dst = out / "images" / split.replace("2017", "")
        if src.exists() and not dst.exists():
            src.rename(dst)

    # Convert annotations → YOLO labels (all 80 classes)
    ann_dir = cache / "annotations"
    for split, ann_file in [("train", "instances_train2017.json"), ("val", "instances_val2017.json")]:
        ann_path = ann_dir / ann_file
        if ann_path.exists():
            _coco_to_yolo_labels(
                annotation_json=ann_path,
                images_dir=out / "images" / split,
                labels_dir=out / "labels" / split,
                class_map={i: i for i in range(80)},  # 1:1 for full COCO
            )

    # Also create RTMODT-filtered labels (person + vehicle only)
    rtmodt_dir = out / "labels_rtmodt"
    for split, ann_file in [("train", "instances_train2017.json"), ("val", "instances_val2017.json")]:
        ann_path = ann_dir / ann_file
        if ann_path.exists():
            _coco_to_yolo_labels(
                annotation_json=ann_path,
                images_dir=out / "images" / split,
                labels_dir=rtmodt_dir / split,
                class_map=COCO_TO_RTMODT,
            )

    # Write dataset yamls
    ds_yaml = out / "dataset.yaml"
    ds_cfg = {
        "path": str(out.resolve()),
        "train": "images/train",
        "val": "images/val",
        "nc": 80,
        "names": "coco80",  # Ultralytics built-in
    }
    ds_yaml.write_text(yaml.dump(ds_cfg, default_flow_style=False, sort_keys=False))

    filtered_yaml = out / "dataset_rtmodt.yaml"
    filtered_cfg = {
        "path": str(out.resolve()),
        "train": "images/train",
        "val": "images/val",
        "nc": 2,
        "names": dict(RTMODT_NAMES),
        "_labels_dir": "labels_rtmodt",
        "_note": "Use labels_rtmodt/ which has only person (0) and vehicle (1) classes.",
    }
    filtered_yaml.write_text(yaml.dump(filtered_cfg, default_flow_style=False, sort_keys=False))

    logger.info("✅  COCO 2017 ready at {}", out)
    return out


def _prepare_mot17(dest: Path) -> Path:
    """Download MOT17 benchmark for tracking evaluation."""
    cache = dest / "_cache"
    out = dest / "MOT17"

    archive = _download(_URLS["mot17"]["data"], cache / "MOT17.zip")
    _extract(archive, dest)

    # MOT17 extracts to dest/MOT17/ with train/ and test/ subdirs
    if out.exists():
        logger.info("✅  MOT17 ready at {}", out)
        logger.info("  Ground-truth:  {}/train/*/gt/gt.txt", out)
        logger.info("  Sequences:     {}/train/*/img1/", out)
    else:
        logger.warning("MOT17 directory not found at expected path: {}", out)

    return out


# =========================================================================
# CLI
# =========================================================================

@click.command()
@click.option(
    "--dataset", "-d",
    required=True,
    type=click.Choice(["coco128", "coco", "mot17"]),
    help="Dataset to download.",
)
@click.option(
    "--dest",
    default="datasets",
    help="Destination root directory.",
)
def main(dataset: str, dest: str) -> None:
    """Download and prepare a dataset for the RTMODT pipeline."""
    dest_path = Path(dest)
    dest_path.mkdir(parents=True, exist_ok=True)

    if dataset == "coco128":
        result = _prepare_coco128(dest_path)
    elif dataset == "coco":
        result = _prepare_coco(dest_path)
    elif dataset == "mot17":
        result = _prepare_mot17(dest_path)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    click.echo(f"\n{'='*60}")
    click.echo(f"  Dataset '{dataset}' is ready at: {result}")
    click.echo(f"{'='*60}")

    if dataset in ("coco128", "coco"):
        click.echo("\nNext steps:")
        click.echo(f"  1. Update config/dataset.yaml  →  data: {result}/dataset.yaml")
        click.echo(f"  2. Train:  python tools/train.py --cfg config/training.yaml")
        click.echo(f"  3. Or run inference directly with pretrained weights:")
        click.echo(f"     python tools/run_pipeline.py -c config/default.yaml -s <video>")
    elif dataset == "mot17":
        click.echo("\nNext steps:")
        click.echo(f"  1. Run tracker on MOT17 sequences")
        click.echo(f"  2. Evaluate:  python -c \"from src.evaluation.metrics import evaluate_tracking; ...\"")


if __name__ == "__main__":
    main()
