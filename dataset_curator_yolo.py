#!/usr/bin/env python3
"""
CleanCrop - find bad labels in a YOLO dataset and fix them from a web dashboard.

Usage (simplest):
    python dataset_curator_yolo.py C:\\path\\to\\yolo_dataset

This scans the dataset, embeds every bounding box with DINOv2/CLIP, flags
suspected mislabels / outliers / duplicates / broken boxes, then opens the
review dashboard in your browser. Fixes you commit there are written back to
the YOLO .txt label files (a backup of every touched file is kept first).
"""
import os
import re
import sys
import json
import time
import shutil
import argparse
import threading
import webbrowser
import http.server
from io import BytesIO
from datetime import datetime
from collections import defaultdict
from urllib.parse import urlparse, parse_qs

import numpy as np
import yaml
from PIL import Image, ImageDraw, ImageOps
from tqdm import tqdm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_EXTS = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
DASHBOARD_FILES = ['index.html', 'styles.css', 'app.js', 'editor.js']
CACHE_VERSION = "v2-sq15"   # bump when crop/embedding preprocessing changes

THUMB_SIZE = 160            # saved review thumbnail (square, px)
EMBED_PAD = 0.15            # context added around the box for embedding
THUMB_PAD = 0.6             # context added around the box for the thumbnail
NUM_NEIGHBORS = 10          # neighbors stored per crop for the dashboard
DUP_FLOOR = 0.90            # lowest duplicate similarity the dashboard slider can reach
MAX_DUP_PAIRS = 20000
OVERLAP_IOU = 0.85          # two boxes in one image overlapping this much = double annotation
TINY_BOX_PX = 4


# ==========================================================================
# DATASET DISCOVERY
# ==========================================================================

def load_class_names(dataset_dir):
    """Class names from data.yaml (list or dict form), falling back to classes.txt."""
    for name in ("data.yaml", "data.yml", "dataset.yaml"):
        yaml_path = os.path.join(dataset_dir, name)
        if os.path.exists(yaml_path):
            with open(yaml_path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f) or {}
            names = data.get('names', [])
            if isinstance(names, dict):
                names = [names[k] for k in sorted(names.keys(), key=int)]
            if names:
                return [str(n) for n in names]
    txt_path = os.path.join(dataset_dir, "classes.txt")
    if os.path.exists(txt_path):
        with open(txt_path, 'r', encoding='utf-8') as f:
            names = [line.strip() for line in f if line.strip()]
        if names:
            return names
    return []


def find_splits(dataset_dir):
    """
    Supports both common YOLO layouts:
      dataset/train/images + dataset/train/labels     (Roboflow style)
      dataset/images/train + dataset/labels/train     (Ultralytics style)
      dataset/images + dataset/labels                 (single split)
    Returns a list of (split_name, images_dir, labels_dir).
    """
    splits = []
    top_images = os.path.join(dataset_dir, "images")
    if os.path.isdir(top_images):
        subdirs = sorted(d for d in os.listdir(top_images) if os.path.isdir(os.path.join(top_images, d)))
        for d in subdirs:
            splits.append((d, os.path.join(top_images, d), os.path.join(dataset_dir, "labels", d)))
        if any(f.lower().endswith(IMAGE_EXTS) for f in os.listdir(top_images)):
            splits.append(("all", top_images, os.path.join(dataset_dir, "labels")))
    for d in sorted(os.listdir(dataset_dir)):
        img_dir = os.path.join(dataset_dir, d, "images")
        if d != "images" and os.path.isdir(img_dir):
            splits.append((d, img_dir, os.path.join(dataset_dir, d, "labels")))
    return splits


def box_iou(a, b):
    ax1, ay1, ax2, ay2 = a[0] - a[2] / 2, a[1] - a[3] / 2, a[0] + a[2] / 2, a[1] + a[3] / 2
    bx1, by1, bx2, by2 = b[0] - b[2] / 2, b[1] - b[3] / 2, b[0] + b[2] / 2, b[1] + b[3] / 2
    iw = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    ih = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = iw * ih
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 0 else 0.0


def scan_dataset(dataset_dir, class_names):
    """
    Reads every label file. Returns (boxes, stats, problems) where boxes is a list of
    per-annotation dicts and problems lists lines that can't be used at all.
    """
    dataset_dir = os.path.abspath(dataset_dir)
    splits = find_splits(dataset_dir)
    if not splits:
        raise SystemExit(f"No 'images' folders found in {dataset_dir}. Expected train/images or images/train layout.")

    boxes, problems = [], []
    stats = {"images": 0, "background_images": 0, "missing_label_files": 0, "splits": {}}
    backgrounds = []  # images with no boxes (moved to report["backgrounds"] by analyze)
    seen_ids = set()

    for split, images_dir, labels_dir in splits:
        img_files = sorted(f for f in os.listdir(images_dir) if f.lower().endswith(IMAGE_EXTS))
        split_boxes = 0
        for img_file in img_files:
            stats["images"] += 1
            base = os.path.splitext(img_file)[0]
            label_path = os.path.join(labels_dir, base + ".txt")
            if not os.path.exists(label_path):
                stats["missing_label_files"] += 1
                stats["background_images"] += 1
                backgrounds.append({"image": img_file, "split": split, "image_path": os.path.join(images_dir, img_file), "missing_label": True})
                continue
            with open(label_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()

            image_boxes = []
            for line_no, line in enumerate(lines):
                parts = line.strip().split()
                if not parts:
                    continue
                try:
                    cls_id = int(float(parts[0]))
                    xc, yc, w, h = (float(p) for p in parts[1:5])  # ValueError if fewer than 5 fields
                except ValueError:
                    problems.append({"label_path": label_path, "line": line_no, "problem": f"Unparseable line: {line.strip()!r}"})
                    continue
                if w <= 0 or h <= 0:
                    problems.append({"label_path": label_path, "line": line_no, "problem": "Zero-size box"})
                    continue

                crop_id = f"{base}_{line_no}"
                if crop_id in seen_ids:
                    crop_id = f"{split}_{crop_id}"
                seen_ids.add(crop_id)

                issues = []
                if cls_id >= len(class_names) or cls_id < 0:
                    issues.append(f"Class id {cls_id} is not in data.yaml")
                if xc - w / 2 < -0.01 or yc - h / 2 < -0.01 or xc + w / 2 > 1.01 or yc + h / 2 > 1.01:
                    issues.append("Box extends outside the image")

                image_boxes.append({
                    "id": crop_id,
                    "split": split,
                    "image": img_file,
                    "image_path": os.path.join(images_dir, img_file),
                    "label_path": label_path,
                    "line": line_no,
                    "class_id": cls_id,
                    "label": class_names[cls_id] if 0 <= cls_id < len(class_names) else f"class_{cls_id}",
                    "bbox": [xc, yc, w, h],
                    "issues": issues,
                })

            if not image_boxes:
                stats["background_images"] += 1
                backgrounds.append({"image": img_file, "split": split, "image_path": os.path.join(images_dir, img_file)})

            # Double annotations: two boxes on (almost) the same spot.
            # Same class: only the later copy is flagged, so deleting every flagged box keeps one.
            # Different classes: both are flagged, the reviewer decides which is right.
            for j in range(len(image_boxes)):
                for i in range(j):
                    a, b = image_boxes[i], image_boxes[j]
                    iou = box_iou(a["bbox"], b["bbox"])
                    if iou < OVERLAP_IOU:
                        continue
                    if a["class_id"] == b["class_id"]:
                        b["issues"].append(f"Duplicate box: same object as line {a['line'] + 1} (IoU {iou:.2f}) - delete this copy")
                        break
                    a["issues"].append(f"Conflicting labels: line {b['line'] + 1} labels the same object '{b['label']}' (IoU {iou:.2f})")
                    b["issues"].append(f"Conflicting labels: line {a['line'] + 1} labels the same object '{a['label']}' (IoU {iou:.2f})")

            boxes.extend(image_boxes)
            split_boxes += len(image_boxes)
        stats["splits"][split] = {"images": len(img_files), "boxes": split_boxes}

    stats["boxes"] = len(boxes)
    stats["unusable_lines"] = len(problems)
    stats["backgrounds"] = backgrounds
    return boxes, stats, problems


# ==========================================================================
# CROPS + EMBEDDINGS
# ==========================================================================

def crop_square(img, cx, cy, side, fill=(114, 114, 114)):
    """Square crop centred on (cx, cy); areas outside the image are padded with gray."""
    side = max(2, int(round(side)))
    x0, y0 = int(round(cx - side / 2)), int(round(cy - side / 2))
    out = Image.new("RGB", (side, side), fill)
    src = img.crop((max(0, x0), max(0, y0), min(img.width, x0 + side), min(img.height, y0 + side)))
    out.paste(src, (max(0, -x0), max(0, -y0)))
    return out, x0, y0


def make_thumbnail(img, box_px):
    """Box with surrounding context, upscaled, with the box outline drawn on top."""
    x1, y1, x2, y2 = box_px
    bw, bh = x2 - x1, y2 - y1
    side = max(bw, bh) + 2 * max(THUMB_PAD * max(bw, bh), 16)
    region, ox, oy = crop_square(img, (x1 + x2) / 2, (y1 + y2) / 2, side, fill=(30, 30, 30))
    scale = THUMB_SIZE / region.width
    thumb = region.resize((THUMB_SIZE, THUMB_SIZE), Image.BICUBIC)
    draw = ImageDraw.Draw(thumb)
    draw.rectangle([(x1 - ox) * scale, (y1 - oy) * scale, (x2 - ox) * scale, (y2 - oy) * scale],
                   outline=(57, 255, 20), width=2)
    return thumb


def make_embedding_input(img, box_px):
    x1, y1, x2, y2 = box_px
    side = max(x2 - x1, y2 - y1) * (1 + 2 * EMBED_PAD) + 4
    region, _, _ = crop_square(img, (x1 + x2) / 2, (y1 + y2) / 2, side)
    return region


class Embedder:
    def __init__(self, model_mode, device):
        import torch
        from transformers import AutoImageProcessor, AutoModel, CLIPProcessor, CLIPModel
        self.torch = torch
        self.mode = model_mode
        self.device = device
        if device == "cpu":
            torch.set_num_threads(os.cpu_count() or 4)
        print(f"Loading {model_mode.upper()} model on {device} (first run downloads it)...")
        if model_mode == "dinov2":
            self.processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
            self.model = AutoModel.from_pretrained("facebook/dinov2-base").to(device)
        else:
            self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
        self.model.eval()

    def __call__(self, images):
        with self.torch.no_grad():
            inputs = self.processor(images=images, return_tensors="pt").to(self.device)
            if self.mode == "dinov2":
                out = self.model(**inputs).last_hidden_state[:, 0]
            else:
                out = self.model.get_image_features(**inputs)
                if not isinstance(out, self.torch.Tensor):  # newer transformers return an output object
                    out = getattr(out, "image_embeds", None) if getattr(out, "image_embeds", None) is not None else out.pooler_output
            return out.float().cpu().numpy()


def cache_key(box):
    mtime = int(os.path.getmtime(box["image_path"]))
    return f"{CACHE_VERSION}|{box['image_path']}|{mtime}|" + ",".join(f"{v:.6f}" for v in box["bbox"])


def build_crops_and_embeddings(boxes, output_dir, model_mode, batch_size, use_cache=True):
    """Saves a thumbnail per box and returns (kept_boxes, embeddings). Embeddings are cached on disk."""
    import torch
    crops_dir = os.path.join(output_dir, "crops")
    cache_dir = os.path.join(output_dir, "cache")
    os.makedirs(crops_dir, exist_ok=True)
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"embeddings_{model_mode}.npz")

    cache = {}
    if use_cache and os.path.exists(cache_path):
        data = np.load(cache_path, allow_pickle=False)
        cache = dict(zip(data["keys"].tolist(), data["vectors"]))
        print(f"Loaded {len(cache)} cached embeddings.")

    for b in boxes:
        b["_key"] = cache_key(b)
        b["path"] = f"crops/{b['id']}.jpg"

    todo_images = defaultdict(list)
    for b in boxes:
        needs_thumb = not os.path.exists(os.path.join(output_dir, b["path"]))
        if needs_thumb or b["_key"] not in cache:
            todo_images[b["image_path"]].append(b)

    embedder = None
    if any(b["_key"] not in cache for bl in todo_images.values() for b in bl):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        embedder = Embedder(model_mode, device)

    skipped = set()
    pending_imgs, pending_boxes = [], []

    def flush():
        if pending_imgs:
            vecs = embedder(pending_imgs)
            for pb, v in zip(pending_boxes, vecs):
                cache[pb["_key"]] = v.astype(np.float32)
            pending_imgs.clear()
            pending_boxes.clear()

    if todo_images:
        print(f"Cropping and embedding boxes from {len(todo_images)} images...")
    for img_path, img_boxes in tqdm(todo_images.items(), unit="img", disable=not todo_images):
        try:
            with Image.open(img_path) as im:
                img = ImageOps.exif_transpose(im).convert("RGB")
        except Exception as e:
            print(f"  ! Could not open {img_path}: {e}")
            skipped.update(b["id"] for b in img_boxes)
            continue
        W, H = img.size
        for b in img_boxes:
            xc, yc, w, h = b["bbox"]
            x1, y1 = max(0, (xc - w / 2) * W), max(0, (yc - h / 2) * H)
            x2, y2 = min(W, (xc + w / 2) * W), min(H, (yc + h / 2) * H)
            if x2 - x1 < 1 or y2 - y1 < 1:
                skipped.add(b["id"])
                continue
            box_px = (x1, y1, x2, y2)
            make_thumbnail(img, box_px).save(os.path.join(output_dir, b["path"]), "JPEG", quality=88)
            if b["_key"] not in cache:
                pending_imgs.append(make_embedding_input(img, box_px))
                pending_boxes.append(b)
                if len(pending_imgs) >= batch_size:
                    flush()
    if embedder is not None:
        flush()

    kept = [b for b in boxes if b["id"] not in skipped and b["_key"] in cache]
    if kept:
        keys = np.array([b["_key"] for b in kept])
        vecs = np.stack([cache[b["_key"]] for b in kept])
        np.savez(cache_path, keys=keys, vectors=vecs)
    emb = np.stack([cache[b["_key"]] for b in kept]).astype(np.float32) if kept else np.zeros((0, 1), np.float32)
    emb /= (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9)
    if skipped:
        print(f"  ! Skipped {len(skipped)} boxes that could not be cropped.")
    return kept, emb


# ==========================================================================
# ANALYSIS
# ==========================================================================

def neighbors_and_duplicates(emb, image_ids, k=NUM_NEIGHBORS, dup_floor=DUP_FLOOR):
    """Top-k cosine neighbors per crop + near-duplicate pairs (from different images), chunked for memory."""
    N = len(emb)
    k = max(1, min(k, N - 1))
    nbr_idx = np.zeros((N, k), np.int32)
    nbr_sim = np.zeros((N, k), np.float32)
    pi, pj, ps = [], [], []
    image_ids = np.asarray(image_ids)
    for s in range(0, N, 1024):
        e = min(N, s + 1024)
        sims = emb[s:e] @ emb.T
        rows = np.arange(e - s)
        sims[rows, rows + s] = -np.inf
        part = np.argpartition(-sims, k - 1, axis=1)[:, :k]
        psims = np.take_along_axis(sims, part, 1)
        order = np.argsort(-psims, axis=1)
        nbr_idx[s:e] = np.take_along_axis(part, order, 1)
        nbr_sim[s:e] = np.take_along_axis(psims, order, 1)

        ii, jj = np.nonzero(sims >= dup_floor)
        ii = ii + s
        keep = (jj > ii) & (image_ids[ii] != image_ids[jj])
        pi.append(ii[keep]); pj.append(jj[keep]); ps.append(sims[ii[keep] - s, jj[keep]])
    pi, pj, ps = np.concatenate(pi), np.concatenate(pj), np.concatenate(ps)
    if len(ps) > MAX_DUP_PAIRS:
        top = np.argsort(-ps)[:MAX_DUP_PAIRS]
        pi, pj, ps = pi[top], pj[top], ps[top]
    order = np.argsort(-ps)
    pairs = [[int(pi[o]), int(pj[o]), round(float(ps[o]), 4)] for o in order]
    return nbr_idx, nbr_sim, pairs


def class_outlier_scores(emb, labels, n_neighbors=20):
    """Per-class Local Outlier Factor (1.0 = typical, higher = more unusual) and centroid similarity."""
    from sklearn.neighbors import LocalOutlierFactor
    N = len(emb)
    lof = np.zeros(N, np.float32)
    centroid_sim = np.zeros(N, np.float32)
    for cls in np.unique(labels):
        idx = np.where(labels == cls)[0]
        c = emb[idx].mean(axis=0)
        centroid_sim[idx] = emb[idx] @ (c / (np.linalg.norm(c) + 1e-9))
        if len(idx) < 6:
            continue
        model = LocalOutlierFactor(n_neighbors=min(n_neighbors, len(idx) - 1), metric='cosine')
        model.fit(emb[idx])
        lof[idx] = -model.negative_outlier_factor_
    return lof, centroid_sim


def label_probe(emb, class_ids, min_count=5):
    """
    Out-of-fold linear classifier on the embeddings (confident-learning style).
    Each crop is scored by a model that never saw it, so a confident disagreement
    with the annotated class is strong evidence of a wrong label.
    Returns (pred_class_id, pred_conf, given_conf) arrays; -1 / nan where not computed.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    N = len(emb)
    pred = np.full(N, -1, np.int32)
    pred_conf = np.full(N, np.nan, np.float32)
    given_conf = np.full(N, np.nan, np.float32)
    classes, counts = np.unique(class_ids, return_counts=True)
    eligible = classes[counts >= min_count]
    if len(eligible) < 2:
        print("  Label check skipped: need at least 2 classes with 5+ boxes.")
        return pred, pred_conf, given_conf
    mask = np.isin(class_ids, eligible)
    X, y = emb[mask], class_ids[mask]
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=0.1))
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    proba = cross_val_predict(clf, X, y, cv=cv, method="predict_proba")
    cols = np.unique(y)
    col_of = {c: i for i, c in enumerate(cols)}
    best = proba.argmax(axis=1)
    pred[mask] = cols[best]
    pred_conf[mask] = proba[np.arange(len(y)), best]
    given_conf[mask] = proba[np.arange(len(y)), [col_of[c] for c in y]]
    return pred, pred_conf, given_conf


def pca_3d(emb):
    X = emb - emb.mean(axis=0)
    _, _, vt = np.linalg.svd(X, full_matrices=False)
    coords = X @ vt[:3].T
    coords /= (np.abs(coords).max() + 1e-9)
    return coords


def tsne_2d(emb):
    """2D similarity map: PCA to 50 dims, then t-SNE (cosine). Falls back to PCA if it fails."""
    N = len(emb)
    try:
        from sklearn.manifold import TSNE
        X = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9)
        if X.shape[1] > 50 and N > 50:
            Xc = X - X.mean(axis=0)
            _, _, vt = np.linalg.svd(Xc, full_matrices=False)
            X = Xc @ vt[:50].T
        perp = float(max(5, min(30, (N - 1) / 3)))
        c = TSNE(n_components=2, perplexity=perp, metric="cosine", init="pca", random_state=0).fit_transform(X)
    except Exception as e:
        print(f"t-SNE failed ({e}); using PCA for the 2D map.")
        c = pca_3d(emb)[:, :2]
    c = c - c.mean(axis=0)
    return c / (np.abs(c).max() + 1e-9)


def analyze(boxes, emb, class_names, args, stats, problems, dataset_dir):
    N = len(boxes)
    labels = np.array([b["label"] for b in boxes])
    class_ids = np.array([b["class_id"] for b in boxes])

    print("Finding nearest neighbors and near-duplicates...")
    image_ids = [b["image_path"] for b in boxes]
    nbr_idx, nbr_sim, dup_pairs = neighbors_and_duplicates(emb, image_ids, dup_floor=min(DUP_FLOOR, args.dup_threshold))

    print("Scoring outliers within each class...")
    lof, centroid_sim = class_outlier_scores(emb, labels)

    print("Cross-checking every label with an out-of-fold classifier...")
    pred, pred_conf, given_conf = label_probe(emb, class_ids)

    coords = pca_3d(emb) if N >= 3 else np.zeros((N, 3))
    print("Building 2D similarity map (t-SNE)...")
    map2d = tsne_2d(emb) if N >= 5 else coords[:, :2]

    items = []
    for i, b in enumerate(boxes):
        p = int(pred[i])
        items.append({
            "index": i,
            "id": b["id"],
            "split": b["split"],
            "image": b["image"],
            "image_path": b["image_path"],
            "label_path": b["label_path"],
            "line": b["line"],
            "class_id": b["class_id"],
            "label": b["label"],
            "bbox": [round(v, 6) for v in b["bbox"]],
            "path": b["path"],
            "issues": b["issues"],
            "lof": round(float(lof[i]), 3),
            "centroid": round(float(centroid_sim[i]), 4),
            "pred": class_names[p] if 0 <= p < len(class_names) else None,
            "pred_conf": None if np.isnan(pred_conf[i]) else round(float(pred_conf[i]), 4),
            "given_conf": None if np.isnan(given_conf[i]) else round(float(given_conf[i]), 4),
            "nbrs": [[int(j), round(float(s), 4)] for j, s in zip(nbr_idx[i], nbr_sim[i])],
            "pca": [round(float(v), 4) for v in coords[i]],
            "map2d": [round(float(v), 4) for v in map2d[i]],
        })

    backgrounds = stats.pop("backgrounds", [])
    print("Computing distribution (cameras, day/night, box sizes, diversity)...")
    out_dir = getattr(args, "output_dir_resolved", None) or os.path.join(SCRIPT_DIR, "curation_output", os.path.basename(os.path.normpath(dataset_dir)))
    dist, img_meta, all_images = compute_distribution(boxes, emb, class_names, backgrounds, dup_pairs, args.dup_threshold, out_dir)
    n_flag = sum(1 for it in items if (it["pred"] and it["pred"] != it["label"] and (it["pred_conf"] or 0) >= args.mislabel_conf)
                 or it["lof"] >= args.lof_threshold)
    dist["readiness"] = readiness_checklist(dist, n_flag)
    for bg in backgrounds:
        m = img_meta.get(bg["image_path"], {})
        bg.update({"camera": m.get("camera"), "device": m.get("device"), "period": m.get("period")})
    golden = None
    if getattr(args, "golden_split", None):
        golden = golden_split(img_meta, all_images, boxes, emb, dup_pairs, args.dup_threshold, args.golden_split, out_dir)

    return {
        "version": 2,
        "dataset_name": os.path.basename(os.path.normpath(dataset_dir)),
        "dataset_dir": os.path.abspath(dataset_dir),
        "model": args.model,
        "created": datetime.now().isoformat(timespec="seconds"),
        "class_names": class_names,
        "params": {
            "dup_threshold": args.dup_threshold,
            "dup_floor": min(DUP_FLOOR, args.dup_threshold),
            "lof_threshold": args.lof_threshold,
            "mislabel_conf": args.mislabel_conf,
        },
        "stats": stats,
        "problems": problems,
        "items": items,
        "duplicates": dup_pairs,
        "backgrounds": backgrounds,
        "distribution": dist,
        "golden_test": golden,
    }


# ==========================================================================
# DISTRIBUTION + GOLDEN-TEST READINESS
# ==========================================================================

NAME_RE = re.compile(r'^(?P<dev>.+?)_(?P<cam>nvr-\d+_\d+)_(?P<date>\d{8})_(?P<time>\d{6})')
NIGHT_BRIGHTNESS = 70       # mean grey level below this = night (only used when the file name has no time)
DAY_HOURS = (6, 18)         # 06:00-17:59 = day
SMALL_PX, LARGE_PX = 32, 96 # longer box side in pixels


def parse_image_name(name):
    """'COE45003_nvr-1_15_20261113_092106_x.jpg' -> {device, camera, date, hour}; 'unknown' when no match."""
    m = NAME_RE.match(os.path.basename(name))
    if not m:
        return {"device": "unknown", "camera": "unknown", "date": None, "hour": None}
    return {"device": m["dev"], "camera": m["cam"], "date": m["date"], "hour": int(m["time"][:2])}


def image_info(paths, output_dir, need_brightness):
    """{path: {"size": [w, h], "brightness": float|None}} - cached in <output>/cache/image_info.json."""
    cache_path = os.path.join(output_dir, "cache", "image_info.json")
    cache = {}
    if os.path.exists(cache_path):
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                cache = json.load(f)
        except Exception:
            cache = {}
    out, changed = {}, False
    for p in paths:
        try:
            key = f"{p}|{os.path.getmtime(p):.0f}"
        except OSError:
            out[p] = {"size": None, "brightness": None}
            continue
        info = cache.get(key)
        if info is None or (need_brightness(p) and info.get("brightness") is None):
            info = dict(info or {})
            try:
                with Image.open(p) as im:
                    info["size"] = list(im.size)
                    if need_brightness(p):
                        im.draft("L", (128, 128))
                        info["brightness"] = round(float(np.asarray(im.convert("L").resize((64, 64))).mean()), 1)
            except Exception:
                info.setdefault("size", None)
            info.setdefault("brightness", None)
            cache[key] = info
            changed = True
        out[p] = info
    if changed:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(cache, f)
    return out


def _mean_pairwise_cos_dist(vecs, max_n=1500, seed=0):
    if len(vecs) < 2:
        return None
    if len(vecs) > max_n:
        vecs = vecs[np.random.default_rng(seed).choice(len(vecs), max_n, replace=False)]
    v = vecs / (np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-8)
    n = len(v)
    s = v.sum(0)
    mean_sim = (float(s @ s) - n) / (n * (n - 1))
    return round(1.0 - mean_sim, 4)


def _duplicate_groups(images, image_pairs):
    """Union-find over images linked by near-duplicate boxes -> {image_path: group_root}."""
    parent = {p: p for p in images}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for a, b in image_pairs:
        if a in parent and b in parent:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra
    return {p: find(p) for p in images}


def _empty_class_row():
    return {"images": 0, "boxes": 0, "day": 0, "night": 0, "unknown": 0, "small": 0, "medium": 0, "large": 0}


def compute_distribution(boxes, emb, class_names, backgrounds, dup_pairs, dup_threshold, output_dir):
    """Per-class/camera/device/day-night/box-size counts + diversity. Returns (dist, per-image meta, images)."""
    all_images = {}
    for b in boxes:
        all_images.setdefault(b["image_path"], {"image": b["image"], "split": b["split"], "boxes": []})["boxes"].append(b)
    for bg in backgrounds:
        all_images.setdefault(bg["image_path"], {"image": bg["image"], "split": bg["split"], "boxes": []})
    meta = {p: parse_image_name(v["image"]) for p, v in all_images.items()}
    info = image_info(list(all_images), output_dir, need_brightness=lambda p: meta[p]["hour"] is None)
    for p, m in meta.items():
        if m["hour"] is not None:
            m["period"], m["period_source"] = ("day" if DAY_HOURS[0] <= m["hour"] < DAY_HOURS[1] else "night"), "time"
        else:
            br = info[p].get("brightness")
            m["period"], m["period_source"] = ("unknown", "none") if br is None else (("night" if br < NIGHT_BRIGHTNESS else "day"), "brightness")
        m["size"] = info[p].get("size")

    def bump(d, k, field, n=1):
        d.setdefault(k, {"images": 0, "boxes": 0})[field] += n

    per_class = {c: _empty_class_row() for c in class_names}
    per_camera, per_device, per_period = {}, {}, {}
    sizes = {"small": 0, "medium": 0, "large": 0}
    cam_class = defaultdict(lambda: defaultdict(int))
    for p, v in all_images.items():
        m = meta[p]
        n = len(v["boxes"])
        for d, k in ((per_camera, m["camera"]), (per_device, m["device"]), (per_period, m["period"])):
            bump(d, k, "images")
            bump(d, k, "boxes", n)
        for c in {b["label"] for b in v["boxes"]}:
            per_class.setdefault(c, _empty_class_row())["images"] += 1
        for b in v["boxes"]:
            pc = per_class[b["label"]]
            pc["boxes"] += 1
            pc[m["period"]] += 1
            cam_class[m["camera"]][b["label"]] += 1
            if m["size"]:
                W, H = m["size"]
                side = max(b["bbox"][2] * W, b["bbox"][3] * H)
                bucket = "small" if side < SMALL_PX else ("large" if side > LARGE_PX else "medium")
                sizes[bucket] += 1
                pc[bucket] += 1

    # Diversity: mean pairwise cosine distance of the box embeddings of each class (0 = all identical)
    labels = np.array([b["label"] for b in boxes])
    for c, pc in per_class.items():
        pc["diversity"] = _mean_pairwise_cos_dist(emb[labels == c]) if len(boxes) else None

    n_img = len(all_images)
    n_bg = len(backgrounds)
    leaks = sum(1 for i, j, s in dup_pairs if s >= dup_threshold and boxes[i]["split"] != boxes[j]["split"])
    dist = {
        "images": n_img,
        "backgrounds": n_bg,
        "background_pct": round(100.0 * n_bg / n_img, 1) if n_img else 0.0,
        "per_class": per_class,
        "per_camera": dict(sorted(per_camera.items())),
        "per_device": dict(sorted(per_device.items())),
        "per_period": per_period,
        "box_sizes": sizes,
        "camera_class_boxes": {k: dict(v) for k, v in sorted(cam_class.items())},
        "cross_split_duplicates": leaks,
        "rules": {"day_hours": list(DAY_HOURS), "night_brightness": NIGHT_BRIGHTNESS, "small_px": SMALL_PX, "large_px": LARGE_PX},
    }
    return dist, meta, all_images


def readiness_checklist(dist, n_flagged_unreviewed):
    """Pass/warn/fail checks before carving out a golden test set."""
    checks = []

    def add(name, status, reason):
        checks.append({"check": name, "status": status, "reason": reason})

    pc = dist["per_class"]
    low = {c: v["boxes"] for c, v in pc.items() if v["boxes"] < 30}
    add("Every class has >= 30 boxes",
        "pass" if not low else ("fail" if any(n < 10 for n in low.values()) else "warn"),
        ("all classes OK: " + ", ".join(f"{c}={v['boxes']}" for c, v in pc.items())) if not low else
        "too few boxes: " + ", ".join(f"{c}={n}" for c, n in low.items()))
    cams = dist["per_camera"]
    lowc = {k: v["images"] for k, v in cams.items() if v["images"] < 5}
    add("Every camera has >= 5 images", "pass" if not lowc else "warn",
        f"{len(cams)} cameras, all >= 5 images" if not lowc else
        f"{len(lowc)} of {len(cams)} cameras below 5: " + ", ".join(f"{k}={n}" for k, n in lowc.items()))
    big = {c: v for c, v in pc.items() if v["boxes"] >= 30}
    miss = [f"{c} (day {v['day']}, night {v['night']})" for c, v in big.items() if not v["day"] or not v["night"]]
    if not big:
        add("Day and night present for every class with >= 30 boxes", "warn", "no class has >= 30 boxes")
    else:
        add("Day and night present for every class with >= 30 boxes", "pass" if not miss else "warn",
            "; ".join(f"{c}: day {v['day']} / night {v['night']}" for c, v in big.items()) if not miss else "missing: " + "; ".join(miss))
    bp = dist["background_pct"]
    add("Backgrounds are 10-35% of images",
        "pass" if 10 <= bp <= 35 else ("fail" if dist["backgrounds"] == 0 else "warn"),
        f"{dist['backgrounds']} of {dist['images']} images ({bp}%)")
    nl = dist["cross_split_duplicates"]
    add("No near-duplicates across splits", "pass" if nl == 0 else "warn",
        "none" if nl == 0 else f"{nl} near-duplicate box pairs sit in different splits (review tab 4)")
    add("Mislabels / outliers reviewed", "pass" if n_flagged_unreviewed == 0 else "warn",
        "nothing left to review" if n_flagged_unreviewed == 0 else f"{n_flagged_unreviewed} flagged boxes not reviewed yet")
    return checks


def golden_split(dist_meta, all_images, boxes, emb, dup_pairs, dup_threshold, frac, output_dir, seed=0):
    """Stratified (camera x main class x day/night) pick of ~frac of the images, preferring images far apart
    in embedding space. Near-duplicate images are kept together (all in or all out). Writes golden_test.txt."""
    rng = np.random.default_rng(seed)
    idx_by_img = defaultdict(list)
    for i, b in enumerate(boxes):
        idx_by_img[b["image_path"]].append(i)
    pairs = [(boxes[i]["image_path"], boxes[j]["image_path"]) for i, j, s in dup_pairs if s >= dup_threshold]
    group_of = _duplicate_groups(list(all_images), pairs)
    groups = defaultdict(list)
    for p, g in group_of.items():
        groups[g].append(p)

    def stratum(p):
        m = dist_meta[p]
        labs = [b["label"] for b in all_images[p]["boxes"]]
        main = max(sorted(set(labs)), key=labs.count) if labs else "background"
        return (m["camera"], main, m["period"])

    def vec(paths):
        ii = [i for p in paths for i in idx_by_img[p]]
        if not ii:
            return None
        v = emb[ii].mean(0)
        return v / (np.linalg.norm(v) + 1e-8)

    strata = defaultdict(list)
    for g in groups:
        strata[stratum(g)].append(g)
    chosen = []
    for key in sorted(strata):
        gs = strata[key]
        n_imgs = sum(len(groups[g]) for g in gs)
        target = int(round(frac * n_imgs))
        if target == 0 and rng.random() < frac * n_imgs:  # small strata: include with probability ~frac*n
            target = 1
        if target == 0:
            continue
        vecs = [vec(groups[g]) for g in gs]
        picked, got = [], 0
        if all(v is not None for v in vecs):
            V = np.stack(vecs)
            first = int(np.argmin(V @ V.mean(0)))     # most atypical first
            picked, got = [first], len(groups[gs[first]])
            mind = 1 - V @ V[first]
            while got < target and len(picked) < len(gs):
                mind[picked] = -1
                nxt = int(np.argmax(mind))            # farthest from everything chosen so far
                picked.append(nxt)
                got += len(groups[gs[nxt]])
                mind = np.minimum(mind, 1 - V @ V[nxt])
        else:
            for k in rng.permutation(len(gs)):
                if got >= target:
                    break
                picked.append(int(k))
                got += len(groups[gs[k]])
        for k in picked:
            chosen.extend(groups[gs[k]])
    chosen = sorted(set(chosen))
    path = os.path.join(output_dir, "golden_test.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(chosen) + ("\n" if chosen else ""))
    return {"file": path, "fraction": frac, "images": len(chosen),
            "boxes": sum(len(all_images[p]["boxes"]) for p in chosen),
            "backgrounds": sum(1 for p in chosen if not all_images[p]["boxes"]), "strata": len(strata)}


# ==========================================================================
# REPORT OUTPUT
# ==========================================================================

def flags_for(item, params):
    flags = []
    if item.get("pred") and item["pred"] != item["label"] and (item.get("pred_conf") or 0) >= params["mislabel_conf"]:
        flags.append(f"Suspected mislabel (looks like '{item['pred']}', {item['pred_conf']:.0%})")
    if item["lof"] >= params["lof_threshold"]:
        flags.append(f"Unusual for its class (LOF {item['lof']:.2f})")
    flags.extend(item["issues"])
    return flags


def save_report(report, output_dir):
    """Writes the JSON + JS report the dashboard loads."""
    blob = json.dumps(report, separators=(",", ":"))
    tmp = os.path.join(output_dir, "curation_report.json.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(blob)
    os.replace(tmp, os.path.join(output_dir, "curation_report.json"))
    tmp = os.path.join(output_dir, "curation_report.js.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        f.write("window.CURATION_REPORT = " + blob + ";\n")
    os.replace(tmp, os.path.join(output_dir, "curation_report.js"))


def write_outputs(report, output_dir):
    import csv
    os.makedirs(output_dir, exist_ok=True)
    save_report(report, output_dir)

    params = report["params"]
    rows = []
    for it in report["items"]:
        flags = flags_for(it, params)
        rows.append({
            "crop_id": it["id"], "split": it["split"], "image": it["image"], "line": it["line"],
            "label": it["label"], "model_thinks": it["pred"] or "", "model_confidence": it["pred_conf"] or "",
            "confidence_in_label": it["given_conf"] if it["given_conf"] is not None else "",
            "lof_score": it["lof"], "flags": " | ".join(flags), "label_file": it["label_path"],
        })
    rows.sort(key=lambda r: (r["flags"] == "", r["confidence_in_label"] if r["confidence_in_label"] != "" else 1.0))
    csv_path = os.path.join(output_dir, "curation_report.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    for sf in DASHBOARD_FILES:
        shutil.copy2(os.path.join(SCRIPT_DIR, sf), os.path.join(output_dir, sf))

    n_mis = sum(1 for it in report["items"] if it.get("pred") and it["pred"] != it["label"] and (it.get("pred_conf") or 0) >= params["mislabel_conf"])
    n_out = sum(1 for it in report["items"] if it["lof"] >= params["lof_threshold"])
    n_iss = sum(1 for it in report["items"] if it["issues"])
    n_dup = sum(1 for p in report["duplicates"] if p[2] >= params["dup_threshold"])
    counts = defaultdict(int)
    for it in report["items"]:
        counts[it["label"]] += 1

    s = report["stats"]
    print("\n" + "=" * 52)
    print("  CLEANCROP SUMMARY -", report["dataset_name"])
    print("=" * 52)
    print(f"  Images: {s['images']}  (no boxes: {s['background_images']})   Boxes: {len(report['items'])}")
    print("  Classes: " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items(), key=lambda x: -x[1])))
    print(f"  Suspected mislabels:     {n_mis}")
    print(f"  Unusual boxes (outlier): {n_out}")
    print(f"  Broken / double boxes:   {n_iss}")
    print(f"  Near-duplicate pairs:    {n_dup}")
    dist = report.get("distribution")
    if dist:
        print(f"  Backgrounds (no boxes):  {dist['backgrounds']} of {dist['images']} images ({dist['background_pct']}%)")
        print("  Day / night images:      " + ", ".join(f"{k}={v['images']}" for k, v in dist["per_period"].items()))
        print("  Cameras (images):        " + ", ".join(f"{k}={v['images']}" for k, v in dist["per_camera"].items()))
        print("  Golden-test readiness:")
        for c in dist.get("readiness", []):
            print(f"    [{c['status'].upper():4}] {c['check']}: {c['reason']}")
    if report.get("golden_test"):
        g = report["golden_test"]
        print(f"  Golden test list:        {g['images']} images / {g['boxes']} boxes -> {g['file']}")
    if report["problems"]:
        print(f"  Unusable label lines:    {len(report['problems'])} (listed in curation_report.json)")
    print(f"  Spreadsheet: {csv_path}")
    print("=" * 52 + "\n")


# ==========================================================================
# DASHBOARD SERVER (static files + review state + frame preview + writeback)
# ==========================================================================

class CurationServer:
    def __init__(self, output_dir):
        self.output_dir = os.path.abspath(output_dir)
        self.report_path = os.path.join(self.output_dir, "curation_report.json")
        self.state_path = os.path.join(self.output_dir, "review_state.json")
        self.lock = threading.Lock()
        with open(self.report_path, "r", encoding="utf-8") as f:
            self.report = json.load(f)
        self.by_id = {it["id"]: it for it in self.report["items"]}

    def frame_jpeg(self, item_id):
        buf = BytesIO()
        self.frame_image(item_id).save(buf, "JPEG", quality=85)
        return buf.getvalue()

    def frame_image(self, item_id):
        """Full image with every box in its label file drawn; the requested box highlighted."""
        item = self.by_id[item_id]
        with Image.open(item["image_path"]) as im:
            img = ImageOps.exif_transpose(im).convert("RGB")
        W, H = img.size
        scale = min(1.0, 1600 / W)
        if scale < 1.0:
            img = img.resize((int(W * scale), int(H * scale)), Image.BILINEAR)
        w, h = img.size
        draw = ImageDraw.Draw(img)
        names = self.report["class_names"]
        palette = [(56, 189, 248), (244, 114, 182), (250, 204, 21), (167, 139, 250), (52, 211, 153), (251, 146, 60)]
        if os.path.exists(item["label_path"]):
            with open(item["label_path"], "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.split()
                    if len(parts) < 5:
                        continue
                    try:
                        c = int(float(parts[0]))
                        xc, yc, bw, bh = (float(p) for p in parts[1:5])
                    except ValueError:
                        continue
                    if max(abs(a - b) for a, b in zip((xc, yc, bw, bh), item["bbox"])) < 1e-4:
                        continue  # drawn last, highlighted
                    col = palette[c % len(palette)]
                    box = [(xc - bw / 2) * w, (yc - bh / 2) * h, (xc + bw / 2) * w, (yc + bh / 2) * h]
                    draw.rectangle(box, outline=col, width=1)
                    draw.text((box[0], max(0, box[1] - 11)), names[c] if 0 <= c < len(names) else str(c), fill=col)
        xc, yc, bw, bh = item["bbox"]
        box = [(xc - bw / 2) * w, (yc - bh / 2) * h, (xc + bw / 2) * w, (yc + bh / 2) * h]
        pad = 6
        draw.rectangle([box[0] - pad, box[1] - pad, box[2] + pad, box[3] + pad], outline=(57, 255, 20), width=3)
        return img

    def plain_jpeg(self, item_id):
        """Full image without drawings, max 1600 px wide (the box editor draws the boxes itself)."""
        item = self.by_id[item_id]
        with Image.open(item["image_path"]) as im:
            img = ImageOps.exif_transpose(im).convert("RGB")
        if img.width > 1600:
            img = img.resize((1600, round(img.height * 1600 / img.width)), Image.BILINEAR)
        buf = BytesIO()
        img.save(buf, "JPEG", quality=90)
        return buf.getvalue()

    def background_jpeg(self, rel, max_w=1600):
        """A background (no-box) image from report["backgrounds"], addressed as '<split>/<image>'."""
        bg = next((b for b in self.report.get("backgrounds", []) if f"{b['split']}/{b['image']}" == rel), None)
        if bg is None:
            raise KeyError(rel)
        with Image.open(bg["image_path"]) as im:
            img = ImageOps.exif_transpose(im).convert("RGB")
        max_w = max(64, min(int(max_w), 1600))
        if img.width > max_w:
            img = img.resize((max_w, round(img.height * max_w / img.width)), Image.BILINEAR)
        buf = BytesIO()
        img.save(buf, "JPEG", quality=85)
        return buf.getvalue()

    def image_labels(self, item_id):
        """All boxes of the image a crop comes from: [{class_id, bbox: [xc, yc, w, h] normalised}]."""
        item = self.by_id[item_id]
        boxes = []
        if os.path.exists(item["label_path"]):
            with open(item["label_path"], "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.split()
                    try:
                        boxes.append({"class_id": int(float(parts[0])), "bbox": [float(p) for p in parts[1:5]]})
                    except (ValueError, IndexError):
                        continue
        return {"image": f"{item['split']}/{item['image']}", "class_names": self.report["class_names"], "boxes": boxes}

    def edit_image(self, item_id, boxes):
        """
        Replace every box of one image (box editor). Backs up the label file, writes the new boxes, then
        re-syncs the report: crops whose box is unchanged keep their line, the others are marked removed
        (edited / new boxes get no crop until the next analysis run).
        """
        item = self.by_id[item_id]
        names = self.report["class_names"]
        lines = []
        for b in boxes:
            cid = b.get("class_id"); bb = b.get("bbox") or []
            if not isinstance(cid, int) or not (0 <= cid < len(names)) or len(bb) != 4:
                raise ValueError(f"invalid box {b!r}")
            xc, yc, w, h = (float(v) for v in bb)
            x1, y1, x2, y2 = max(0.0, xc - w / 2), max(0.0, yc - h / 2), min(1.0, xc + w / 2), min(1.0, yc + h / 2)
            if x2 - x1 < 1e-4 or y2 - y1 < 1e-4:
                raise ValueError(f"empty box {b!r}")
            lines.append(f"{cid} {(x1 + x2) / 2:.6f} {(y1 + y2) / 2:.6f} {x2 - x1:.6f} {y2 - y1:.6f}\n")
        label_path = item["label_path"]
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        rel = os.path.relpath(label_path, self.report["dataset_dir"])
        if rel.startswith(".."):
            rel = os.path.basename(label_path)
        backup = os.path.join(self.output_dir, "label_backups", stamp, rel)
        os.makedirs(os.path.dirname(backup), exist_ok=True)
        if os.path.exists(label_path):
            shutil.copy2(label_path, backup)
        with open(label_path, "w", encoding="utf-8") as f:
            f.writelines(lines)
        parsed = [(int(l.split()[0]), [float(v) for v in l.split()[1:5]]) for l in lines]
        kept = removed = 0
        for it in self.report["items"]:
            if it["label_path"] != label_path or it.get("removed"):
                continue
            match = next((n for n, (c, bb) in enumerate(parsed) if c == it["class_id"]
                          and max(abs(a - b) for a, b in zip(bb, it["bbox"])) < 2e-6), None)
            if match is None:
                it["removed"] = True; it["edited"] = True; removed += 1
            else:
                it["line"] = match; kept += 1
        save_report(self.report, self.output_dir)
        return {"status": "success", "boxes": len(lines), "backup": backup,
                "message": f"Saved {len(lines)} box(es) for {rel} ({kept} unchanged, {removed} edited/removed). Backup: {backup}"}

    def load_state(self):
        if os.path.exists(self.state_path):
            with open(self.state_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}

    def save_state(self, state):
        tmp = self.state_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(state, f)
        os.replace(tmp, self.state_path)

    def writeback(self, actions):
        """
        actions: [{"id": crop_id, "action": "DELETE" | "RELABEL", "new_class_id": int}]
        Backs up every touched label file, edits it, then updates the report so the
        dashboard reloads in sync with the files on disk.
        """
        names = self.report["class_names"]
        by_file = defaultdict(list)
        errors = []
        for act in actions:
            item = self.by_id.get(act.get("id"))
            if item is None or item.get("removed"):
                errors.append(f"Unknown or already removed crop: {act.get('id')}")
                continue
            if act["action"] == "RELABEL":
                cid = act.get("new_class_id")
                if not isinstance(cid, int) or not (0 <= cid < len(names)):
                    errors.append(f"{item['id']}: invalid class id {cid!r}")
                    continue
            elif act["action"] != "DELETE":
                errors.append(f"{item['id']}: unknown action {act['action']!r}")
                continue
            by_file[item["label_path"]].append((item, act))

        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_root = os.path.join(self.output_dir, "label_backups", stamp)
        dataset_dir = self.report["dataset_dir"]
        deleted = relabeled = 0

        for label_path, file_actions in by_file.items():
            with open(label_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
            rel = os.path.relpath(label_path, dataset_dir)
            if rel.startswith(".."):
                rel = os.path.basename(label_path)
            os.makedirs(os.path.dirname(os.path.join(backup_root, rel)), exist_ok=True)
            shutil.copy2(label_path, os.path.join(backup_root, rel))

            def bbox_of(line):
                parts = line.split()
                try:
                    return [float(p) for p in parts[1:5]] if len(parts) >= 5 else None
                except ValueError:
                    return None

            drop, new_class = set(), {}
            for item, act in file_actions:
                target = None
                if item["line"] < len(lines):
                    bb = bbox_of(lines[item["line"]])
                    if bb and max(abs(a - b) for a, b in zip(bb, item["bbox"])) < 1e-4:
                        target = item["line"]
                if target is None:  # file changed since the scan - fall back to coordinate match
                    best, best_d = None, 1e-3
                    for n, line in enumerate(lines):
                        bb = bbox_of(line)
                        if bb:
                            d = max(abs(a - b) for a, b in zip(bb, item["bbox"]))
                            if d < best_d:
                                best, best_d = n, d
                    target = best
                if target is None:
                    errors.append(f"{item['id']}: box not found in {label_path}")
                    continue
                if act["action"] == "DELETE":
                    drop.add(target)
                    item["removed"] = True
                    deleted += 1
                else:
                    cid = act["new_class_id"]
                    new_class[target] = cid
                    item["relabeled_from"] = item.get("relabeled_from") or item["label"]
                    item["class_id"], item["label"] = cid, names[cid]
                    relabeled += 1

            out = []
            for n, line in enumerate(lines):
                if n in drop:
                    continue
                if n in new_class:
                    parts = line.split()
                    line = " ".join([str(new_class[n])] + parts[1:]) + "\n"
                out.append(line)
            with open(label_path, "w", encoding="utf-8") as f:
                f.writelines(out)

            # Re-sync line numbers of the remaining items from this file
            for it in self.report["items"]:
                if it["label_path"] == label_path and not it.get("removed"):
                    for n, line in enumerate(out):
                        bb = bbox_of(line)
                        if bb and max(abs(a - b) for a, b in zip(bb, it["bbox"])) < 1e-4:
                            it["line"] = n
                            break

        save_report(self.report, self.output_dir)
        return {
            "status": "success" if not errors else ("partial" if (deleted or relabeled) else "error"),
            "message": f"Deleted {deleted} box(es), relabeled {relabeled} box(es) in {len(by_file)} label file(s).",
            "backup_dir": backup_root if by_file else None,
            "errors": errors,
        }


def make_handler(server):
    class Handler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *a, **kw):
            super().__init__(*a, directory=server.output_dir, **kw)

        def log_message(self, fmt, *args):
            pass  # keep the console readable

        def _json(self, obj, code=200):
            body = json.dumps(obj).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _body(self):
            return json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))).decode("utf-8") or "{}")

        def end_headers(self):
            # The report changes after writeback - never let the browser serve a stale copy
            if self.path.split("?")[0].endswith((".js", ".json", ".html", "/")):
                self.send_header("Cache-Control", "no-store")
            super().end_headers()

        def do_GET(self):
            url = urlparse(self.path)
            if url.path == "/api/frame":
                crop_id = parse_qs(url.query).get("id", [""])[0]
                try:
                    data = server.frame_jpeg(crop_id)
                except Exception as e:
                    return self._json({"error": str(e)}, 404)
                self.send_response(200)
                self.send_header("Content-Type", "image/jpeg")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
            elif url.path in ("/api/image", "/api/labels"):
                crop_id = parse_qs(url.query).get("id", [""])[0]
                try:
                    if url.path == "/api/labels":
                        with server.lock:
                            return self._json(server.image_labels(crop_id))
                    data = server.plain_jpeg(crop_id)
                except Exception as e:
                    return self._json({"error": str(e)}, 404)
                self.send_response(200)
                self.send_header("Content-Type", "image/jpeg")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
            elif url.path == "/api/bgimage":
                q = parse_qs(url.query)
                try:
                    data = server.background_jpeg(q.get("path", [""])[0], q.get("w", ["1600"])[0])
                except Exception as e:
                    return self._json({"error": str(e)}, 404)
                self.send_response(200)
                self.send_header("Content-Type", "image/jpeg")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
            elif url.path == "/api/state":
                with server.lock:
                    self._json(server.load_state())
            elif url.path == "/api/ping":
                self._json({"ok": True})
            else:
                super().do_GET()

        def do_POST(self):
            try:
                if self.path == "/api/state":
                    with server.lock:
                        server.save_state(self._body())
                    self._json({"status": "ok"})
                elif self.path == "/api/editimage":
                    body = self._body()
                    with server.lock:
                        result = server.edit_image(body.get("id", ""), body.get("boxes", []))
                    print(f"[CleanCrop] Box editor: {result['message']}")
                    self._json(result)
                elif self.path == "/api/writeback":
                    actions = self._body().get("actions", [])
                    with server.lock:
                        result = server.writeback(actions)
                        if result["status"] != "error":
                            server.save_state({})
                    print(f"[CleanCrop] Writeback: {result['message']}" + (f" Backup: {result['backup_dir']}" if result['backup_dir'] else ""))
                    for err in result["errors"]:
                        print(f"[CleanCrop]   ! {err}")
                    self._json(result)
                else:
                    self._json({"error": "not found"}, 404)
            except Exception as e:
                self._json({"status": "error", "message": str(e)}, 500)

    return Handler


def serve(output_dir, port, open_browser=True):
    server = CurationServer(output_dir)
    handler = make_handler(server)
    httpd = None
    for p in range(port, port + 20):
        try:
            httpd = http.server.ThreadingHTTPServer(("127.0.0.1", p), handler)
            port = p
            break
        except OSError:
            continue
    if httpd is None:
        raise SystemExit(f"No free port found between {port} and {port + 19}.")
    url = f"http://localhost:{port}/index.html"
    print(f"[CleanCrop] Dashboard running at {url}")
    print("[CleanCrop] Keep this window open while reviewing. Press Ctrl+C to stop.")
    if open_browser:
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[CleanCrop] Server stopped.")


# ==========================================================================
# EXPORT FLAGGED IMAGES TO A FOLDER (for checking in any image viewer)
# ==========================================================================

def _font(size):
    from PIL import ImageFont
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # Pillow < 10.1
        return ImageFont.load_default()


def _wrap(text, width_chars):
    import textwrap
    return textwrap.wrap(text, width_chars) or [""]


def _with_side_panel(frame, crop_path, title, lines, color):
    """Frame on the left, zoomed crop + explanation on the right."""
    panel_w = 360
    out = Image.new("RGB", (frame.width + panel_w, max(frame.height, 560)), (13, 17, 23))
    out.paste(frame, (0, 0))
    draw = ImageDraw.Draw(out)
    x = frame.width + 20
    draw.text((x, 16), title, fill=color, font=_font(24))
    if os.path.exists(crop_path):
        with Image.open(crop_path) as c:
            out.paste(c.convert("RGB").resize((320, 320), Image.BICUBIC), (x, 56))
    y = 392
    for line in lines:
        for part in _wrap(line, 30):
            draw.text((x, y), part, fill=(201, 209, 217), font=_font(17))
            y += 22
        y += 6
    return out


def export_review_folder(output_dir):
    """
    Writes every box that still needs checking (using the dashboard's current
    thresholds and skipping boxes you already reviewed) into to_verify/<problem>/
    as an annotated full image, plus a CSV index pointing at the label file + line.
    """
    import csv
    server = CurationServer(output_dir)
    report = server.report
    state = server.load_state()
    th = state.get("thresholds") or {}
    p = report["params"]
    mis_conf = th.get("mis", p["mislabel_conf"])
    lof_thr = th.get("lof", p["lof_threshold"])
    dup_thr = th.get("dup", p["dup_threshold"])
    decided = set(state.get("deleted", [])) | set(state.get("kept", [])) | set((state.get("relabeled") or {}).keys())
    ignored_dups = set(state.get("ignoredDuplicates", []))

    root = os.path.join(server.output_dir, "to_verify")
    if os.path.isdir(root):
        shutil.rmtree(root)
    folders = {k: os.path.join(root, k) for k in ("1_wrong_labels", "2_double_boxes", "3_unusual_boxes", "4_near_duplicates")}
    for f in folders.values():
        os.makedirs(f)

    items = [it for it in report["items"] if not it.get("removed")]
    open_items = [it for it in items if it["id"] not in decided and not it.get("relabeled_from")]
    mislabels = sorted((it for it in open_items if it.get("pred") and it["pred"] != it["label"] and (it.get("pred_conf") or 0) >= mis_conf),
                       key=lambda it: -it["pred_conf"])
    issues = [it for it in items if it["issues"] and it["id"] not in decided]
    unusual = sorted((it for it in open_items if it["lof"] >= lof_thr), key=lambda it: -it["lof"])
    all_items = report["items"]
    dups = []
    for i, j, sim in report["duplicates"]:
        a, b = all_items[i], all_items[j]
        if sim >= dup_thr and not (a.get("removed") or b.get("removed")) and a["id"] not in decided and b["id"] not in decided \
                and f"{a['id']}|{b['id']}" not in ignored_dups:
            dups.append((a, b, sim))
    dups.sort(key=lambda d: (d[0]["split"] == d[1]["split"], -d[2]))

    rows = []

    def row(folder, fname, problem, it, suggestion=""):
        rows.append({"folder": folder, "file": fname, "problem": problem, "box_id": it["id"], "split": it["split"],
                     "label": it["label"], "suggestion": suggestion, "image": it["image_path"],
                     "label_file": it["label_path"], "line_in_label_file": it["line"] + 1})

    tasks = [("1_wrong_labels", it) for it in mislabels] + [("2_double_boxes", it) for it in issues] + \
            [("3_unusual_boxes", it) for it in unusual]
    for folder, it in tqdm(tasks, desc="Exporting boxes", unit="img"):
        crop = os.path.join(server.output_dir, it["path"])
        if folder == "1_wrong_labels":
            idx = mislabels.index(it) + 1
            title = f"'{it['label']}' -> '{it['pred']}'?"
            lines = [f"Labeled: {it['label']}", f"Looks like: {it['pred']} ({it['pred_conf']:.0%} sure)",
                     f"{it['split']}/{it['image']}", f"Label file line {it['line'] + 1}"]
            fname = f"{idx:03d}_labeled-{it['label']}_looks-like-{it['pred']}_{it['pred_conf']:.0%}_{it['id']}.jpg"
            row(folder, fname, "Suspected wrong label", it, it["pred"])
            color = (245, 158, 11)
        elif folder == "2_double_boxes":
            idx = issues.index(it) + 1
            title = "Box problem"
            lines = list(it["issues"]) + [f"{it['split']}/{it['image']}", f"Label file line {it['line'] + 1}"]
            fname = f"{idx:03d}_{it['label']}_{it['id']}.jpg"
            row(folder, fname, "; ".join(it["issues"]), it, "delete this copy" if "Duplicate box" in " ".join(it["issues"]) else "")
            color = (167, 139, 250)
        else:
            idx = unusual.index(it) + 1
            title = f"Unusual '{it['label']}'"
            lines = [f"Unusual score: {it['lof']:.2f}", "Is this really a " + it["label"] + "?",
                     f"{it['split']}/{it['image']}", f"Label file line {it['line'] + 1}"]
            fname = f"{idx:03d}_score-{it['lof']:.2f}_{it['label']}_{it['id']}.jpg"
            row(folder, fname, f"Unusual for its class (score {it['lof']:.2f})", it)
            color = (244, 63, 94)
        img = _with_side_panel(server.frame_image(it["id"]), crop, title, lines, color)
        img.save(os.path.join(folders[folder], fname), "JPEG", quality=88)

    for n, (a, b, sim) in enumerate(tqdm(dups, desc="Exporting duplicate pairs", unit="pair"), 1):
        fa, fb = server.frame_image(a["id"]), server.frame_image(b["id"])
        w = 900
        fa = fa.resize((w, int(fa.height * w / fa.width)))
        fb = fb.resize((w, int(fb.height * w / fb.width)))
        out = Image.new("RGB", (2 * w + 20, max(fa.height, fb.height) + 60), (13, 17, 23))
        out.paste(fa, (0, 60))
        out.paste(fb, (w + 20, 60))
        leak = a["split"] != b["split"]
        d = ImageDraw.Draw(out)
        d.text((12, 14), f"{sim:.1%} similar   {'TRAIN/VAL LEAK   ' if leak else ''}A: {a['split']}/{a['image']}   B: {b['split']}/{b['image']}",
               fill=(244, 63, 94) if leak else (0, 242, 254), font=_font(22))
        fname = f"{n:03d}_{'LEAK_' if leak else ''}{sim:.3f}_{a['id']}__{b['id']}.jpg"
        out.save(os.path.join(folders["4_near_duplicates"], fname), "JPEG", quality=85)
        row("4_near_duplicates", fname, f"Near-duplicate ({sim:.1%}){' train/val leak' if leak else ''} - box A", a)
        row("4_near_duplicates", fname, f"Near-duplicate ({sim:.1%}){' train/val leak' if leak else ''} - box B", b)

    csv_path = os.path.join(root, "to_verify.csv")
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["folder", "file", "problem", "box_id", "split", "label", "suggestion",
                                               "image", "label_file", "line_in_label_file"])
        writer.writeheader()
        writer.writerows(rows)
    with open(os.path.join(root, "README.txt"), "w", encoding="utf-8") as f:
        f.write(
            "CleanCrop - boxes to verify\n"
            "===========================\n"
            "Green thick box = the box in question. Thin coloured boxes = the other boxes in that image.\n"
            "The right panel shows the box zoomed in and why it was flagged.\n\n"
            "1_wrong_labels     The class looks wrong. Check the suggestion.\n"
            "2_double_boxes     The same object is boxed twice (delete the copy) or has two different labels.\n"
            "3_unusual_boxes    Does not look like the rest of its class: other bird/animal, background, bad box.\n"
            "4_near_duplicates  Almost identical boxes in two images. LEAK = one is in train, the other in val.\n\n"
            "to_verify.csv lists every file with the image path, label file and line number to edit.\n"
            "Easiest way to fix: open the CleanCrop dashboard (CleanCrop.bat -> Continue reviewing)\n"
            "and use Keep / Relabel / Delete there; it edits the label files for you and keeps backups.\n")

    print(f"\nExported to {root}")
    print(f"  1_wrong_labels:    {len(mislabels)}")
    print(f"  2_double_boxes:    {len(issues)}")
    print(f"  3_unusual_boxes:   {len(unusual)}")
    print(f"  4_near_duplicates: {len(dups)} pairs ({sum(1 for a, b, _ in dups if a['split'] != b['split'])} train/val leaks)")
    return root


# ==========================================================================
# LABELING-TOOL ROUND TRIP (X-AnyLabeling, LabelImg, CVAT ... any YOLO tool)
# ==========================================================================

LABELING_PREFIX = {"1_wrong_labels": "1-wrong", "2_double_boxes": "2-double", "3_unusual_boxes": "3-unusual", "4_near_duplicates": "4-duplicate"}


def export_for_labeling(output_dir):
    """
    Copies the original images that contain flagged boxes + their YOLO labels into
    for_labeling/{images,labels} with classes.txt, ready to open in a labeling tool.
    File names get a problem prefix so they sort by problem type.
    """
    import csv
    # Regenerate so the list matches the current review state and thresholds
    root_verify = export_review_folder(output_dir)
    csv_path = os.path.join(root_verify, "to_verify.csv")
    with open(csv_path, "r", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    with open(os.path.join(output_dir, "curation_report.json"), "r", encoding="utf-8") as f:
        class_names = json.load(f)["class_names"]

    per_image = {}
    for r in rows:  # prefix = most important problem per image (folders are numbered by priority)
        info = per_image.setdefault(r["image"], {"folder": r["folder"], "label_file": r["label_file"], "problems": []})
        info["folder"] = min(info["folder"], r["folder"])
        info["problems"].append(f"line {r['line_in_label_file']}: {r['problem']}")

    root = os.path.join(os.path.abspath(output_dir), "for_labeling")
    if os.path.isdir(root):
        edited = _edited_since_export(root)
        if edited:
            raise SystemExit(f"{len(edited)} annotation file(s) in {root} were edited and not copied back yet "
                             f"(e.g. {os.path.basename(edited[0])}).\nRun 'Copy fixed labels back' first, or delete that folder to discard the edits.")
        shutil.rmtree(root)
    os.makedirs(os.path.join(root, "images"))
    os.makedirs(os.path.join(root, "labels"))
    with open(os.path.join(root, "classes.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(class_names) + "\n")

    flagged_lines = defaultdict(dict)  # label file -> {line index: problem}
    for r in rows:
        line = int(r["line_in_label_file"]) - 1
        prev = flagged_lines[r["label_file"]].get(line)
        flagged_lines[r["label_file"]][line] = f"{prev}; {r['problem']}" if prev else r["problem"]

    manifest, listing = {}, []
    for img_path, info in sorted(per_image.items(), key=lambda kv: (kv[1]["folder"], os.path.basename(kv[0]))):
        stem = f"{LABELING_PREFIX[info['folder']]}__{os.path.splitext(os.path.basename(img_path))[0]}"
        ext = os.path.splitext(img_path)[1]
        shutil.copy2(img_path, os.path.join(root, "images", stem + ext))
        if os.path.exists(info["label_file"]):
            shutil.copy2(info["label_file"], os.path.join(root, "labels", stem + ".txt"))
        _write_xanylabeling_json(os.path.join(root, "images", stem + ext), info["label_file"], class_names,
                                 flagged_lines[info["label_file"]])
        manifest[stem] = {"image": img_path, "label_file": info["label_file"]}
        listing.append({"file": stem + ext, "problems": " | ".join(dict.fromkeys(info["problems"])), "original_image": img_path})

    with open(os.path.join(root, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=1)
    with open(os.path.join(root, "what_to_check.csv"), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=["file", "problems", "original_image"])
        w.writeheader()
        w.writerows(listing)
    with open(os.path.join(root, "HOW_TO.txt"), "w", encoding="utf-8") as f:
        f.write(
            "Fix these images in X-AnyLabeling (or any YOLO labeling tool)\n"
            "=============================================================\n"
            f"{len(manifest)} images. File name prefix = main problem:\n"
            "  1-wrong     a box probably has the wrong class\n"
            "  2-double    the same object is boxed twice - delete the extra box\n"
            "  3-unusual   a box does not look like its class - check it\n"
            "  4-duplicate near-identical image in train and val (no label edit needed;\n"
            "              consider deleting one of the pair from the dataset)\n"
            "what_to_check.csv says which line / box in each image is suspicious.\n"
            "The ../to_verify folder has the same boxes highlighted in green.\n\n"
            "X-AnyLabeling:\n"
            "  1. File > Open Dir  ->  this folder's 'images' subfolder.\n"
            "     The boxes load automatically from the .json file next to each image.\n"
            "     Suspicious boxes have the reason in their description (double-click the box).\n"
            "  2. Fix the boxes (also add any pigeons that have no box!). Use the class names\n"
            "     from classes.txt exactly. Save each image (Ctrl+S, or turn on auto-save).\n"
            "  3. Run CleanCrop.bat, choose [5] 'Copy fixed labels back into the dataset'\n"
            "     (only changed files are copied; originals are backed up first).\n\n"
            "Other YOLO tools: the 'labels' folder has the same annotations as YOLO .txt files.\n"
            "If a .json exists for an image it wins over the .txt when copying back.\n")
    print(f"\n{len(manifest)} images copied to {root}")
    return root


def _write_xanylabeling_json(image_path, label_file, class_names, flagged):
    """X-AnyLabeling / LabelMe JSON next to the image, with the YOLO boxes as rectangles."""
    with Image.open(image_path) as im:
        W, H = ImageOps.exif_transpose(im).size
    shapes = []
    if os.path.exists(label_file):
        with open(label_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
        for n, line in enumerate(lines):
            parts = line.split()
            try:
                c = int(float(parts[0]))
                xc, yc, w, h = (float(p) for p in parts[1:5])
            except (ValueError, IndexError):
                continue
            x1, y1, x2, y2 = (xc - w / 2) * W, (yc - h / 2) * H, (xc + w / 2) * W, (yc + h / 2) * H
            shapes.append({
                "label": class_names[c] if 0 <= c < len(class_names) else str(c),
                "score": None,
                "points": [[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
                "group_id": None,
                "description": f"CHECK: {flagged[n]}" if n in flagged else "",
                "difficult": n in flagged,
                "shape_type": "rectangle",
                "flags": {},
                "attributes": {},
                "kie_linking": [],
            })
    doc = {"version": "4.0.3", "flags": {}, "checked": False, "shapes": shapes,
           "imagePath": os.path.basename(image_path), "imageData": None, "imageHeight": H, "imageWidth": W}
    with open(os.path.splitext(image_path)[0] + ".json", "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=2)


def _json_to_yolo(json_path, class_names):
    """Returns (yolo_text, problems) from an X-AnyLabeling / LabelMe JSON."""
    with open(json_path, "r", encoding="utf-8") as f:
        doc = json.load(f)
    W, H = doc.get("imageWidth"), doc.get("imageHeight")
    out, problems = [], []
    for s in doc.get("shapes", []):
        if s.get("shape_type") not in ("rectangle", "polygon") or len(s.get("points", [])) < 2:
            problems.append(f"skipped a '{s.get('shape_type')}' shape")
            continue
        label = s.get("label", "")
        if label not in class_names:
            problems.append(f"unknown class '{label}' (allowed: {', '.join(class_names)})")
            continue
        xs = [min(max(p[0], 0), W) for p in s["points"]]
        ys = [min(max(p[1], 0), H) for p in s["points"]]
        x1, x2, y1, y2 = min(xs), max(xs), min(ys), max(ys)
        if x2 - x1 < 1 or y2 - y1 < 1:
            problems.append("skipped a zero-size box")
            continue
        out.append(f"{class_names.index(label)} {(x1 + x2) / 2 / W:.6f} {(y1 + y2) / 2 / H:.6f} {(x2 - x1) / W:.6f} {(y2 - y1) / H:.6f}")
    return ("\n".join(out) + "\n") if out else "", problems


def _edited_since_export(root):
    """Annotation files in for_labeling changed after it was exported (= work that would be lost)."""
    manifest_path = os.path.join(root, "manifest.json")
    if not os.path.exists(manifest_path):
        return []
    t0 = os.path.getmtime(manifest_path) + 2
    edited = []
    for sub in ("images", "labels"):
        d = os.path.join(root, sub)
        if os.path.isdir(d):
            edited += [os.path.join(d, f) for f in os.listdir(d)
                       if f.endswith((".json", ".txt")) and os.path.getmtime(os.path.join(d, f)) > t0]
    synced_marker = os.path.join(root, ".synced")
    if os.path.exists(synced_marker):
        ts = os.path.getmtime(synced_marker)
        edited = [e for e in edited if os.path.getmtime(e) > ts]

    def has_content(path):  # labeling tools auto-save empty files for images they opened without boxes
        try:
            if path.endswith(".json"):
                with open(path, "r", encoding="utf-8") as f:
                    return bool(json.load(f).get("shapes"))
            return os.path.getsize(path) > 0
        except (OSError, ValueError):
            return True
    return [e for e in edited if has_content(e)]


def _parse_label_lines(text):
    out = []
    for line in text.splitlines():
        parts = line.split()
        if len(parts) >= 5:
            try:
                out.append((int(float(parts[0])), *[round(float(p), 4) for p in parts[1:5]]))
            except ValueError:
                out.append(line.strip())
    return sorted(out, key=str)


def sync_labels_back(output_dir):
    """Copies label files edited in for_labeling/labels back into the dataset (with backups)."""
    root = os.path.join(os.path.abspath(output_dir), "for_labeling")
    manifest_path = os.path.join(root, "manifest.json")
    if not os.path.exists(manifest_path):
        raise SystemExit(f"Nothing to copy back: {manifest_path} not found.")
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    with open(os.path.join(output_dir, "curation_report.json"), "r", encoding="utf-8") as f:
        report_head = json.load(f)
    dataset_dir, class_names = report_head["dataset_dir"], report_head["class_names"]

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_root = os.path.join(os.path.abspath(output_dir), "label_backups", stamp)
    changed = unchanged = missing = 0
    refused = []
    for stem, info in manifest.items():
        json_path = os.path.join(root, "images", stem + ".json")
        edited = os.path.join(root, "labels", stem + ".txt")
        if os.path.exists(json_path):  # X-AnyLabeling edits live in the JSON
            new_text, problems = _json_to_yolo(json_path, class_names)
            if any(p.startswith("unknown class") for p in problems):
                refused.append(f"{stem}: " + "; ".join(problems))
                continue
            for p in problems:
                print(f"  ! {stem}: {p}")
        elif os.path.exists(edited):
            with open(edited, "r", encoding="utf-8") as f:
                new_text = f.read()
        else:
            missing += 1
            continue
        old_text = ""
        if os.path.exists(info["label_file"]):
            with open(info["label_file"], "r", encoding="utf-8") as f:
                old_text = f.read()
        if _parse_label_lines(new_text) == _parse_label_lines(old_text):
            unchanged += 1
            continue
        if not new_text.strip() and _parse_label_lines(old_text):
            # Usually an empty file auto-saved by the tool before the boxes were loaded
            refused.append(f"{stem}: has no boxes but the dataset has {len(_parse_label_lines(old_text))}. "
                           f"If you really want to remove all boxes, delete them in the dataset label file by hand.")
            continue
        if os.path.exists(info["label_file"]):
            rel = os.path.relpath(info["label_file"], dataset_dir)
            if rel.startswith(".."):
                rel = os.path.basename(info["label_file"])
            os.makedirs(os.path.dirname(os.path.join(backup_root, rel)), exist_ok=True)
            shutil.copy2(info["label_file"], os.path.join(backup_root, rel))
        os.makedirs(os.path.dirname(info["label_file"]), exist_ok=True)
        with open(info["label_file"], "w", encoding="utf-8") as f:
            f.write(new_text)
        changed += 1
        print(f"  updated {info['label_file']}")

    with open(os.path.join(root, ".synced"), "w") as f:
        f.write(stamp)
    print(f"\nCopied back {changed} changed label file(s); {unchanged} unchanged" + (f"; {missing} had no label file (skipped)" if missing else "") + ".")
    if refused:
        print(f"\nNOT copied ({len(refused)} image(s) - fix and run this again):")
        for r in refused:
            print(f"  - {r}")
    if changed:
        print(f"Originals backed up in {backup_root}")
        print("Run 'Analyse again' so the dashboard reflects the new labels.")


# ==========================================================================
# MAIN
# ==========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="CleanCrop: find and fix bad labels in a YOLO dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("dataset", nargs="?", help="Path to the YOLO dataset folder (contains data.yaml)")
    parser.add_argument("--dataset-dir", help="Same as the positional dataset argument")
    parser.add_argument("--output-dir", help="Where to write results (default: curation_output/<dataset name> next to this script)")
    parser.add_argument("--model", choices=["dinov2", "clip"], default="dinov2", help="Embedding model")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--dup-threshold", type=float, default=0.97, help="Initial near-duplicate similarity threshold")
    parser.add_argument("--lof-threshold", type=float, default=1.5, help="Initial outlier (LOF) threshold")
    parser.add_argument("--mislabel-conf", type=float, default=0.7, help="Initial confidence needed to flag a mislabel")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--serve-only", action="store_true", help="Open the dashboard for an existing analysis without recomputing")
    parser.add_argument("--no-serve", action="store_true", help="Only run the analysis, don't start the dashboard")
    parser.add_argument("--no-browser", action="store_true", help="Don't open the browser automatically")
    parser.add_argument("--no-cache", action="store_true", help="Recompute all embeddings")
    parser.add_argument("--export-folder", action="store_true", help="Export every flagged box as an annotated image into <output>/to_verify, then exit")
    parser.add_argument("--export-labeling", action="store_true", help="Copy flagged images + YOLO labels into <output>/for_labeling for X-AnyLabeling etc., then exit")
    parser.add_argument("--sync-labels", action="store_true", help="Copy labels edited in <output>/for_labeling back into the dataset (with backups), then exit")
    parser.add_argument("--golden-split", type=float, default=None, metavar="FRAC",
                        help="Also write golden_test.txt: a stratified (camera x class x day/night), diverse ~FRAC of the images; near-duplicates stay together")
    parser.add_argument("--serve", action="store_true", help=argparse.SUPPRESS)  # kept for old commands; serving is the default now
    args = parser.parse_args()

    dataset_dir = args.dataset or args.dataset_dir
    if not dataset_dir:
        parser.error("please give the dataset folder, e.g.  python dataset_curator_yolo.py C:\\data\\my_yolo_dataset")
    dataset_dir = os.path.abspath(dataset_dir.strip('"'))
    if not os.path.isdir(dataset_dir):
        raise SystemExit(f"Dataset folder not found: {dataset_dir}")
    output_dir = os.path.abspath(args.output_dir or os.path.join(SCRIPT_DIR, "curation_output", os.path.basename(os.path.normpath(dataset_dir))))

    if args.export_folder or args.export_labeling or args.sync_labels:
        if not os.path.exists(os.path.join(output_dir, "curation_report.json")):
            raise SystemExit(f"No analysis found in {output_dir}. Run the analysis first.")
        if args.sync_labels:
            sync_labels_back(output_dir)
            return
        root = export_for_labeling(output_dir) if args.export_labeling else export_review_folder(output_dir)
        if os.name == "nt" and not args.no_browser:
            os.startfile(root)
        return

    if args.serve_only:
        if not os.path.exists(os.path.join(output_dir, "curation_report.json")):
            raise SystemExit(f"No analysis found in {output_dir}. Run without --serve-only first.")
        for sf in DASHBOARD_FILES:  # pick up dashboard updates
            shutil.copy2(os.path.join(SCRIPT_DIR, sf), os.path.join(output_dir, sf))
        serve(output_dir, args.port, not args.no_browser)
        return

    t0 = time.time()
    print(f"[1/4] Scanning dataset: {dataset_dir}")
    class_names = load_class_names(dataset_dir)
    if not class_names:
        print("  ! No data.yaml / classes.txt found - classes will be shown as class_0, class_1, ...")
    boxes, stats, problems = scan_dataset(dataset_dir, class_names)
    print(f"  Classes: {class_names}")
    print(f"  {stats['images']} images, {stats['boxes']} boxes, splits: " + ", ".join(f"{k}={v['images']}" for k, v in stats["splits"].items()))
    for p in problems[:10]:
        print(f"  ! {p['label_path']} line {p['line'] + 1}: {p['problem']}")
    if not boxes:
        raise SystemExit("No boxes found in the label files.")

    print(f"[2/4] Building crops and {args.model.upper()} embeddings -> {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    boxes, emb = build_crops_and_embeddings(boxes, output_dir, args.model, args.batch_size, use_cache=not args.no_cache)

    args.output_dir_resolved = output_dir
    if args.golden_split is not None and not (0 < args.golden_split < 1):
        raise SystemExit("--golden-split must be between 0 and 1, e.g. 0.15")
    print("[3/4] Analysing")
    report = analyze(boxes, emb, class_names, args, stats, problems, dataset_dir)

    print("[4/4] Writing report")
    write_outputs(report, output_dir)
    # Fresh analysis: earlier unsaved review decisions refer to old crops
    state_path = os.path.join(output_dir, "review_state.json")
    if os.path.exists(state_path):
        os.remove(state_path)
        print("  Note: un-committed review decisions from the previous run were cleared.")
    print(f"Done in {time.time() - t0:.0f}s.")

    if not args.no_serve:
        serve(output_dir, args.port, not args.no_browser)


if __name__ == "__main__":
    main()
