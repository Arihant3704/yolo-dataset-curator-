#!/usr/bin/env python3
import os
import sys
import yaml
import argparse
import shutil
import json
import torch
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModel, CLIPProcessor, CLIPModel
from sklearn.neighbors import LocalOutlierFactor

def load_yaml_config(yaml_path):
    """Loads classes map from data.yaml file."""
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    # data.yaml names are list of classes
    names = data.get('names', [])
    if isinstance(names, dict):
        # Convert dictionary format to list
        names = [names[i] for i in sorted(names.keys())]
    return names

def parse_yolo_dataset(dataset_dir, class_names):
    """
    Scans the YOLO dataset structure and maps images to label coordinates.
    Returns: list of dicts containing image paths, labels, and bounding box dimensions.
    """
    splits = ['train', 'valid', 'test']
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
    
    crops_metadata = []
    crop_counter = 0
    
    for split in splits:
        split_dir = os.path.join(dataset_dir, split)
        images_dir = os.path.join(split_dir, 'images')
        labels_dir = os.path.join(split_dir, 'labels')
        
        if not os.path.exists(images_dir):
            continue
            
        print(f"Scanning split '{split}' images...")
        img_files = [f for f in os.listdir(images_dir) if f.lower().endswith(image_extensions)]
        
        for img_file in tqdm(img_files):
            img_path = os.path.join(images_dir, img_file)
            
            # Find corresponding label file
            base_name, _ = os.path.splitext(img_file)
            label_file = base_name + '.txt'
            label_path = os.path.join(labels_dir, label_file)
            
            if not os.path.exists(label_path):
                continue
                
            # Read label boxes
            with open(label_path, 'r') as f:
                lines = f.readlines()
                
            for box_idx, line in enumerate(lines):
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                    
                class_id = int(parts[0])
                class_name = class_names[class_id] if class_id < len(class_names) else f"class_{class_id}"
                
                x_center = float(parts[1])
                y_center = float(parts[2])
                w = float(parts[3])
                h = float(parts[4])
                
                crops_metadata.append({
                    "crop_idx": box_idx,
                    "crop_id": f"{base_name}_crop_{box_idx}",
                    "original_image_name": img_file,
                    "original_image_path": img_path,
                    "split": split,
                    "class_id": class_id,
                    "class_name": class_name,
                    "bbox_norm": [x_center, y_center, w, h]
                })
                crop_counter += 1
                
    print(f"Found {crop_counter} bounding box crops across all splits.")
    return crops_metadata

def crop_and_resize_images(crops_metadata, output_crops_dir):
    """
    Crops images to bounding box dimensions, saves crops, and adds paths to metadata.
    """
    os.makedirs(output_crops_dir, exist_ok=True)
    print("Extracting and saving crops...")
    
    valid_crops = []
    for meta in tqdm(crops_metadata):
        img_path = meta["original_image_path"]
        
        try:
            with Image.open(img_path) as img:
                img_w, img_h = img.size
                
                # Decode normalized coordinates
                xc, yc, w, h = meta["bbox_norm"]
                xmin = int((xc - w / 2.0) * img_w)
                ymin = int((yc - h / 2.0) * img_h)
                xmax = int((xc + w / 2.0) * img_w)
                ymax = int((yc + h / 2.0) * img_h)
                
                # Clip coordinates
                xmin = max(0, xmin)
                ymin = max(0, ymin)
                xmax = min(img_w, xmax)
                ymax = min(img_h, ymax)
                
                if xmax <= xmin or ymax <= ymin:
                    continue # Skip invalid box sizes
                    
                # Crop and save crop
                crop = img.crop((xmin, ymin, xmax, ymax))
                
                # Save crop with recognizable name
                crop_filename = f"{meta['crop_id']}_{meta['class_name']}.jpg"
                crop_path = os.path.join(output_crops_dir, crop_filename)
                crop.save(crop_path, "JPEG")
                
                # Add path relative to output directory for frontend loading
                meta["crop_path"] = os.path.join("demo_dataset", "crops", crop_filename)
                meta["crop_abs_path"] = crop_path
                valid_crops.append(meta)
                
        except Exception as e:
            print(f"Error cropping {img_path}: {e}")
            
    print(f"Successfully extracted {len(valid_crops)} crop images.")
    return valid_crops

def extract_embeddings(crops_metadata, model_mode="dinov2", batch_size=32, device="cuda"):
    """
    Extracts visual embeddings from cropped images using CLIP or DINOv2.
    """
    print(f"Extracting embeddings using {model_mode.upper()} on {device}...")
    
    # Load model and processor
    if model_mode == "dinov2":
        processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
        model = AutoModel.from_pretrained("facebook/dinov2-base").to(device)
    else:  # clip
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
        
    model.eval()
    embeddings = []
    
    # Batch processing
    for i in tqdm(range(0, len(crops_metadata), batch_size)):
        batch = crops_metadata[i:i + batch_size]
        images = []
        
        for meta in batch:
            try:
                img = Image.open(meta["crop_abs_path"]).convert("RGB")
                images.append(img)
            except Exception as e:
                print(f"Error opening crop image {meta['crop_abs_path']}: {e}")
                
        if not images:
            continue
            
        with torch.no_grad():
            if model_mode == "dinov2":
                inputs = processor(images=images, return_tensors="pt").to(device)
                outputs = model(**inputs)
                # DINOv2 cls token is at index 0 of sequence length
                batch_emb = outputs.last_hidden_state[:, 0].cpu().numpy()
            else: # clip
                inputs = processor(images=images, return_tensors="pt").to(device)
                batch_emb = model.get_image_features(**inputs).cpu().numpy()
                
        embeddings.append(batch_emb)
        
    embeddings = np.vstack(embeddings)
    
    # L2 Normalization
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / (norms + 1e-9)
    
    print(f"Extracted embedding vectors of shape {embeddings.shape}.")
    return embeddings

def detect_quality_anomalies(embeddings, crops_metadata, dup_threshold=0.95, outlier_neighbors=20, lof_threshold=1.5, mislabel_k=5, mislabel_consensus=0.5):
    """
    Runs duplicate, outlier, and mislabeled curation checks.
    """
    N = len(crops_metadata)
    labels = np.array([m["class_name"] for m in crops_metadata])
    ids = [m["crop_id"] for m in crops_metadata]
    
    # Initialize flags
    is_duplicate = [False] * N
    is_outlier = [False] * N
    is_mislabeled = [False] * N
    
    duplicate_pairs = []
    duplicate_indices = set()
    
    # 1. Near-Duplicates (Cosine similarity threshold check)
    print("Detecting duplicate pairs...")
    for i in range(N):
        for j in range(i+1, N):
            # Compute dot product of L2 normalized embeddings
            sim = float(np.dot(embeddings[i], embeddings[j]))
            if sim >= dup_threshold:
                is_duplicate[i] = True
                is_duplicate[j] = True
                duplicate_indices.add(i)
                duplicate_indices.add(j)
                duplicate_pairs.append({
                    "pairKey": f"{ids[i]}-{ids[j]}" if ids[i] < ids[j] else f"{ids[j]}-{ids[i]}",
                    "similarity": sim,
                    "indexA": i,
                    "indexB": j,
                    "idA": ids[i],
                    "idB": ids[j],
                    "label": labels[i]
                })
                
    # 2. Outliers (Local Outlier Factor - density based unsupervised algorithm)
    print("Detecting class outliers using Local Outlier Factor...")
    outlier_scores_centroid = np.zeros(N)
    outlier_scores_knn = np.zeros(N)
    
    # We do density check per-class to avoid domain offset
    unique_classes = np.unique(labels)
    
    for cls in unique_classes:
        class_idx = np.where(labels == cls)[0]
        if len(class_idx) < 5:
            print(f"Skipping LOF for '{cls}' (too few samples: {len(class_idx)}).")
            continue
            
        class_embs = embeddings[class_idx]
        
        # Fit LOF (density algorithm)
        k = min(outlier_neighbors, len(class_idx) - 1)
        lof = LocalOutlierFactor(n_neighbors=k, metric='cosine')
        lof.fit_predict(class_embs)
        
        # -negative_outlier_factor_ yields positive scores where > 1.5 is standard outlier
        cls_lof_scores = -lof.negative_outlier_factor_
        
        for local_idx, idx in enumerate(class_idx):
            outlier_scores_knn[idx] = float(cls_lof_scores[local_idx])
            if cls_lof_scores[local_idx] >= lof_threshold:
                is_outlier[idx] = True
                
        # Also compute standard class centroid distance as secondary check
        centroid = np.mean(class_embs, axis=0)
        centroid = centroid / (np.linalg.norm(centroid) + 1e-9)
        centroid_similarities = np.dot(class_embs, centroid)
        
        for local_idx, idx in enumerate(class_idx):
            outlier_scores_centroid[idx] = float(centroid_similarities[local_idx])
            
    # 3. Likely-Mislabeled Detection (KNN consensus)
    print("Detecting suspected mislabels...")
    mislabeled_items = []
    
    for i in range(N):
        # Calculate cosine similarity of query i to all other items
        sims = np.dot(embeddings, embeddings[i])
        
        # Sort indices descending, excluding own index at 0
        sorted_indices = np.argsort(sims)[::-1]
        sorted_indices = sorted_indices[sorted_indices != i]
        
        # Get top-k nearest neighbors
        knn_indices = sorted_indices[:mislabel_k]
        knn_labels = labels[knn_indices]
        
        # Check own label consensus
        own_label = labels[i]
        same_label_count = np.sum(knn_labels == own_label)
        own_ratio = same_label_count / mislabel_k
        
        if own_ratio < mislabel_consensus:
            is_mislabeled[i] = True
            
            # Find majority class among neighbors
            classes, counts = np.unique(knn_labels, return_counts=True)
            suggested_label = classes[np.argmax(counts)]
            
            # Extract nearest neighbors detail for report
            neighbors_detail = []
            for n_idx in knn_indices:
                neighbors_detail.append({
                    "id": ids[n_idx],
                    "label": str(labels[n_idx]),
                    "sim": float(sims[n_idx]),
                    "path": crops_metadata[n_idx]["crop_path"]
                })
                
            mislabeled_items.append({
                "index": i,
                "id": ids[i],
                "label": str(own_label),
                "suggested_label": str(suggested_label),
                "reason": f"Only {same_label_count}/{mislabel_k} neighbors are '{own_label}'. Suggested: '{suggested_label}'.",
                "neighbors": neighbors_detail
            })
            
    # Combine outlier info into output lists
    outliers_all = []
    for i in range(N):
        if is_outlier[i]:
            outliers_all.append({
                "index": i,
                "id": ids[i],
                "label": str(labels[i]),
                "path": crops_metadata[i]["crop_path"],
                "centroid_similarity": outlier_scores_centroid[i],
                "knn_similarity": 1.0 / (outlier_scores_knn[i] + 1e-9), # Map lof density score inversely
                "reason": f"Density outlier (LOF score: {outlier_scores_knn[i]:.2f}).",
                "is_centroid": False,
                "is_knn": True
            })
            
    # Format individual items summaries
    items_summary = []
    for i in range(N):
        items_summary.append({
            "index": i,
            "id": ids[i],
            "label": str(labels[i]),
            "path": crops_metadata[i]["crop_path"],
            "original_image_path": crops_metadata[i]["original_image_path"],
            "crop_idx": crops_metadata[i]["crop_idx"],
            "bbox_norm": crops_metadata[i]["bbox_norm"],
            "centroid_similarity": outlier_scores_centroid[i],
            "knn_similarity": outlier_scores_knn[i], # Store raw LOF score
            "is_duplicate": is_duplicate[i],
            "is_outlier": is_outlier[i],
            "is_mislabeled": is_mislabeled[i],
            "suggested_label": next((m["suggested_label"] for m in mislabeled_items if m["index"] == i), ""),
            "mislabel_reason": next((m["reason"] for m in mislabeled_items if m["index"] == i), "")
        })
        
    # Return consolidated dictionary
    clean_indices = set(range(N)) - duplicate_indices - set(idx for idx in range(N) if is_outlier[idx]) - set(idx for idx in range(N) if is_mislabeled[idx])
    
    # Reconstruct class names list ordered by class_id
    max_class_id = max(m["class_id"] for m in crops_metadata)
    class_names = [f"class_{i}" for i in range(max_class_id + 1)]
    for m in crops_metadata:
        class_names[m["class_id"]] = m["class_name"]
        
    report = {
        "summary": {
            "total_samples": N,
            "duplicate_count": len(duplicate_indices),
            "duplicate_pairs_count": len(duplicate_pairs),
            "outlier_count": len(outliers_all),
            "mislabel_count": len(mislabeled_items),
            "clean_count": len(clean_indices),
            "duplicate_threshold": dup_threshold,
            "outlier_threshold_centroid": 0.85, # Visual tuner defaults
            "outlier_threshold_knn": lof_threshold,
            "mislabel_k": mislabel_k
        },
        "class_names": class_names,
        "duplicates": duplicate_pairs,
        "outliers": outliers_all,
        "mislabels": mislabeled_items,
        "items": items_summary,
        "embeddings": embeddings.tolist()
    }
    
    return report

def write_reports(report, crops_metadata, output_dir):
    """
    Saves curation outputs to CSV, JSON, JS, and TXT actions list.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Output CSV Curation List
    csv_rows = []
    for item in report["items"]:
        meta = crops_metadata[item["index"]]
        
        # Decide Action recommendation
        action = "KEEP"
        reason = "Clean"
        
        if item["is_duplicate"]:
            action = "DISCARD"
            reason = "Near-duplicate"
        elif item["is_outlier"]:
            action = "DISCARD"
            reason = "Density outlier (LOF)"
        elif item["is_mislabeled"]:
            action = "DISCARD" # Recommend discard or edit
            reason = f"Suspected mislabel (Suggest: {item['suggested_label']})"
            
        csv_rows.append({
            "crop_id": item["id"],
            "class_label": item["label"],
            "original_image": meta["original_image_name"],
            "split": meta["split"],
            "action_recommendation": action,
            "reason": reason,
            "is_duplicate": item["is_duplicate"],
            "is_outlier": item["is_outlier"],
            "is_mislabeled": item["is_mislabeled"],
            "suggested_label": item["suggested_label"],
            "lof_score": item["knn_similarity"],
            "centroid_similarity": item["centroid_similarity"]
        })
        
    df = pd.DataFrame(csv_rows)
    csv_path = os.path.join(output_dir, "curation_report.csv")
    df.to_csv(csv_path, index=False)
    print(f"CSV report written to {csv_path}")
    
    # 2. Output text audit actions file
    txt_path = os.path.join(output_dir, "curation_actions.txt")
    with open(txt_path, 'w') as f:
        f.write("=== CleanCrop Action Curation Log ===\n")
        f.write(f"Total Crops Evaluated: {report['summary']['total_samples']}\n")
        f.write(f"Keep Recommendations: {report['summary']['clean_count']}\n")
        f.write(f"Discard Recommendations: {report['summary']['total_samples'] - report['summary']['clean_count']}\n\n")
        
        for row in csv_rows:
            f.write(f"{row['crop_id']}: {row['action_recommendation']} - {row['reason']}\n")
    print(f"Action instructions file written to {txt_path}")
    
    # 3. Create interactive dashboard workspace inside output_dir
    demo_dir = os.path.join(output_dir, "demo_dataset")
    os.makedirs(demo_dir, exist_ok=True)
    
    # Save curation_report.json
    json_path = os.path.join(demo_dir, "curation_report.json")
    with open(json_path, 'w') as f:
        json.dump(report, f, indent=2)
        
    # Save curation_report.js
    js_path = os.path.join(demo_dir, "curation_report.js")
    with open(js_path, 'w') as f:
        f.write(f"window.CURATION_REPORT = {json.dumps(report, indent=2)};\n")
        
    # Copy dashboard source files from active workspace to output folder
    src_files = ['index.html', 'styles.css', 'app.js']
    for sf in src_files:
        src_path = os.path.join(os.getcwd(), sf)
        if os.path.exists(src_path):
            shutil.copy2(src_path, os.path.join(output_dir, sf))
            
    print(f"Curation reports and dashboard copied to '{output_dir}'.")
    
    # Print short summary to console
    print("\n" + "="*40)
    print("           CURATION SUMMARY")
    print("="*40)
    print(f"Total samples processed:  {report['summary']['total_samples']}")
    print(f"Clean samples (KEEP):     {report['summary']['clean_count']}")
    print(f"Flagged duplicates:       {report['summary']['duplicate_count']} ({report['summary']['duplicate_pairs_count']} pairs)")
    print(f"Flagged density outliers: {report['summary']['outlier_count']}")
    print(f"Suspected wrong labels:   {report['summary']['mislabel_count']}")
    print(f"Dashboard view:           Open '{os.path.join(output_dir, 'index.html')}' in your browser!")
    print("="*40 + "\n")

GLOBAL_CLASS_NAMES = []

def process_writeback_actions(actions):
    """
    Applies writeback actions to label files.
    Each action in actions is a dict:
    {
      "original_image_path": str,
      "crop_idx": int,
      "bbox_norm": [x, y, w, h],
      "action": "DELETE" or "RELABEL",
      "new_class_id": int (optional)
    }
    """
    import os
    from collections import defaultdict
    global GLOBAL_CLASS_NAMES
    
    # Try loading from common paths if empty
    if not GLOBAL_CLASS_NAMES:
        for path in ["auto/road_traffic", "demo_dataset", "."]:
            yaml_path = os.path.join(path, "data.yaml")
            if os.path.exists(yaml_path):
                try:
                    GLOBAL_CLASS_NAMES = load_yaml_config(yaml_path)
                    break
                except Exception:
                    pass
                    
    actions_by_file = defaultdict(list)
    
    for act in actions:
        img_path = act["original_image_path"]
        # Determine label path
        dir_name, img_file = os.path.split(img_path)
        parent_dir = os.path.dirname(dir_name)
        base_name, _ = os.path.splitext(img_file)
        label_path = os.path.join(parent_dir, 'labels', base_name + '.txt')
        actions_by_file[label_path].append(act)
        
    results = []
    for label_path, file_actions in actions_by_file.items():
        if not os.path.exists(label_path):
            results.append({
                "label_path": label_path,
                "status": "error",
                "error": f"Label file does not exist at {label_path}"
            })
            continue
            
        try:
            with open(label_path, 'r') as f:
                lines = f.readlines()
                
            # Parse lines into structured objects
            structured_lines = []
            for line_idx, line in enumerate(lines):
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                structured_lines.append({
                    "original_idx": line_idx,
                    "class_id": int(parts[0]),
                    "bbox": [float(p) for p in parts[1:5]],
                    "original_line": line
                })
                
            # Mark which ones to delete or modify
            deleted_indices = set()
            modified_lines = {} # original_idx -> new class_id
            
            for act in file_actions:
                target_idx = act["crop_idx"]
                bbox_norm = act["bbox_norm"]
                action_type = act["action"]
                
                # Find matching line by index first, fall back to bbox matching
                matched_line = None
                if target_idx < len(structured_lines):
                    candidate = structured_lines[target_idx]
                    # Check bbox similarity to be safe
                    diffs = [abs(a - b) for a, b in zip(candidate["bbox"], bbox_norm)]
                    if max(diffs) < 0.05:
                        matched_line = candidate
                        
                if matched_line is None:
                    # Search all lines for closest bbox match
                    best_match = None
                    min_diff = 1.0
                    for line_obj in structured_lines:
                        diffs = [abs(a - b) for a, b in zip(line_obj["bbox"], bbox_norm)]
                        max_diff = max(diffs)
                        if max_diff < min_diff:
                            min_diff = max_diff
                            best_match = line_obj
                    if min_diff < 0.05:
                        matched_line = best_match
                        
                if matched_line is not None:
                    orig_idx = matched_line["original_idx"]
                    if action_type == "DELETE":
                        deleted_indices.add(orig_idx)
                    elif action_type == "RELABEL":
                        new_class_val = act.get("new_class_id")
                        if isinstance(new_class_val, str):
                            if new_class_val in GLOBAL_CLASS_NAMES:
                                new_class_id = GLOBAL_CLASS_NAMES.index(new_class_val)
                            else:
                                try:
                                    new_class_id = int(new_class_val)
                                except ValueError:
                                    new_class_id = 0
                        else:
                            try:
                                new_class_id = int(new_class_val)
                            except (ValueError, TypeError):
                                new_class_id = 0
                        modified_lines[orig_idx] = new_class_id
                else:
                    results.append({
                        "label_path": label_path,
                        "status": "warning",
                        "warning": f"Could not find matching box coordinates {bbox_norm}"
                    })
                    
            # Reconstruct the label file content
            new_lines = []
            for line_idx, line_obj in enumerate(structured_lines):
                orig_idx = line_obj["original_idx"]
                if orig_idx in deleted_indices:
                    continue
                if orig_idx in modified_lines:
                    new_class_id = modified_lines[orig_idx]
                    bbox_str = " ".join(f"{v:.6f}" for v in line_obj["bbox"])
                    new_lines.append(f"{new_class_id} {bbox_str}\n")
                else:
                    new_lines.append(line_obj["original_line"])
                    
            # Write back
            with open(label_path, 'w') as f:
                f.writelines(new_lines)
                
            results.append({
                "label_path": label_path,
                "status": "success",
                "deleted_count": len(deleted_indices),
                "modified_count": len(modified_lines)
            })
        except Exception as ex:
            results.append({
                "label_path": label_path,
                "status": "error",
                "error": str(ex)
            })
            
    return results

import http.server
import json

class CleanCropRequestHandler(http.server.SimpleHTTPRequestHandler):
    def do_POST(self):
        if self.path == '/api/writeback':
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            try:
                data = json.loads(post_data.decode('utf-8'))
                actions = data.get('actions', [])
                
                # Apply writeback actions
                results = process_writeback_actions(actions)
                
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                
                response = {
                    "status": "success",
                    "message": f"Successfully processed {len(actions)} writeback actions.",
                    "results": results
                }
                self.wfile.write(json.dumps(response).encode('utf-8'))
            except Exception as e:
                self.send_response(500)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                response = {
                    "status": "error",
                    "message": str(e)
                }
                self.wfile.write(json.dumps(response).encode('utf-8'))
        else:
            self.send_response(404)
            self.end_headers()
            
    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'POST, GET, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()

def main():
    global GLOBAL_CLASS_NAMES
    parser = argparse.ArgumentParser(description="CleanCrop End-to-End YOLO Curation Pipeline")
    parser.add_argument("--dataset-dir", type=str, default="auto/road_traffic", help="Path to YOLO dataset folder")
    parser.add_argument("--output-dir", type=str, default="road_traffic_curated", help="Path to output curation folder")
    parser.add_argument("--model", type=str, choices=["clip", "dinov2"], default="dinov2", help="Feature extractor mode (clip or dinov2)")
    parser.add_argument("--batch-size", type=int, default=32, help="Extraction batch size")
    parser.add_argument("--dup-threshold", type=float, default=0.95, help="Similarity threshold for near-duplicate detection")
    parser.add_argument("--lof-threshold", type=float, default=1.3, help="Local Outlier Factor score threshold for outliers")
    parser.add_argument("--knn-k", type=int, default=5, help="K parameter for mislabeled KNN classification consensus")
    parser.add_argument("--serve", action="store_true", help="Start a local HTTP server to view the interactive dashboard")
    parser.add_argument("--serve-only", action="store_true", help="Only start the local HTTP server to view the dashboard without running curation")
    parser.add_argument("--port", type=int, default=8000, help="Port for the local HTTP server")
    
    args = parser.parse_args()

    if args.serve_only:
        import http.server
        import socketserver
        
        yaml_path = os.path.join(args.dataset_dir, "data.yaml")
        if os.path.exists(yaml_path):
            try:
                GLOBAL_CLASS_NAMES = load_yaml_config(yaml_path)
            except Exception:
                pass
                
        # Change directory to the output folder containing the dashboard files
        os.chdir(os.path.abspath(args.output_dir))
        
        Handler = CleanCropRequestHandler
        socketserver.TCPServer.allow_reuse_address = True
        
        print(f"\n[CleanCrop] Starting local dashboard server at http://localhost:{args.port}/index.html")
        print("[CleanCrop] Press Ctrl+C to terminate the server.")
        
        try:
            with socketserver.TCPServer(("", args.port), Handler) as httpd:
                httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n[CleanCrop] Server stopped.")
            sys.exit(0)
        sys.exit(0)
        
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 1. Parse YAML config
    yaml_path = os.path.join(args.dataset_dir, "data.yaml")
    if not os.path.exists(yaml_path):
        print(f"Error: YOLO data.yaml configuration file not found at {yaml_path}!")
        sys.exit(1)
        
    GLOBAL_CLASS_NAMES = load_yaml_config(yaml_path)
    class_names = GLOBAL_CLASS_NAMES
    print(f"Classes list successfully loaded: {class_names}")
    
    # 2. Scan dataset images and labels
    crops_metadata = parse_yolo_dataset(args.dataset_dir, class_names)
    if not crops_metadata:
        print("No annotations found in labels splits directory!")
        sys.exit(1)
        
    # 3. Crop and resize images
    output_crops_dir = os.path.join(args.output_dir, "demo_dataset", "crops")
    valid_metadata = crop_and_resize_images(crops_metadata, output_crops_dir)
    if not valid_metadata:
        print("Failed to crop images!")
        sys.exit(1)
        
    # 4. Extract embeddings
    embeddings = extract_embeddings(valid_metadata, model_mode=args.model, batch_size=args.batch_size, device=device)
    
    # 5. Detect anomalies
    report = detect_quality_anomalies(
        embeddings, 
        valid_metadata,
        dup_threshold=args.dup_threshold,
        outlier_neighbors=20,
        lof_threshold=args.lof_threshold,
        mislabel_k=args.knn_k,
        mislabel_consensus=0.5
    )
    
    # 6. Save reports and copy dashboard
    write_reports(report, valid_metadata, args.output_dir)

    if args.serve:
        import http.server
        import socketserver
        
        # Change directory to the output folder containing the dashboard files
        os.chdir(os.path.abspath(args.output_dir))
        
        Handler = CleanCropRequestHandler
        socketserver.TCPServer.allow_reuse_address = True
        
        print(f"\n[CleanCrop] Starting local dashboard server at http://localhost:{args.port}/index.html")
        print("[CleanCrop] Press Ctrl+C to terminate the server.")
        
        try:
            with socketserver.TCPServer(("", args.port), Handler) as httpd:
                httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n[CleanCrop] Server stopped.")

if __name__ == "__main__":
    main()
